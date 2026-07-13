from __future__ import annotations

import os
import re
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATABASE_PATH = PROJECT_ROOT / "data" / "isaac_wiki.sqlite3"


@dataclass
class StoredPage:
    title: str
    extract: str
    url: str
    source: str
    pageid: int | None = None


def database_path() -> Path:
    configured = os.getenv("ISAAC_WIKI_DB")
    return Path(configured).expanduser().resolve() if configured else DEFAULT_DATABASE_PATH


def count_pages() -> int:
    with _connect() as connection:
        row = connection.execute("SELECT COUNT(*) FROM pages").fetchone()
    return int(row[0])


def get_page(title: str) -> StoredPage | None:
    normalized = _normalize_title(title)
    if not normalized:
        return None

    with _connect() as connection:
        row = connection.execute(
            """
            SELECT title, extract, url, source, pageid
            FROM pages
            WHERE title = ? COLLATE NOCASE
            ORDER BY
                CASE
                    WHEN source LIKE '%wiki.gg-zh-api%'
                        OR source LIKE '%wiki.gg/zh/api.php%' THEN 0
                    WHEN source LIKE '%wiki.gg-en-api%'
                        OR source LIKE '%wiki.gg/api.php%' THEN 1
                    ELSE 2
                END,
                updated_at DESC
            LIMIT 1
            """,
            (normalized,),
        ).fetchone()
    return _row_to_page(row) if row else None


def search_pages(query: str, limit: int = 5) -> list[StoredPage]:
    query = re.sub(r"\s+", " ", query).strip()
    if not query or limit <= 0:
        return []

    candidates: dict[int, sqlite3.Row] = {}
    with _connect() as connection:
        for row in _search_fts(connection, query, max(limit * 8, 40)):
            candidates[int(row["id"])] = row

        tokens = _search_tokens(query)
        if tokens:
            clauses = []
            parameters: list[str | int] = []
            for token in tokens[:8]:
                clauses.append("(title LIKE ? OR extract LIKE ?)")
                pattern = f"%{token}%"
                parameters.extend((pattern, pattern))
            parameters.append(max(limit * 16, 80))
            rows = connection.execute(
                f"""
                SELECT id, title, extract, url, source, pageid
                FROM pages
                WHERE {' OR '.join(clauses)}
                LIMIT ?
                """,
                parameters,
            ).fetchall()
            for row in rows:
                candidates[int(row["id"])] = row

    ranked = sorted(
        candidates.values(),
        key=lambda row: (
            _relevance_score(row["title"], row["extract"], query),
            _source_score(row["source"]),
        ),
        reverse=True,
    )
    results: list[StoredPage] = []
    seen_titles: set[str] = set()
    for row in ranked:
        title_key = row["title"].casefold()
        if title_key in seen_titles:
            continue
        seen_titles.add(title_key)
        results.append(_row_to_page(row))
        if len(results) >= limit:
            break
    return results


def upsert_page(page: StoredPage) -> None:
    upsert_pages([page])


def upsert_pages(pages: list[StoredPage]) -> int:
    clean_pages = [page for page in pages if page.title.strip() and page.extract.strip()]
    if not clean_pages:
        return 0

    now = datetime.now(UTC).isoformat()
    rows = [
        (
            _normalize_title(page.title),
            page.extract.strip(),
            page.url,
            page.source,
            page.pageid,
            now,
        )
        for page in clean_pages
    ]
    with _connect() as connection:
        connection.executemany(
            """
            INSERT INTO pages(title, extract, url, source, pageid, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(title, source) DO UPDATE SET
                extract = excluded.extract,
                url = excluded.url,
                pageid = excluded.pageid,
                updated_at = excluded.updated_at
            """,
            rows,
        )
    return len(rows)


def _connect() -> sqlite3.Connection:
    path = database_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    _ensure_schema(connection)
    return connection


def _ensure_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS pages (
            id INTEGER PRIMARY KEY,
            title TEXT NOT NULL COLLATE NOCASE,
            extract TEXT NOT NULL,
            url TEXT NOT NULL,
            source TEXT NOT NULL,
            pageid INTEGER,
            updated_at TEXT NOT NULL,
            UNIQUE(title, source)
        );

        CREATE INDEX IF NOT EXISTS idx_pages_title ON pages(title COLLATE NOCASE);
        CREATE INDEX IF NOT EXISTS idx_pages_pageid ON pages(pageid);

        CREATE VIRTUAL TABLE IF NOT EXISTS pages_fts USING fts5(
            title,
            extract,
            content='pages',
            content_rowid='id',
            tokenize='unicode61 remove_diacritics 2'
        );

        CREATE TRIGGER IF NOT EXISTS pages_ai AFTER INSERT ON pages BEGIN
            INSERT INTO pages_fts(rowid, title, extract)
            VALUES (new.id, new.title, new.extract);
        END;

        CREATE TRIGGER IF NOT EXISTS pages_ad AFTER DELETE ON pages BEGIN
            INSERT INTO pages_fts(pages_fts, rowid, title, extract)
            VALUES ('delete', old.id, old.title, old.extract);
        END;

        CREATE TRIGGER IF NOT EXISTS pages_au AFTER UPDATE ON pages BEGIN
            INSERT INTO pages_fts(pages_fts, rowid, title, extract)
            VALUES ('delete', old.id, old.title, old.extract);
            INSERT INTO pages_fts(rowid, title, extract)
            VALUES (new.id, new.title, new.extract);
        END;
        """
    )


def _search_fts(
    connection: sqlite3.Connection,
    query: str,
    limit: int,
) -> list[sqlite3.Row]:
    tokens = _search_tokens(query)
    if not tokens:
        return []
    expression = " OR ".join(f'"{token.replace(chr(34), chr(34) * 2)}"' for token in tokens[:8])
    try:
        return connection.execute(
            """
            SELECT pages.id, pages.title, pages.extract, pages.url, pages.source, pages.pageid
            FROM pages_fts
            JOIN pages ON pages.id = pages_fts.rowid
            WHERE pages_fts MATCH ?
            ORDER BY bm25(pages_fts, 8.0, 1.0)
            LIMIT ?
            """,
            (expression, limit),
        ).fetchall()
    except sqlite3.OperationalError:
        return []


def _search_tokens(query: str) -> list[str]:
    tokens = re.findall(r"[\w\u3400-\u9fff'’-]+", query, flags=re.UNICODE)
    return list(dict.fromkeys(token for token in tokens if token))


def _relevance_score(title: str, extract: str, query: str) -> tuple[int, int, int]:
    title_folded = title.casefold()
    query_folded = query.casefold()
    tokens = [token.casefold() for token in _search_tokens(query)]
    exact = int(title_folded == query_folded)
    title_phrase = int(query_folded in title_folded)
    title_hits = sum(token in title_folded for token in tokens)
    body_hits = sum(token in extract.casefold() for token in tokens)
    return exact, title_phrase * 10 + title_hits * 4 + body_hits, -len(title)


def _source_score(source: str) -> int:
    source = source.casefold()
    if "wiki.gg-zh-api" in source or "wiki.gg/zh/api.php" in source:
        return 3
    if "wiki.gg-en-api" in source or "wiki.gg/api.php" in source:
        return 2
    return 1


def _row_to_page(row: sqlite3.Row) -> StoredPage:
    return StoredPage(
        title=row["title"],
        extract=row["extract"],
        url=row["url"],
        source=row["source"],
        pageid=row["pageid"],
    )


def _normalize_title(title: str) -> str:
    return re.sub(r"\s+", " ", title.replace("_", " ")).strip()
