from __future__ import annotations

import html
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass

from tools.wiki_store import StoredPage, get_page, search_pages, upsert_page


DEFAULT_REMOTE_APIS = (
    "https://bindingofisaacrebirth.wiki.gg/zh/api.php",
    "https://bindingofisaacrebirth.wiki.gg/api.php",
)
USER_AGENT = "IsaacWikiAgent/1.0 (local knowledge base with MediaWiki fallback)"


@dataclass
class SearchResult:
    title: str
    snippet: str
    pageid: int | None = None
    page_url: str | None = None
    source: str = ""
    retrieved_from: str = "unknown"

    @property
    def url(self) -> str:
        return self.page_url or _page_url(DEFAULT_REMOTE_APIS[0], self.title)


@dataclass
class WikiPage:
    title: str
    extract: str
    url: str
    pageid: int | None = None
    source: str = ""
    retrieved_from: str = "unknown"


class WikiApiError(RuntimeError):
    """Raised when neither the local database nor a remote wiki returns content."""


def search_wiki(query: str, limit: int = 5) -> list[SearchResult]:
    """Search SQLite first and use a public MediaWiki API only on a local miss."""
    local_results = search_pages(query, limit=limit)
    if local_results:
        return [
            SearchResult(
                title=page.title,
                snippet=_matching_snippet(page.extract, query),
                pageid=page.pageid,
                page_url=page.url,
                source=page.source,
                retrieved_from="local_database",
            )
            for page in local_results
        ]

    errors: list[str] = []
    merged: list[SearchResult] = []
    seen: set[str] = set()
    for api_url in _remote_apis():
        try:
            payload = _request_json(
                api_url,
                {
                    "action": "query",
                    "list": "search",
                    "srsearch": query,
                    "srlimit": str(limit),
                    "format": "json",
                    "formatversion": "2",
                },
            )
        except WikiApiError as exc:
            errors.append(str(exc))
            continue

        for row in payload.get("query", {}).get("search", []):
            title = row.get("title", "")
            if not title or title.casefold() in seen:
                continue
            seen.add(title.casefold())
            merged.append(
                SearchResult(
                    title=title,
                    snippet=_clean_html(row.get("snippet", "")),
                    pageid=row.get("pageid"),
                    page_url=_page_url(api_url, title),
                    source=api_url,
                    retrieved_from="remote_api",
                )
            )
            if len(merged) >= limit:
                return merged

    if merged:
        return merged
    if errors:
        raise WikiApiError("；".join(errors))
    return []


def get_wiki_page(title: str) -> WikiPage:
    """Read an exact local page first; fetch and cache it only when absent."""
    local_page = get_page(title)
    if local_page is not None:
        return _stored_to_wiki_page(local_page, retrieved_from="local_database")

    errors: list[str] = []
    for api_url in _remote_apis():
        try:
            payload = _request_json(
                api_url,
                {
                    "action": "query",
                    "prop": "extracts",
                    "explaintext": "1",
                    "exsectionformat": "plain",
                    "redirects": "1",
                    "titles": title,
                    "format": "json",
                    "formatversion": "2",
                },
            )
        except WikiApiError as exc:
            errors.append(str(exc))
            continue

        pages = payload.get("query", {}).get("pages", [])
        if not pages or pages[0].get("missing"):
            continue
        page = pages[0]
        resolved_title = page.get("title", title)
        extract = _normalize_text(page.get("extract", ""))
        if not extract:
            continue
        stored = StoredPage(
            title=resolved_title,
            extract=extract,
            url=_page_url(api_url, resolved_title),
            source=api_url,
            pageid=page.get("pageid"),
        )
        upsert_page(stored)
        return _stored_to_wiki_page(stored, retrieved_from="remote_api")

    detail = f"（{'；'.join(errors)}）" if errors else ""
    raise WikiApiError(f"本地数据库和在线 Wiki 都没有页面：{title}{detail}")


def _request_json(api_url: str, params: dict[str, str]) -> dict:
    request = urllib.request.Request(
        f"{api_url}?{urllib.parse.urlencode(params)}",
        headers={"Accept": "application/json", "User-Agent": USER_AGENT},
    )

    last_error: Exception | None = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                charset = response.headers.get_content_charset() or "utf-8"
                return json.loads(response.read().decode(charset, errors="replace"))
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code < 500:
                break
        except Exception as exc:
            last_error = exc
        if attempt < 2:
            time.sleep(0.8 * (attempt + 1))

    if isinstance(last_error, urllib.error.HTTPError):
        raise WikiApiError(f"{api_url} 返回 HTTP {last_error.code} {last_error.reason}") from last_error
    if last_error is not None:
        raise WikiApiError(f"{api_url} 请求失败：{last_error}") from last_error
    raise WikiApiError(f"{api_url} 请求失败，原因未知")


def _remote_apis() -> tuple[str, ...]:
    if os.getenv("ISAAC_WIKI_OFFLINE", "").strip().casefold() in {"1", "true", "yes", "on"}:
        return ()
    configured = os.getenv("ISAAC_WIKI_REMOTE_APIS", "")
    if not configured.strip():
        return DEFAULT_REMOTE_APIS
    return tuple(url.strip() for url in configured.split(",") if url.strip())


def _page_url(api_url: str, title: str) -> str:
    script_path = api_url.removesuffix("/api.php")
    return f"{script_path}/wiki/{urllib.parse.quote(title.replace(' ', '_'))}"


def _stored_to_wiki_page(page: StoredPage, retrieved_from: str) -> WikiPage:
    return WikiPage(
        title=page.title,
        extract=page.extract,
        url=page.url,
        pageid=page.pageid,
        source=page.source,
        retrieved_from=retrieved_from,
    )


def _matching_snippet(text: str, query: str, max_chars: int = 260) -> str:
    folded = text.casefold()
    positions = [folded.find(token.casefold()) for token in re.findall(r"[\w\u3400-\u9fff'’-]+", query)]
    positions = [position for position in positions if position >= 0]
    start = max(0, (min(positions) if positions else 0) - 60)
    snippet = text[start : start + max_chars].replace("\n", " ").strip()
    if start:
        snippet = "..." + snippet
    if start + max_chars < len(text):
        snippet += "..."
    return snippet


def _clean_html(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text)
    return html.unescape(_normalize_text(text))


def _normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
