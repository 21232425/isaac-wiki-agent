from __future__ import annotations

import argparse
import bz2
import gzip
import html
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO, Iterator

from tools.wiki_store import StoredPage, count_pages, database_path, upsert_pages


ARCHIVE_DUMP_URL = (
    "https://archive.org/download/wiki-bindingofisaacrebirth.wiki.gg-20240831/"
    "bindingofisaacrebirth.wiki.gg-20240831-history.xml.zst"
)
ENGLISH_API_URL = "https://bindingofisaacrebirth.wiki.gg/api.php"
CHINESE_API_URL = "https://bindingofisaacrebirth.wiki.gg/zh/api.php"
USER_AGENT = "IsaacWikiAgent/1.0 (offline knowledge base builder)"


def sync_api(api_url: str, source: str, delay: float = 0.1, max_pages: int | None = None) -> int:
    continuation: dict[str, str] = {}
    imported = 0
    while True:
        list_params = {
            "action": "query",
            "list": "allpages",
            "apnamespace": "0",
            "apfilterredir": "nonredirects",
            "aplimit": "500",
            "format": "json",
            "formatversion": "2",
            **continuation,
        }
        listing = _request_json(api_url, list_params)
        allpages = listing.get("query", {}).get("allpages", [])
        for start in range(0, len(allpages), 50):
            rows = allpages[start : start + 50]
            if max_pages is not None:
                rows = rows[: max_pages - imported]
            if not rows:
                return imported

            content = _request_json(
                api_url,
                {
                    "action": "query",
                    "pageids": "|".join(str(row["pageid"]) for row in rows),
                    "prop": "revisions",
                    "rvprop": "ids|timestamp|content",
                    "rvslots": "main",
                    "format": "json",
                    "formatversion": "2",
                },
            )
            batch: list[StoredPage] = []
            for page in content.get("query", {}).get("pages", []):
                revisions = page.get("revisions", [])
                revision = revisions[0] if revisions else {}
                wikitext = revision.get("slots", {}).get("main", {}).get("content", "")
                extract = _wikitext_to_text(wikitext)
                title = page.get("title", "").strip()
                if not title or not extract:
                    continue
                batch.append(
                    StoredPage(
                        title=title,
                        extract=extract,
                        url=_page_url(api_url, title),
                        source=source,
                        pageid=page.get("pageid"),
                    )
                )

            imported += upsert_pages(batch)
            print(f"已同步 {imported} 页（数据库共 {count_pages()} 页）", flush=True)
            if max_pages is not None and imported >= max_pages:
                return imported
            time.sleep(max(0.0, delay))

        continuation = listing.get("continue", {})
        if not continuation:
            return imported


def download_dump(url: str, output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response, output.open("wb") as target:
        total = int(response.headers.get("Content-Length", "0"))
        downloaded = 0
        while chunk := response.read(1024 * 1024):
            target.write(chunk)
            downloaded += len(chunk)
            if total:
                print(f"下载进度：{downloaded / total:.1%}", end="\r", flush=True)
    print(f"已下载：{output}")
    return output


def import_xml_dump(path: Path, source: str, base_url: str) -> int:
    imported = 0
    batch: list[StoredPage] = []
    with _open_dump(path) as stream:
        context = ET.iterparse(stream, events=("end",))
        for _event, element in context:
            if _local_name(element.tag) != "page":
                continue
            page = _parse_xml_page(element, source=source, base_url=base_url)
            element.clear()
            if page is None:
                continue
            batch.append(page)
            if len(batch) >= 250:
                imported += upsert_pages(batch)
                batch.clear()
                print(f"已导入 {imported} 页", flush=True)

    imported += upsert_pages(batch)
    print(f"导入完成：{imported} 页；数据库共 {count_pages()} 页")
    return imported


@contextmanager
def _open_dump(path: Path) -> Iterator[BinaryIO]:
    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith(".bz2"):
        with bz2.open(path, "rb") as stream:
            yield stream
        return
    if suffixes.endswith(".gz"):
        with gzip.open(path, "rb") as stream:
            yield stream
        return
    if suffixes.endswith(".zst"):
        try:
            import zstandard
        except ImportError as exc:
            raise RuntimeError("导入 .zst 文件需要先安装 zstandard。") from exc
        with path.open("rb") as compressed:
            frame_parameters = zstandard.get_frame_parameters(compressed.read(64))
            compressed.seek(0)
            decompressor = zstandard.ZstdDecompressor(
                max_window_size=frame_parameters.window_size
            )
            with decompressor.stream_reader(compressed) as stream:
                yield stream
        return
    with path.open("rb") as stream:
        yield stream


def _parse_xml_page(element: ET.Element, source: str, base_url: str) -> StoredPage | None:
    title = _child_text(element, "title")
    namespace = _child_text(element, "ns")
    pageid_text = _child_text(element, "id")
    if namespace != "0" or not title:
        return None

    latest_revision: tuple[str, str] | None = None
    for child in element:
        if _local_name(child.tag) != "revision":
            continue
        timestamp = _child_text(child, "timestamp")
        text = _child_text(child, "text")
        if text and (latest_revision is None or timestamp >= latest_revision[0]):
            latest_revision = (timestamp, text)
    if latest_revision is None:
        return None

    extract = _wikitext_to_text(latest_revision[1])
    if not extract:
        return None
    return StoredPage(
        title=title,
        extract=extract,
        url=base_url.rstrip("/") + "/" + urllib.parse.quote(title.replace(" ", "_")),
        source=source,
        pageid=int(pageid_text) if pageid_text.isdigit() else None,
    )


def _wikitext_to_text(text: str) -> str:
    text = re.sub(r"<!--.*?-->", " ", text, flags=re.DOTALL)
    text = re.sub(r"<ref\b[^>]*>.*?</ref\s*>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<ref\b[^>]*/\s*>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\{\|.*?\|\}", " ", text, flags=re.DOTALL)
    previous = None
    while previous != text:
        previous = text
        text = re.sub(r"\{\{[^{}]*\}\}", " ", text, flags=re.DOTALL)
    text = re.sub(r"\[\[(?:File|Image):[^\]]+\]\]", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\[\[[^\]|]+\|([^\]]+)\]\]", r"\1", text)
    text = re.sub(r"\[\[([^\]]+)\]\]", r"\1", text)
    text = re.sub(r"\[https?://\S+\s+([^\]]+)\]", r"\1", text)
    text = re.sub(r"\[https?://[^\]]+\]", " ", text)
    text = re.sub(r"={2,6}\s*(.*?)\s*={2,6}", r"\n\1\n", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("'''", "").replace("''", "")
    return _normalize_text(html.unescape(text))


def _request_json(api_url: str, params: dict[str, str]) -> dict:
    request = urllib.request.Request(
        f"{api_url}?{urllib.parse.urlencode(params)}",
        headers={"Accept": "application/json", "User-Agent": USER_AGENT},
    )
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=40) as response:
                charset = response.headers.get_content_charset() or "utf-8"
                return json.loads(response.read().decode(charset, errors="replace"))
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"MediaWiki API 请求失败：{last_error}") from last_error


def _page_url(api_url: str, title: str) -> str:
    script_path = api_url.removesuffix("/api.php")
    return f"{script_path}/wiki/{urllib.parse.quote(title.replace(' ', '_'))}"


def _child_text(element: ET.Element, name: str) -> str:
    for child in element:
        if _local_name(child.tag) == name:
            return child.text or ""
    return ""


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="构建与更新本地以撒 Wiki SQLite 数据库。")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("status", help="显示本地数据库路径和页面数量")

    sync_parser = subparsers.add_parser("sync-api", help="从公开 MediaWiki API 同步正文")
    sync_parser.add_argument("--api", default=ENGLISH_API_URL)
    sync_parser.add_argument("--source", default="wiki.gg-en-api")
    sync_parser.add_argument("--delay", type=float, default=0.1)
    sync_parser.add_argument("--max-pages", type=int)

    bootstrap_parser = subparsers.add_parser("bootstrap", help="同步 wiki.gg 英文和中文分站")
    bootstrap_parser.add_argument("--delay", type=float, default=0.1)

    download_parser = subparsers.add_parser("download-dump", help="下载 Internet Archive XML 转储")
    download_parser.add_argument("--url", default=ARCHIVE_DUMP_URL)
    download_parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/downloads/bindingofisaacrebirth-20240831-history.xml.zst"),
    )

    import_parser = subparsers.add_parser("import-dump", help="导入 MediaWiki XML 转储")
    import_parser.add_argument("path", type=Path)
    import_parser.add_argument("--source", default="wiki.gg-en-archive-20240831")
    import_parser.add_argument("--base-url", default="https://bindingofisaacrebirth.wiki.gg/wiki")

    args = parser.parse_args()
    if args.command == "status":
        print(f"数据库：{database_path()}")
        print(f"页面数：{count_pages()}")
    elif args.command == "sync-api":
        sync_api(args.api, args.source, delay=args.delay, max_pages=args.max_pages)
    elif args.command == "bootstrap":
        sync_api(ENGLISH_API_URL, "wiki.gg-en-api", delay=args.delay)
        sync_api(CHINESE_API_URL, "wiki.gg-zh-api", delay=args.delay)
    elif args.command == "download-dump":
        download_dump(args.url, args.output)
    elif args.command == "import-dump":
        import_xml_dump(args.path, args.source, args.base_url)


if __name__ == "__main__":
    main()
