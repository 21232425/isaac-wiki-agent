from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass

from tools.llm import LlmApiError, LlmConfigError, is_llm_configured, summarize_with_llm
from tools.mediawiki import SearchResult, WikiApiError, WikiPage, get_wiki_page, search_wiki


MAX_QUOTE_CHARS = 900
MAX_LLM_CONTEXT_CHARS = 12000
MAX_SEARCH_RESULTS = 8
MAX_PAGES_TO_READ = 4


@dataclass
class AgentAnswer:
    question: str
    search_results: list[SearchResult]
    page: WikiPage | None
    pages: list[WikiPage]
    answer: str


class IsaacWikiAgent:
    """A small rule-based agent that searches and reads Isaac HuijiWiki pages."""

    def answer(self, question: str) -> AgentAnswer:
        queries = self._build_search_queries(question)
        try:
            results = self._search_many(queries)
        except WikiApiError as exc:
            return AgentAnswer(
                question=question,
                search_results=[],
                page=None,
                pages=[],
                answer=(
                    "暂时无法访问以撒 Wiki 的 API，因此不能完成在线查询。\n\n"
                    f"检索词：{', '.join(queries)}\n"
                    f"错误信息：{exc}\n\n"
                    "这在 NLR 项目里对应“信息源可用性核验”：如果数据源禁止程序访问、"
                    "需要登录或需要浏览器验证，就不能直接把它当成稳定数据库。"
                    "后续可以改用官方开放 API、授权接口、本地缓存或离线文档索引。"
                ),
            )

        if not results:
            return AgentAnswer(
                question=question,
                search_results=[],
                page=None,
                pages=[],
                answer=(
                    "没有在以撒 Wiki 中检索到明确结果。\n"
                    f"检索词：{', '.join(queries)}\n"
                    "建议换一个更具体的名称，例如道具、角色、成就、挑战、怪物、Boss 或机制名。"
                ),
            )

        pages = self._read_relevant_pages(results, direct_titles=queries[:2])
        if not pages:
            return AgentAnswer(
                question=question,
                search_results=results,
                page=None,
                pages=[],
                answer=(
                    "找到了相关搜索结果，但暂时无法读取页面正文。\n\n"
                    + self._format_search_results(results)
                ),
            )

        answer = self._compose_answer(question, pages, results)
        return AgentAnswer(
            question=question,
            search_results=results,
            page=pages[0],
            pages=pages,
            answer=answer,
        )

    def _build_search_queries(self, question: str) -> list[str]:
        raw = question.strip()
        cleaned = self._build_search_query(question)
        core = self._extract_core_entity(raw)
        queries: list[str] = []

        def add(value: str) -> None:
            value = re.sub(r"\s+", " ", value).strip()
            if value and value not in queries:
                queries.append(value)

        add(core)
        add(cleaned)
        add(raw)

        # Keep category words for wiki search. They are often useful for non-item pages.
        category_words = [
            "角色",
            "人物",
            "成就",
            "挑战",
            "挑战房",
            "Boss",
            "boss",
            "怪物",
            "敌人",
            "机制",
            "房间",
            "结局",
            "饰品",
            "卡牌",
            "符文",
            "道具",
        ]
        for word in category_words:
            if word in raw and cleaned and word not in cleaned:
                add(f"{cleaned} {word}")
                add(f"{word} {cleaned}")

        # Common English aliases should stay searchable instead of being swallowed by Chinese filler words.
        english_terms = re.findall(r"[A-Za-z][A-Za-z0-9'_-]*", raw)
        for term in english_terms:
            add(term)
            alias = self._alias_query(term)
            add(alias)

        return queries[:6]

    def _alias_query(self, term: str) -> str:
        aliases = {
            "lost": "游魂",
            "the lost": "游魂",
            "judas": "犹大",
            "isaac": "以撒",
            "magdalene": "抹大拉",
            "cain": "该隐",
            "eve": "夏娃",
            "samson": "参孙",
            "azazel": "阿撒泻勒",
            "lilith": "莉莉丝",
            "keeper": "店主",
            "apollyon": "亚玻伦",
            "bethany": "伯大尼",
            "jacob": "雅各",
            "esau": "以扫",
        }
        return aliases.get(term.strip().lower(), "")

    def _extract_core_entity(self, question: str) -> str:
        text = question.strip()
        text = re.sub(r"[？?！!。，,：:；;“”\"'（）()\[\]{}]", " ", text)
        intent_phrases = [
            "怎么解锁",
            "如何解锁",
            "怎么获得",
            "如何获得",
            "怎么打",
            "怎么过",
            "怎么玩",
            "是什么",
            "有什么效果",
            "有什么作用",
            "有什么机制",
            "介绍一下",
            "告诉我",
            "请问",
            "以撒的结合",
            "以撒",
            "wiki",
        ]
        for phrase in intent_phrases:
            text = text.replace(phrase, " ")
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _build_search_query(self, question: str) -> str:
        cleaned = question.strip()
        cleaned = re.sub(r"[？?！!。，,：:；;“”\"'（）()\[\]{}]", " ", cleaned)
        stop_words = [
            "以撒",
            "的结合",
            "忏悔",
            "重生",
            "wiki",
            "是什么",
            "有什么",
            "怎么",
            "如何",
            "查询",
            "介绍",
            "一下",
            "请问",
            "告诉我",
        ]
        for word in stop_words:
            cleaned = cleaned.replace(word, " ")
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned or question.strip()

    def _search_many(self, queries: list[str]) -> list[SearchResult]:
        merged: list[SearchResult] = []
        seen: set[str] = set()
        for query in queries:
            for result in search_wiki(query, limit=MAX_SEARCH_RESULTS):
                key = str(result.pageid) if result.pageid is not None else result.title
                if key in seen:
                    continue
                seen.add(key)
                if result.title == query:
                    merged.insert(0, result)
                else:
                    merged.append(result)
        return merged[:MAX_SEARCH_RESULTS]

    def _read_relevant_pages(self, results: list[SearchResult], direct_titles: list[str]) -> list[WikiPage]:
        pages: list[WikiPage] = []
        seen_titles: set[str] = set()
        for title in direct_titles:
            try:
                page = get_wiki_page(title)
            except WikiApiError:
                continue
            if page.title in seen_titles:
                continue
            pages.append(page)
            seen_titles.add(page.title)
            if len(pages) >= MAX_PAGES_TO_READ:
                return pages

        for result in results[:MAX_SEARCH_RESULTS]:
            try:
                page = get_wiki_page(result.title)
            except WikiApiError:
                continue
            if page.title in seen_titles:
                continue
            pages.append(page)
            seen_titles.add(page.title)
            if len(pages) >= MAX_PAGES_TO_READ:
                break
        return pages

    def _compose_answer(
        self,
        question: str,
        pages: list[WikiPage],
        results: list[SearchResult],
    ) -> str:
        context = self._build_multi_page_context(question, pages, max_chars=MAX_LLM_CONTEXT_CHARS)
        summary = self._build_fallback_summary(question, pages)
        if is_llm_configured():
            try:
                llm_answer = summarize_with_llm(
                    question=question,
                    title="多个以撒 Wiki 页面",
                    url="见下方来源列表",
                    context=context,
                )
                return (
                    f"{llm_answer}\n\n"
                    "检索到的其他候选结果：\n"
                    + self._format_search_results(results)
                )
            except (LlmConfigError, LlmApiError) as exc:
                return (
                    "已检索到 Wiki 页面，但调用大模型总结失败，因此退回页面摘录模式。\n\n"
                    f"大模型错误：{exc}\n\n"
                    f"页面摘录：\n{summary}\n\n"
                    "已读取来源：\n"
                    + self._format_pages(pages)
                    + "\n\n"
                    "其他候选结果：\n"
                    + self._format_search_results(results)
                )

        return (
            f"问题：{question}\n\n"
            "我在以撒 Wiki 中检索到以下相关页面：\n"
            + self._format_pages(pages)
            + "\n\n"
            f"页面摘录：\n{summary}\n\n"
            "说明：当前未检测到 OPENAI_API_KEY，所以先返回检索到的页面摘录。"
            "配置 OPENAI_API_KEY 后，程序会自动调用大模型生成总结回答。\n\n"
            "其他候选结果：\n"
            + self._format_search_results(results)
        )

    def _build_multi_page_context(self, question: str, pages: list[WikiPage], max_chars: int) -> str:
        blocks: list[str] = []
        total = 0
        per_page_limit = max(1200, max_chars // max(1, len(pages)))
        for index, page in enumerate(pages, start=1):
            excerpt = self._extract_relevant_excerpt(question, page.extract, max_chars=per_page_limit)
            block = f"[来源 {index}]\n标题：{page.title}\n链接：{page.url}\n内容：\n{excerpt}\n"
            if total + len(block) > max_chars and blocks:
                break
            blocks.append(block)
            total += len(block)
        return "\n---\n".join(blocks)

    def _build_fallback_summary(self, question: str, pages: list[WikiPage]) -> str:
        excerpts = []
        for page in pages:
            excerpt = self._extract_relevant_excerpt(question, page.extract, max_chars=MAX_QUOTE_CHARS // 2)
            excerpts.append(f"## {page.title}\n{excerpt}\n来源：{page.url}")
        return "\n\n".join(excerpts)

    def _extract_relevant_excerpt(self, question: str, text: str, max_chars: int = MAX_QUOTE_CHARS) -> str:
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        if not paragraphs:
            return text[:max_chars]

        keywords = [token for token in re.split(r"\s+", self._build_search_query(question)) if token]
        scored: list[tuple[int, int, str]] = []
        for index, paragraph in enumerate(paragraphs):
            score = sum(2 for keyword in keywords if keyword and keyword in paragraph)
            if index <= 2:
                score += 1
            scored.append((score, -index, paragraph))

        scored.sort(reverse=True)
        selected: list[str] = []
        total_len = 0
        for score, _neg_index, paragraph in scored:
            if score == 0 and selected:
                continue
            if total_len + len(paragraph) > max_chars and selected:
                continue
            selected.append(paragraph)
            total_len += len(paragraph)
            if total_len >= max_chars:
                break

        if not selected:
            selected = paragraphs[:2]

        excerpt = "\n\n".join(selected)
        if len(excerpt) > max_chars:
            excerpt = excerpt[:max_chars].rstrip() + "..."
        return excerpt

    def _format_pages(self, pages: list[WikiPage]) -> str:
        return "\n".join(f"{i}. {page.title}\n   {page.url}" for i, page in enumerate(pages, start=1))

    def _format_search_results(self, results: list[SearchResult]) -> str:
        lines = []
        for i, result in enumerate(results, start=1):
            snippet = f" - {result.snippet}" if result.snippet else ""
            lines.append(f"{i}. {result.title}{snippet}\n   {result.url}")
        return "\n".join(lines)


def run_once(question: str) -> None:
    agent = IsaacWikiAgent()
    result = agent.answer(question)
    print(result.answer)


def run_repl() -> None:
    agent = IsaacWikiAgent()
    print("Isaac Wiki Agent Demo")
    print("输入问题后回车；输入 exit 或 quit 退出。")
    while True:
        question = input("\n你的问题> ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue
        result = agent.answer(question)
        print("\n" + result.answer)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Query Isaac HuijiWiki through a tiny agent.")
    parser.add_argument("question", nargs="*", help="Question to ask, for example: 硫磺火有什么效果")
    parser.add_argument("--interactive", "-i", action="store_true", help="Start an interactive shell")
    args = parser.parse_args()

    if args.interactive or not args.question:
        run_repl()
    else:
        run_once(" ".join(args.question))


if __name__ == "__main__":
    main()
