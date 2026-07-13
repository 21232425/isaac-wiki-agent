from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass

# 假设原有的媒体文件/Wiki工具接口保持不变
from tools.mediawiki import SearchResult, WikiApiError, WikiPage, get_wiki_page, search_wiki
from openai import OpenAI, APIError


ALLOW_ONLINE_WIKI = False


@dataclass
class AgentAnswer:
    question: str
    search_results: list[SearchResult]
    page: WikiPage | None
    pages: list[WikiPage]
    answer: str
    online_requested: bool = False
    online_enabled: bool = False
    online_used: bool = False
    tools_used: bool = False
    memory_fallback: bool = False

# 核心：赋予大模型人设与行动指南
SYSTEM_PROMPT = """你是一个精通《以撒的结合：忏悔》的资深 Wiki 助手。你的任务是通过调用工具，为玩家提供精准、详细的解答。

【数据源架构】
1. search_wiki 和 read_wiki_page 默认读取本地 SQLite 数据库。
2. 程序会单独告知你当前是否拥有联网权限。拥有权限时，由你根据问题和本地结果自主决定是否把工具参数 online 设为 true；不要要求用户使用固定口令。
3. 优先使用本地数据库。只有本地信息缺失、明显不相关、需要核对最新内容，或用户明确要求核对网页时，才考虑联网。
4. 没有联网权限时只能使用本地数据库，不得声称已经访问网页。
5. 在线读取成功后会自动写回本地数据库，供后续查询复用。
6. 数据来源由交互界面统一附在回答末尾。除非用户明确询问，否则正文不要主动描述本地数据库、联网模式、缓存过程或来源链接。

【行动指南】
1. 意图分析：如果用户提问模糊（例如“吐绿水的苍蝇”或“通关里以撒解锁的道具”），请先利用你的内在游戏知识推测可能的道具/怪物/机制名称。
2. 搜索（search_wiki）：利用推测出的关键词（中英文皆可），调用工具进行搜索。
3. 读取（read_wiki_page）：分析搜索结果的标题，选取最相关的标题调用读取工具，获取页面正文。
4. 验证与重试：如果读取的内容不包含用户需要的答案，你可以尝试搜索其他关键词并再次读取；不要重复调用同一个无结果的查询。
5. 最终回答：优先基于工具内容给出准确的中文回答。如果数据库未命中或内容明显无关，可以使用你自身已有的《以撒的结合》游戏知识直接回答，但不要虚构不确定的精确数值。
6. 如果用户输入“介绍你能做什么”或询问你的能力，请详细介绍可查询的角色、道具、怪物、Boss、机制与解锁内容。"""

class IsaacWikiAgent:
    """基于 Tool-Calling 架构的以撒 Wiki 智能体"""

    def __init__(self, allow_online: bool = ALLOW_ONLINE_WIKI):
        self.allow_online = allow_online
        # 1. 配置 DeepSeek 的 API Key 和 Base URL
        # 注意：不要把 platform.deepseek.com 填进 base_url，真正的 API 接口地址是 api.deepseek.com
        # 安全代码 ✅
        self.client = OpenAI(
            # 只保留读取环境变量的逻辑，删掉后面的真实 Key
            api_key=os.getenv("DEEPSEEK_API_KEY"), 
            base_url="https://api.deepseek.com" 
        )
        
        # 2. 修改为你指定的模型名称
        # （注：目前 DeepSeek 官方标准调用名称通常为 deepseek-chat 或 deepseek-reasoner，
        # 如果 deepseek-v4-pro 报错“模型不存在”，请去控制台确认一下准确的模型调用名并替换）
        self.model = "deepseek-v4-pro"
        
        # 定义大模型可以使用的工具列表
        online_search_property = {
            "online": {
                "type": "boolean",
                "description": "是否绕过本地数据库并从 wiki.gg 在线搜索。仅在本地结果不足或需要核对网页时设为 true。",
            }
        } if self.allow_online else {}
        online_read_property = {
            "online": {
                "type": "boolean",
                "description": "是否绕过本地缓存并从 wiki.gg 在线读取当前正文。仅在确有必要时设为 true。",
            }
        } if self.allow_online else {}

        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "search_wiki",
                    "description": "搜索以撒 Wiki。输入关键词，返回相关的 Wiki 页面标题和简短摘要列表。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "搜索关键词，例如道具名、角色名、机制等，支持中英文。"
                            },
                            **online_search_property,
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "read_wiki_page",
                    "description": "读取指定的以撒 Wiki 页面完整正文内容。必须传入准确的页面标题。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "title": {
                                "type": "string",
                                "description": "要读取的 Wiki 页面完整标题（通常来源于 search_wiki 的结果）。"
                            },
                            **online_read_property,
                        },
                        "required": ["title"]
                    }
                }
            }
        ]

    def answer(
        self,
        question: str,
        history: list[dict[str, str]] | None = None,
    ) -> AgentAnswer:
        # 用于追踪本次对话中大模型调用了哪些结果，保留你原有的数据结构返回
        accumulated_search_results: list[SearchResult] = []
        accumulated_pages: list[WikiPage] = []
        online_used = False
        tools_used = False
        memory_fallback = False

        permission_prompt = (
            "当前联网权限：已开启。你可以自主决定将工具参数 online 设为 true，但应优先查询本地数据库。"
            if self.allow_online
            else "当前联网权限：已关闭。工具只能读取本地数据库，不得声称访问过网页。"
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "system", "content": permission_prompt},
        ]
        if history:
            for message in history[-5:]:
                if message.get("role") not in {"user", "assistant"}:
                    continue
                clean_content = _remove_dsml(str(message.get("content", ""))).strip()
                if clean_content and not _contains_dsml(clean_content):
                    messages.append({"role": message["role"], "content": clean_content})
        messages.append({"role": "user", "content": question})

        # 开启 ReAct (Reasoning and Acting) 循环，设置最大轮数防止死循环
        max_iterations = 6
        final_answer_text = ""

        def execute_and_record(function_name: str, args: dict) -> str:
            nonlocal memory_fallback, online_used, tools_used
            tools_used = True
            tool_result, results, page, used_online = self._execute_tool_call(
                function_name,
                args,
            )
            online_used = online_used or used_online
            _extend_unique_results(accumulated_search_results, results)
            if page and all(existing.url != page.url for existing in accumulated_pages):
                accumulated_pages.append(page)
            if function_name in {"search_wiki", "read_wiki_page"} and not results and page is None:
                memory_fallback = True
            return tool_result

        try:
            for iteration in range(max_iterations):
                final_iteration = iteration == max_iterations - 1
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=self.tools,
                    tool_choice="none" if final_iteration else "auto",
                )
                
                response_message = response.choices[0].message
                response_content = response_message.content or ""
                dsml_tool_calls = _parse_dsml_tool_calls(response_content)

                if dsml_tool_calls and not response_message.tool_calls:
                    tool_results = []
                    for function_name, args in dsml_tool_calls:
                        print(f"[Agent 思考中] 兼容执行: {function_name}, 参数: {args}")
                        tool_results.append(
                            f"{function_name}: {execute_and_record(function_name, args)}"
                        )
                    messages.append({"role": "assistant", "content": "我会结合查询结果继续回答。"})
                    messages.append({
                        "role": "system",
                        "content": (
                            "以下是刚才内部工具的返回结果。请继续回答用户原问题，不要输出 DSML、XML、"
                            "工具调用标签或调用参数。如果结果为空、报错或与问题无关，请停止重复检索，"
                            "改用你自身已有的《以撒的结合》游戏知识自然作答。\n\n"
                            + "\n\n".join(tool_results)
                        ),
                    })
                    continue

                if _contains_dsml(response_content):
                    memory_fallback = True
                    messages.append({
                        "role": "system",
                        "content": (
                            "上一条回复包含无法识别的内部工具标记。不要再次输出任何 DSML、XML 或工具标签；"
                            "请直接根据已有资料和自身游戏知识回答用户原问题。"
                        ),
                    })
                    continue

                # 【修复核心】将对象转为纯字典，解决代理 API 兼容性导致的“失忆”问题
                messages.append(response_message.model_dump(exclude_none=True))

                # 如果模型没有调用工具，说明它认为已经收集到足够信息
                if not response_message.tool_calls:
                    final_answer_text = response_content.strip()
                    break

                # 如果模型决定调用工具
                for tool_call in response_message.tool_calls:
                    function_name = tool_call.function.name
                    
                    # 【增加监控】把 Agent 大脑里的想法打印在终端里！
                    print(f"[Agent 思考中] 决定调用: {function_name}, 参数: {tool_call.function.arguments}")
                    
                    try:
                        args = json.loads(tool_call.function.arguments)
                    except json.JSONDecodeError:
                        args = {}

                    tool_result_str = execute_and_record(function_name, args)

                    # 将工具执行的结果追加到历史记录中，供大模型下一步判断
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": function_name,
                        "content": tool_result_str
                    })

            if not final_answer_text:
                memory_fallback = True
                fallback_response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages + [{
                        "role": "system",
                        "content": (
                            "现在停止调用工具，直接回答用户最初的问题。可以使用你自身已有的《以撒的结合》"
                            "游戏知识补充数据库缺失内容。只输出自然的中文答案，不要输出 DSML、XML 或工具标签。"
                        ),
                    }],
                )
                final_answer_text = _remove_dsml(
                    fallback_response.choices[0].message.content or ""
                ).strip()
                if _contains_dsml(final_answer_text):
                    final_answer_text = ""
        except APIError as exc:
            final_answer_text = f"调用大模型 API 时发生错误：{exc}\n\n请检查 API Key 状态或网络连通性。"
        except Exception as exc:
            final_answer_text = f"代理执行过程中发生未处理异常：{exc}"

        if not final_answer_text:
            final_answer_text = "我暂时无法生成可靠答案，请换一种说法后再试。"

        return AgentAnswer(
            question=question,
            search_results=accumulated_search_results,
            page=accumulated_pages[0] if accumulated_pages else None,
            pages=accumulated_pages,
            answer=final_answer_text,
            online_requested=online_used,
            online_enabled=self.allow_online,
            online_used=online_used,
            tools_used=tools_used,
            memory_fallback=memory_fallback,
        )

    def _execute_tool_call(
        self,
        function_name: str,
        args: dict,
    ) -> tuple[str, list[SearchResult], WikiPage | None, bool]:
        use_online = self.allow_online and _as_bool(args.get("online", False))
        if function_name == "search_wiki":
            tool_result, results = self._tool_search_wiki(
                str(args.get("query", "")),
                use_online=use_online,
            )
            used_online = use_online or any(
                result.retrieved_from == "remote_api" for result in results
            )
            return tool_result, results, None, used_online
        if function_name == "read_wiki_page":
            tool_result, page = self._tool_read_wiki_page(
                str(args.get("title", "")),
                use_online=use_online,
            )
            used_online = use_online or bool(
                page and page.retrieved_from == "remote_api"
            )
            return tool_result, [], page, used_online
        return f"错误：未知的工具调用 '{function_name}'", [], None, False

    def _tool_search_wiki(
        self,
        query: str,
        use_online: bool = False,
    ) -> tuple[str, list[SearchResult]]:
        """封装 search_wiki 供大模型调用，返回 (供大模型阅读的文本, 原始数据对象)"""
        if not query:
            return "错误：搜索关键词不能为空。", []
        use_online = self.allow_online and use_online
        try:
            results = search_wiki(query, limit=5, allow_remote=use_online)
            if not results:
                if use_online:
                    return (
                        f"在线 Wiki 未找到关于 '{query}' 的结果。请停止重复检索，改用已有游戏知识回答。",
                        [],
                    )
                return (
                    f"本地数据库未找到关于 '{query}' 的结果。请停止重复检索，改用已有游戏知识回答。",
                    [],
                )
            
            # 将结果格式化为大模型易于理解的纯文本
            formatted_text = f"关于 '{query}' 的搜索结果如下：\n"
            for i, res in enumerate(results, start=1):
                formatted_text += (
                    f"{i}. 标题: {res.title}\n"
                    f"   摘要: {res.snippet}\n"
                )
            return formatted_text, results
        except WikiApiError as exc:
            return f"执行 Wiki 搜索 API 失败：{exc}", []

    def _tool_read_wiki_page(
        self,
        title: str,
        use_online: bool = False,
    ) -> tuple[str, WikiPage | None]:
        """封装 get_wiki_page 供大模型调用，返回 (供大模型阅读的正文, 原始数据对象)"""
        if not title:
            return "错误：页面标题不能为空。", None
        use_online = self.allow_online and use_online
        try:
            page = get_wiki_page(title, allow_remote=use_online)
            # 限制返回给大模型的字符数，防止超长报错
            extract_text = page.extract[:8000] if page.extract else "（该页面没有正文内容）"
            formatted_text = (
                f"页面 '{page.title}' 的正文内容摘录：\n"
                f"{extract_text}"
            )
            return formatted_text, page
        except WikiApiError as exc:
            return (
                f"读取页面 '{title}' 失败（错误信息：{exc}）。请停止重复读取，改用已有游戏知识回答。",
                None,
            )


_DSML_INVOKE_PATTERN = re.compile(
    r'<[|｜]{2}DSML[|｜]{2}invoke\s+name=["\']([^"\']+)["\'][^>]*>'
    r'(.*?)</[|｜]{2}DSML[|｜]{2}invoke>',
    re.DOTALL | re.IGNORECASE,
)
_DSML_PARAMETER_PATTERN = re.compile(
    r'<[|｜]{2}DSML[|｜]{2}parameter\s+name=["\']([^"\']+)["\'][^>]*>'
    r'(.*?)</[|｜]{2}DSML[|｜]{2}parameter>',
    re.DOTALL | re.IGNORECASE,
)
_DSML_BLOCK_PATTERN = re.compile(
    r'<[|｜]{2}DSML[|｜]{2}tool_calls[^>]*>.*?'
    r'</[|｜]{2}DSML[|｜]{2}tool_calls>',
    re.DOTALL | re.IGNORECASE,
)


def _parse_dsml_tool_calls(content: str) -> list[tuple[str, dict[str, str]]]:
    calls = []
    for invoke_match in _DSML_INVOKE_PATTERN.finditer(content):
        function_name = invoke_match.group(1).strip()
        arguments = {
            parameter_match.group(1).strip(): parameter_match.group(2).strip()
            for parameter_match in _DSML_PARAMETER_PATTERN.finditer(invoke_match.group(2))
        }
        calls.append((function_name, arguments))
    return calls


def _contains_dsml(content: str) -> bool:
    return bool(re.search(r'[|｜]{2}DSML[|｜]{2}', content, re.IGNORECASE))


def _remove_dsml(content: str) -> str:
    return _DSML_BLOCK_PATTERN.sub("", content)


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().casefold() in {"1", "true", "yes", "on"}

def _extend_unique_results(
    existing: list[SearchResult],
    new_results: list[SearchResult],
) -> None:
    seen = {(result.title.casefold(), result.url) for result in existing}
    for result in new_results:
        key = (result.title.casefold(), result.url)
        if key not in seen:
            existing.append(result)
            seen.add(key)


def run_once(question: str) -> None:
    agent = IsaacWikiAgent()
    print("Agent 正在思考并检索中，请稍候...\n" + "="*40)
    result = agent.answer(question)
    print("\n最终回答:\n" + result.answer)


def run_repl() -> None:
    agent = IsaacWikiAgent()
    print("Isaac Wiki Agent Demo (Tool-Calling 版本)")
    print("输入问题后回车；输入 exit 或 quit 退出。")
    while True:
        question = input("\n你的问题> ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue
        print("Agent 正在思考并调用工具...")
        result = agent.answer(question)
        print("\n" + result.answer)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="通过 Tool-Calling 架构的 Agent 查询以撒 Wiki。")
    parser.add_argument("question", nargs="*", help="你想问的问题，例如：那个打通里以撒解锁的道具叫什么")
    parser.add_argument("--interactive", "-i", action="store_true", help="启动交互式命令行")
    args = parser.parse_args()

    if args.interactive or not args.question:
        run_repl()
    else:
        run_once(" ".join(args.question))


if __name__ == "__main__":
    main()
