from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass

# 假设原有的媒体文件/Wiki工具接口保持不变
from tools.mediawiki import SearchResult, WikiApiError, WikiPage, get_wiki_page, search_wiki
from openai import OpenAI, APIError

@dataclass
class AgentAnswer:
    question: str
    search_results: list[SearchResult]
    page: WikiPage | None
    pages: list[WikiPage]
    answer: str

# 核心：赋予大模型人设与行动指南
SYSTEM_PROMPT = """你是一个精通《以撒的结合：忏悔》的资深 Wiki 助手。你的任务是通过调用工具，为玩家提供精准、详细的解答。

【数据源架构】
1. search_wiki 和 read_wiki_page 都由程序实现为“本地 SQLite 数据库优先”。
2. 只有本地数据库没有命中时，工具才会访问 wiki.gg 的公开 MediaWiki API。
3. 在线读取成功后会自动写回本地数据库，供后续查询复用。
4. 工具结果会明确给出“来源路径”。回答信息来源问题时必须以该字段为准，不得声称程序没有本地数据库能力。

【行动指南】
1. 意图分析：如果用户提问模糊（例如“吐绿水的苍蝇”或“通关里以撒解锁的道具”），请先利用你的内在游戏知识推测可能的道具/怪物/机制名称。
2. 搜索（search_wiki）：利用推测出的关键词（中英文皆可），调用工具进行搜索。
3. 读取（read_wiki_page）：分析搜索结果的标题，选取最相关的标题调用读取工具，获取页面正文。
4. 验证与重试：如果读取的内容不包含用户需要的答案，你可以尝试搜索其他关键词并再次读取。
5. 最终回答：基于你读取到的工具返回内容给出详细、准确的中文回答。不要编造游戏数据（无幻觉）。"""

class IsaacWikiAgent:
    """基于 Tool-Calling 架构的以撒 Wiki 智能体"""

    def __init__(self):
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
                            }
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
                            }
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

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        if history:
            messages.extend(
                message
                for message in history[-5:]
                if message.get("role") in {"user", "assistant"} and message.get("content")
            )
        messages.append({"role": "user", "content": question})

        # 开启 ReAct (Reasoning and Acting) 循环，设置最大轮数防止死循环
        max_iterations = 6
        final_answer_text = ""

        try:
            for _ in range(max_iterations):
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=self.tools,
                    tool_choice="auto"
                )
                
                response_message = response.choices[0].message
                # 【修复核心】将对象转为纯字典，解决代理 API 兼容性导致的“失忆”问题
                messages.append(response_message.model_dump(exclude_none=True))

                # 如果模型没有调用工具，说明它认为已经收集到足够信息
                if not response_message.tool_calls:
                    final_answer_text = response_message.content
                    break

                # 如果模型决定调用工具
                for tool_call in response_message.tool_calls:
                    function_name = tool_call.function.name
                    
                    # 【增加监控】把 Agent 大脑里的想法打印在终端里！
                    print(f"👀 [Agent 思考中] 决定调用: {function_name}, 参数: {tool_call.function.arguments}")
                    
                    try:
                        args = json.loads(tool_call.function.arguments)
                    except json.JSONDecodeError:
                        args = {}

                    tool_result_str = ""

                    if function_name == "search_wiki":
                        query = args.get("query", "")
                        tool_result_str, results = self._tool_search_wiki(query)
                        accumulated_search_results.extend(results)
                        
                    elif function_name == "read_wiki_page":
                        title = args.get("title", "")
                        tool_result_str, page = self._tool_read_wiki_page(title)
                        if page:
                            accumulated_pages.append(page)
                            
                    else:
                        tool_result_str = f"错误：未知的工具调用 '{function_name}'"

                    # 将工具执行的结果追加到历史记录中，供大模型下一步判断
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": function_name,
                        "content": tool_result_str
                    })
            else:
                final_answer_text = "代理经过多次搜索与阅读依然未能得出结论，请尝试提供更准确的名称或更具体的问题。"

        except APIError as exc:
            final_answer_text = f"调用大模型 API 时发生错误：{exc}\n\n请检查 API Key 状态或网络连通性。"
        except Exception as exc:
            final_answer_text = f"代理执行过程中发生未处理异常：{exc}"

        return AgentAnswer(
            question=question,
            search_results=accumulated_search_results,
            page=accumulated_pages[0] if accumulated_pages else None,
            pages=accumulated_pages,
            answer=final_answer_text
        )

    def _tool_search_wiki(self, query: str) -> tuple[str, list[SearchResult]]:
        """封装 search_wiki 供大模型调用，返回 (供大模型阅读的文本, 原始数据对象)"""
        if not query:
            return "错误：搜索关键词不能为空。", []
        try:
            results = search_wiki(query, limit=5)
            if not results:
                return f"未找到关于 '{query}' 的搜索结果，请尝试其他关键词。", []
            
            # 将结果格式化为大模型易于理解的纯文本
            formatted_text = f"关于 '{query}' 的搜索结果如下：\n"
            for i, res in enumerate(results, start=1):
                route = _source_route_label(res.retrieved_from)
                formatted_text += (
                    f"{i}. 标题: {res.title}\n"
                    f"   摘要: {res.snippet}\n"
                    f"   来源路径: {route}\n"
                    f"   原始数据源: {res.source or res.url}\n"
                )
            return formatted_text, results
        except WikiApiError as exc:
            return f"执行 Wiki 搜索 API 失败：{exc}", []

    def _tool_read_wiki_page(self, title: str) -> tuple[str, WikiPage | None]:
        """封装 get_wiki_page 供大模型调用，返回 (供大模型阅读的正文, 原始数据对象)"""
        if not title:
            return "错误：页面标题不能为空。", None
        try:
            page = get_wiki_page(title)
            # 限制返回给大模型的字符数，防止超长报错
            extract_text = page.extract[:8000] if page.extract else "（该页面没有正文内容）"
            route = _source_route_label(page.retrieved_from)
            formatted_text = (
                f"页面 '{page.title}' 的正文内容摘录：\n"
                f"来源路径: {route}\n"
                f"原始数据源: {page.source or page.url}\n\n"
                f"{extract_text}"
            )
            return formatted_text, page
        except WikiApiError as exc:
            return f"读取页面 '{title}' 失败，请检查标题是否完全一致（错误信息：{exc}）。", None


def _source_route_label(retrieved_from: str) -> str:
    if retrieved_from == "local_database":
        return "本地 SQLite 数据库"
    if retrieved_from == "remote_api":
        return "在线 wiki.gg API（本地未命中后兜底）"
    return "未知"


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
