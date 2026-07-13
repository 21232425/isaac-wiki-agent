from __future__ import annotations

import json
import os
import urllib.error
import urllib.request


API_BASE_URL = "https://letaicode.cn/codex"
RESPONSES_API_URL = f"{API_BASE_URL.rstrip('/')}/responses"
CHAT_COMPLETIONS_API_URL = f"{API_BASE_URL.rstrip('/')}/chat/completions"
DEFAULT_API_STYLE = "auto"
DEFAULT_API_KEY = ""
DEFAULT_MODEL = "gpt-5.4"


class LlmConfigError(RuntimeError):
    """Raised when the LLM client is not configured."""


class LlmApiError(RuntimeError):
    """Raised when the OpenAI API request fails."""


def is_llm_configured() -> bool:
    return bool(_get_api_key())


def summarize_with_llm(question: str, title: str, url: str, context: str) -> str:
    """Summarize retrieved wiki context with an OpenAI-compatible API."""
    api_key = _get_api_key()
    if not api_key:
        raise LlmConfigError("OPENAI_API_KEY is not set and DEFAULT_API_KEY is empty.")

    model = os.getenv("OPENAI_MODEL", DEFAULT_MODEL)
    prompt = _build_prompt(question=question, title=title, url=url, context=context)
    system_prompt = (
        "你是一个严谨的中文资料检索助手。你只能基于用户提供的来源内容回答。"
        "如果来源内容不足以回答问题，必须明确说明不足。回答要简洁、准确，并保留来源。"
    )
    api_style = os.getenv("OPENAI_API_STYLE", DEFAULT_API_STYLE).strip().lower()

    errors: list[str] = []
    styles = ["responses", "chat_completions"] if api_style == "auto" else [api_style]
    for style in styles:
        try:
            if style == "responses":
                body = _post_json(
                    RESPONSES_API_URL,
                    api_key,
                    {
                        "model": model,
                        "input": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt},
                        ],
                        "temperature": 0.2,
                        "max_output_tokens": 900,
                    },
                )
                text = _extract_response_text(body)
            elif style in {"chat", "chat_completions", "chat-completions"}:
                body = _post_json(
                    CHAT_COMPLETIONS_API_URL,
                    api_key,
                    {
                        "model": model,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt},
                        ],
                        "temperature": 0.2,
                        "max_tokens": 900,
                    },
                )
                text = _extract_chat_completion_text(body)
            else:
                raise LlmApiError(f"Unsupported OPENAI_API_STYLE: {style}")

            if text:
                return text.strip()
            errors.append(f"{style}: API returned no text output.")
        except LlmApiError as exc:
            errors.append(f"{style}: {exc}")

    raise LlmApiError("All LLM API styles failed.\n" + "\n".join(errors))


def _get_api_key() -> str:
    return os.getenv("OPENAI_API_KEY") or DEFAULT_API_KEY


def _post_json(url: str, api_key: str, payload: dict) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            return _loads_json_or_raise(resp.status, resp.headers.get("Content-Type", ""), raw)
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        content_type = exc.headers.get("Content-Type", "")
        try:
            body = _loads_json_or_raise(exc.code, content_type, raw)
        except LlmApiError:
            raise LlmApiError(
                f"HTTP {exc.code} {exc.reason}; non-JSON response: {_preview(raw)}"
            ) from exc
        raise LlmApiError(f"HTTP {exc.code} {exc.reason}; JSON response: {body}") from exc
    except Exception as exc:  # noqa: BLE001 - keep CLI output readable.
        raise LlmApiError(f"request failed: {exc}") from exc


def _loads_json_or_raise(status: int, content_type: str, raw: str) -> dict:
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise LlmApiError(
            f"HTTP {status}; expected JSON but got Content-Type={content_type!r}; "
            f"response preview: {_preview(raw)}"
        ) from exc


def _preview(text: str, limit: int = 500) -> str:
    text = text.replace("\r", "\\r").replace("\n", "\\n")
    return text[:limit] + ("..." if len(text) > limit else "")


def _build_prompt(question: str, title: str, url: str, context: str) -> str:
    return f"""请根据以下以撒 Wiki 检索资料回答用户问题。

用户问题：
{question}

来源说明：
{title}
{url}

检索资料：
{context}

回答要求：
1. 先直接回答问题，不要复述检索过程。
2. 根据问题类型组织答案：
   - 道具/饰品/卡牌/符文：概括效果、机制、重要协同或注意事项。
   - 角色：概括解锁方式、初始属性/道具、玩法特点和注意事项。
   - 成就：概括解锁条件、相关奖励和注意事项。
   - 挑战：概括规则限制、通关目标、奖励和策略要点。
   - 怪物/Boss/房间/机制：概括出现条件、行为机制、应对方式。
3. 不要编造页面内容中没有的信息。
4. 如果检索资料中有多个候选页面，请判断哪个最相关；不确定时要说明不确定。
5. 结尾列出实际使用的来源标题和链接。
"""


def _extract_response_text(body: dict) -> str:
    if isinstance(body.get("output_text"), str):
        return body["output_text"]

    chunks: list[str] = []
    for item in body.get("output", []):
        for content in item.get("content", []):
            if content.get("type") in {"output_text", "text"} and isinstance(content.get("text"), str):
                chunks.append(content["text"])
    return "\n".join(chunks)


def _extract_chat_completion_text(body: dict) -> str:
    choices = body.get("choices", [])
    if not choices:
        return ""
    message = choices[0].get("message", {})
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                chunks.append(item["text"])
        return "\n".join(chunks)
    return ""
