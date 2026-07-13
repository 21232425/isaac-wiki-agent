# streamlit run web_app.py
import os
import secrets
import inspect

import streamlit as st
from true_agent import IsaacWikiAgent  # 直接引入你写好的 Agent


AGENT_STATE_VERSION = 2


def _source_label(retrieved_from: str) -> str:
    if retrieved_from == "local_database":
        return "本地数据库"
    if retrieved_from == "remote_api":
        return "在线 Wiki 兜底并已缓存"
    return "来源未知"


def _answer_with_history(agent, prompt: str, history: list[dict[str, str]]):
    parameters = inspect.signature(agent.answer).parameters
    if "history" in parameters:
        return agent.answer(prompt, history=history)
    return agent.answer(prompt)


# 设置网页标题和布局
st.set_page_config(page_title="以撒 Wiki 智能助手", page_icon="👼", layout="centered")

# ================= 密码拦截模块 =================
app_password = os.getenv("APP_PASSWORD")
if not app_password:
    st.error("缺少 APP_PASSWORD。请在 Streamlit Secrets 或环境变量中配置访问密码。")
    st.stop()

# 初始化登录状态
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

# 如果未登录，显示密码输入界面并阻止后续代码运行
if not st.session_state.authenticated:
    st.title("🔒 访问受限")
    st.caption("请输入密码以使用以撒 Wiki 智能助手")
    
    # 密码输入框，type="password" 会将输入内容显示为星号
    password = st.text_input("请输入密码：", type="password")
    
    if st.button("进入系统"):
        if secrets.compare_digest(password, app_password):
            st.session_state.authenticated = True
            st.success("密码正确！正在加载助手...")
            st.rerun()  # 刷新页面，跳过登录界面进入主程序
        else:
            st.error("密码错误，请重新输入。")
            
    # 重要：阻止未输入正确密码的用户执行后续的 Agent 初始化和聊天代码
    st.stop()
# ================================================

# 下方的代码只有在 st.session_state.authenticated 为 True 时才会执行
st.title("👼 以撒的结合 Wiki 智能助手")
st.caption("基于 DeepSeek 与 Tool-Calling 架构，优先检索本地 Wiki 数据库，未命中时访问 wiki.gg。")

# 初始化或更新 session_state 中的 Agent 实例。
# 版本号可避免代码热更新后继续使用旧类创建的实例。
if (
    "agent" not in st.session_state
    or st.session_state.get("agent_state_version") != AGENT_STATE_VERSION
):
    st.session_state.agent = IsaacWikiAgent()
    st.session_state.agent_state_version = AGENT_STATE_VERSION

# 初始化聊天历史记录
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "你好！我是以撒 Wiki 助手。想查什么道具、Boss 或机制？直接问我吧！"}
    ]

# 渲染历史聊天记录
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 接收用户输入
if prompt := st.chat_input("例如：打通里以撒解锁的那个换道具的叫什么？"):
    # 1. 把用户的问题显示在界面上
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 2. 调用你的 Agent 生成回答
    with st.chat_message("assistant"):
        with st.spinner("Agent 正在翻阅 Wiki 思考中，请稍候..."):
            try:
                # 调用 true_agent.py 中的 answer 方法
                result = _answer_with_history(
                    st.session_state.agent,
                    prompt,
                    st.session_state.messages[:-1],
                )
                response_text = result.answer
                
                # 如果你想在网页上展示它查了哪些网页，可以加上下面这段（可选）
                if result.pages:
                    sources = "\n\n**参考页面：**\n" + "\n".join(
                        [
                            f"- [{p.title}]({p.url})"
                            f"（{_source_label(p.retrieved_from)}）"
                            for p in result.pages
                        ]
                    )
                    response_text += sources

            except Exception as e:
                response_text = f"抱歉，查询时出现了错误：{e}"
        
        # 显示回答
        st.markdown(response_text)
    
    # 保存助手的回答到历史记录
    st.session_state.messages.append({"role": "assistant", "content": response_text})
