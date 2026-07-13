# 以撒 Wiki 智能助手

这个项目使用 DeepSeek Tool Calling 回答《以撒的结合》问题。Wiki 工具现在采用以下顺序：

1. 默认只查询本地 SQLite 数据库 `data/isaac_wiki.sqlite3`。
2. 只有用户明确要求“请联网搜索”等操作时，才强制查询公开的 `wiki.gg` MediaWiki API，并用当前正文更新缓存。
3. 在线读取成功后自动写回 SQLite，后续相同查询直接走本地。

原先使用的灰机 Wiki API 会对脚本请求返回 HTTP 403。伪造浏览器请求头并不是稳定或合适的解决方式，所以默认在线源已改为公开 API 可正常访问的 `wiki.gg`。

## 安装

```powershell
pip install -r requirements.txt
```

## 构建本地数据库

推荐直接从当前 `wiki.gg` API 同步英文主站和中文分站：

```powershell
python sync_wiki.py bootstrap
```

查看数据库状态：

```powershell
python sync_wiki.py status
```

只同步英文站：

```powershell
python sync_wiki.py sync-api
```

### 使用 Internet Archive XML 转储

Internet Archive 保存了一份 2024-08-31 的完整 `wiki.gg` 文本历史转储。正文压缩包约 17 MB，不需要下载约 444 MB 的图片包。

```powershell
python sync_wiki.py download-dump
python sync_wiki.py import-dump data/downloads/bindingofisaacrebirth-20240831-history.xml.zst
```

导入器支持 `.xml`、`.xml.gz`、`.xml.bz2` 和 `.xml.zst`。XML 中的 Wiki 标记会转换成便于 Agent 阅读的纯文本；若需要更干净、更新的正文，优先使用 `sync-api`。

注意：这个历史转储解压后约 1.1 GB，并使用大窗口 Zstandard 压缩。导入时需要足够内存；机器资源有限时请直接使用 `bootstrap` 或 `sync-api`。

## 运行

配置 DeepSeek Key 和访问密码：

```powershell
$env:DEEPSEEK_API_KEY="sk-xxxx"
$env:APP_PASSWORD="请换成强密码"
```

启动网页：

```powershell
streamlit run web_app.py
```

启动命令行：

```powershell
python true_agent.py -i
```

## 配置

- `ISAAC_WIKI_DB`：覆盖 SQLite 数据库路径。
- `ISAAC_WIKI_REMOTE_APIS`：逗号分隔的 MediaWiki API 地址。默认依次使用 `wiki.gg` 中文和英文 API。
- `ISAAC_WIKI_OFFLINE=1`：完全禁止在线兜底，只查询本地数据库。

无论是否开启离线模式，只要本地库命中，程序就不会发起网络请求。

## 部署到 GitHub 与 Streamlit Community Cloud

建议把 `isaac_wiki` 目录单独作为 GitHub 仓库，不要把上级目录中的其他项目和文档一起上传。

本项目需要提交 `data/isaac_wiki.sqlite3`，它是应用启动时使用的本地底库。不要提交 `data/downloads/` 中的原始历史转储；部署运行不需要它，而且 `.gitignore` 已将其排除。

### 1. 创建并推送 GitHub 仓库

先在 GitHub 创建一个空仓库，例如 `isaac-wiki-agent`，不要勾选自动创建 README。然后在本目录执行：

```powershell
cd D:\work\东南智能学工平台\isaac_wiki
git init
git branch -M main
git add .
git status
git commit -m "Initial Streamlit deployment"
git remote add origin https://github.com/你的用户名/isaac-wiki-agent.git
git push -u origin main
```

提交前请确认 `git status` 中包含 `data/isaac_wiki.sqlite3`，但不包含 `.streamlit/secrets.toml`、`letcode_codex配置文件.txt`、`__pycache__` 和 `data/downloads/`。

### 2. 部署到 Streamlit Community Cloud

1. 打开 `https://share.streamlit.io` 并连接 GitHub。
2. 点击 **Create app**，选择刚创建的仓库和 `main` 分支。
3. Main file path 填写 `web_app.py`。
4. 在 **Advanced settings** 中选择 Python 3.12。
5. 在 Secrets 中填写：

```toml
DEEPSEEK_API_KEY = "sk-你的真实Key"
APP_PASSWORD = "你的访问密码"
```

6. 点击 Deploy。`requirements.txt` 会自动安装 Python 依赖。

Streamlit 容器中对 SQLite 的在线回写只属于运行时缓存，重启或重新部署后可能恢复为 GitHub 仓库中的初始数据库；因此重要更新应在本地执行 `python sync_wiki.py bootstrap` 后重新提交数据库。

## 数据来源与许可

- `wiki.gg` 页面内容采用 [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)；数据库为每页保留原始页面 URL 和来源。
- Internet Archive 转储来源：`wiki-bindingofisaacrebirth.wiki.gg-20240831`。
- `data/isaac_wiki.sqlite3` 中的衍生 Wiki 文本按 CC BY-SA 4.0 共享；页面作者与修订历史可通过数据库保留的原始页面链接和上述转储查询。
- 请保留来源链接与署名，并遵守各页面可能标注的额外条款。项目不下载图片或游戏资源。
