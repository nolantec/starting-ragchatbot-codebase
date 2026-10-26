# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概览

课程资料 RAG 系统——一个全栈 Web 应用，用户可以用自然语言查询课程内容，系统通过语义搜索 + Claude AI 生成带来源引用的回答。

技术栈：FastAPI 后端 + 原生 HTML/CSS/JS 前端 + ChromaDB 向量库 + Anthropic Claude API。

## 常用命令

```bash
# 安装依赖
uv sync

# 启动开发服务器（热重载）
cd backend && uv run uvicorn app:app --reload --port 8000
# 或从项目根目录：
./run.sh

# 访问应用
# Web 界面:   http://localhost:8000
# API 文档:   http://localhost:8000/docs

# 环境变量
cp .env.example .env  # 填入 ANTHROPIC_API_KEY
```

项目无测试框架、无 lint 工具、无构建步骤。前端是纯静态文件，由 FastAPI 直接托管。

## 架构总览

这是一个**工具调用式 RAG** 系统，关键区别：不是每次查询都先检索再生成，而是由 Claude 自主判断是否需要调用搜索工具。

### 请求流转（前端 → 后端 → AI → 检索 → 返回）

```
用户输入 → script.js:sendMessage()
  → POST /api/query (app.py)
    → RAGSystem.query()
      → SessionManager.get_conversation_history()  [取对话历史]
      → AIGenerator.generate_response()  [调用 Claude API，带工具定义]
        → Claude 决定: 直接回答 OR 调用 search_course_content 工具
        → 若工具调用: ToolManager.execute_tool()
          → CourseSearchTool.execute()
            → VectorStore.search()
              → _resolve_course_name()  [course_catalog 集合模糊匹配课程名]
              → course_content.query()  [语义检索 top-K 内容块]
            → _format_results()  [格式化结果，记录 last_sources]
        → Claude 二次调用  [基于检索结果生成最终回答]
      → SessionManager.add_exchange()  [保存对话历史]
      → 返回 (answer, sources)
  → QueryResponse JSON
→ 前端 marked.js 渲染 Markdown + Sources 折叠区
```

### 核心设计决策

- **双集合向量存储**：`course_catalog`（课程标题/元数据，用于模糊匹配课程名）+ `course_content`（分块课程内容，用于语义搜索）。当用户说"MCP 课"时，先在 catalog 里做语义匹配找到完整课程名，再用它过滤内容搜索。
- **本地嵌入模型**：使用 `sentence-transformers/all-MiniLM-L6-v2` 本地生成向量，不调外部 embedding API。
- **会话管理**：纯内存存储（`SessionManager.sessions` 字典），进程重启即丢失。`MAX_HISTORY=2` 轮对话。
- **单轮搜索限制**：系统提示词限制"每次查询最多一次搜索"，避免多轮工具调用导致延迟过高。
- **文档格式**：`docs/` 下的 `.txt` 文件遵循固定格式——前三行是 Course Title / Course Link / Course Instructor，之后是 `Lesson N: Title` 分隔的课时内容。由 `DocumentProcessor` 解析。

## 模块职责

| 模块 | 职责 |
|------|------|
| `backend/app.py` | FastAPI 应用、路由、静态文件托管、启动时加载 docs |
| `backend/rag_system.py` | RAG 编排器，串联所有子系统 |
| `backend/ai_generator.py` | Claude API 封装，处理 tool_use 循环 |
| `backend/search_tools.py` | ToolManager + CourseSearchTool，向量检索入口 |
| `backend/vector_store.py` | ChromaDB 封装，两个集合的增删查 |
| `backend/session_manager.py` | 内存会话与对话历史 |
| `backend/document_processor.py` | 课程文档解析 + 句子级分块（带重叠） |
| `backend/models.py` | Pydantic 模型：Course / Lesson / CourseChunk |
| `backend/config.py` | 配置 dataclass，从 .env 读 ANTHROPIC_API_KEY |
| `frontend/script.js` | 聊天 UI + API 调用 + Markdown 渲染 |

## 注意事项

- 后端工作目录是 `backend/`，ChromaDB 默认存在 `backend/chroma_db/`（相对路径）。
- 前端通过相对路径 `/api` 调用后端，由 FastAPI 的 `StaticFiles` 挂载在 `/` 提供服务。
- `main.py` 是占位文件，与应用运行无关。
- 添加新课程：把符合格式的 `.txt` 文件放进 `docs/`，重启服务会自动加载（跳过已存在的课程）。
