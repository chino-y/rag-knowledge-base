# RAG 知识库问答系统

基于 **LangChain + ChromaDB + DeepSeek** 的检索增强生成（Retrieval-Augmented Generation，RAG）学习与演示项目。

## 什么是 RAG？

RAG（检索增强生成）把「信息检索」和「大语言模型生成」结合起来：用户提问时，系统先从本地知识库中检索相关文档片段，再把这些片段作为上下文一起交给 LLM 生成答案。这样模型就能基于你的私有知识回答，大幅减少"幻觉"（编造内容）。

## 核心流程

```
文档加载 → 文本分割 → 向量化(Embedding) → 存入向量库(ChromaDB)
                                                ↓
用户提问 → 相似度检索 → 拼接 Prompt → LLM(DeepSeek) 生成答案
```

## 功能特性

- 📄 多格式文档加载：TXT / Markdown / PDF / DOCX
- ✂️ 文本分割：`RecursiveCharacterTextSplitter`，可调 `chunk_size` / `chunk_overlap`
- 🧬 向量化：HuggingFace Embedding（`BAAI/bge-m3`），本地 CPU/GPU 运行
- 🗄️ 持久化向量库：ChromaDB，构建一次、反复使用
- 🤖 RAG 生成：DeepSeek
- 🌐 Web 服务：FastAPI，提供 HTTP 问答接口

## 环境要求

- Python 3.9+
- 约 2GB 磁盘空间（嵌入模型 + 向量库）
- 能访问 DeepSeek API 的密钥
- （可选）GPU 可加速向量化，CPU 也能跑

## 目录结构

```
rag-study/
├── 02-本地大模型.py             # 用 ModelScope 加载本地 Qwen3 模型推理
├── 03-openai框架部署本地大模型.py # 用 OpenAI 兼容框架调用 Ollama 本地模型
├── 04-embadding模型下载.py       # 下载 BAAI/bge-m3 嵌入模型到 models/embeddings
├── 05-RAG.py                   # RAG 完整流程（核心），导出 rag() 供服务调用
├── main.py                     # 独立的 RAG 演示脚本
├── server.py                   # FastAPI Web 服务入口
├── requirements.txt            # 依赖清单
├── .env.example                # 环境变量模板（复制为 .env 使用）
├── LICENSE                     # MIT 许可证
├── docs/                       # 语料目录（放入你自己的文档）
│   ├── rag_intro.txt
│   └── langchain_guide.md
├── models/                     # 本地模型（不提交，需自行下载）
└── chroma_db_docs/             # 向量库（不提交，由 docs 构建）
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

> 依赖均为最新版（未锁版本）。如遇版本冲突，可自行降级相关包。

### 2. 配置环境变量

```bash
# Windows
copy .env.example .env

# Linux / macOS
cp .env.example .env
```

编辑 `.env`，填入你的 DeepSeek API Key：

```
DEEPSEEK_API_KEY=sk-你的密钥
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL_NAME=deepseek-chat
```

> 密钥只写在 `.env` 里，`.env` 已被 `.gitignore` 忽略，不会提交到仓库。

### 3. 下载嵌入模型

```bash
python 04-embadding模型下载.py
```

首次运行会自动从 HuggingFace 下载 `BAAI/bge-m3` 到 `models/embeddings`（国内用户已默认走 `hf-mirror.com` 镜像）。

### 4. 准备语料

把你要问答的文档放进 `docs/` 目录，支持格式：

| 格式 | 说明 |
|---|---|
| `.txt` | 纯文本，直接读取 |
| `.md` | Markdown，保留标题层级 |
| `.pdf` | PDF 文档 |
| `.docx` | Word 文档 |

> 仓库自带了 `rag_intro.txt` 和 `langchain_guide.md` 两个示例文档，可直接跑通。
> 隐私提示：本项目演示用的 `emotion.txt`（心理疏导对话）因涉及隐私未提交，请自行准备语料。

### 5. 构建知识库并测试检索

```bash
python 05-RAG.py
```

会依次执行：**加载文档 → 文本分割 → 向量化 → 存入 ChromaDB → 检索测试**。构建完成后，向量库持久化在 `chroma_db_docs/`，之后无需重复构建。

### 6. 启动 Web 服务

```bash
python server.py
# 或
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```

访问：

- 问答接口：`http://localhost:8000/ai_chat?query=什么是RAG？`
- Swagger 文档：`http://localhost:8000/docs`

## Web API 接口

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/` | 健康检查 / 首页 |
| GET | `/ai_chat?query=你的问题` | RAG 问答，返回 `{"ai_text": "..."}` |

示例：

```bash
curl "http://localhost:8000/ai_chat?query=什么是RAG？"
```

返回：

```json
{"ai_text": "RAG 全称是检索增强生成……"}
```

## 环境变量说明

| 变量 | 说明 | 默认值 | 必填 |
|---|---|---|---|
| `DEEPSEEK_API_KEY` | DeepSeek API 密钥 | — | ✅ |
| `DEEPSEEK_BASE_URL` | DeepSeek API 地址 | `https://api.deepseek.com` | ❌ |
| `DEEPSEEK_MODEL_NAME` | 模型名称 | `deepseek-chat` | ❌ |
| `HF_ENDPOINT` | HuggingFace 镜像（国内加速） | `https://hf-mirror.com` | ❌ |

## 各文件说明

- **`02-本地大模型.py`** — 用 ModelScope 加载本地 Qwen3 模型（默认路径 `./models/qwen3`），演示思维链（thinking/non-thinking）推理。
- **`03-openai框架部署本地大模型.py`** — 用 OpenAI 兼容接口调用 Ollama 本地模型（需本地先跑 Ollama 并拉取对应模型）。
- **`04-embadding模型下载.py`** — 下载 `BAAI/bge-m3` 嵌入模型到 `./models/embeddings`。
- **`05-RAG.py`** — RAG 核心。包含文档加载、文本分割、向量库构建/加载、检索测试，以及供外部调用的 `rag()` 函数（带全局缓存，懒加载模型）。
- **`main.py`** — 独立 RAG 演示，直接运行做一次完整问答（加载 `./chroma_db` 向量库）。
- **`server.py`** — FastAPI 服务，调用 `05-RAG.py` 的 `rag()` 提供 HTTP 接口。

## 常见问题

**Q：向量库构建慢 / 大文档跑很久？**
知识库只需构建一次，之后 `rag()` 会直接加载磁盘上的数据（懒加载 + 全局缓存），不会重复构建。

**Q：模型下载很慢？**
已默认设置 `HF_ENDPOINT=https://hf-mirror.com` 镜像。若仍慢，检查网络或改用其他镜像。

**Q：检索结果不准？**
调整 `chunk_size` 和 `chunk_overlap`：精准事实检索用小块（100–200），通用问答用中块（300–500），长篇摘要用大块（800–1500）。

**Q：为什么 `models/`、`chroma_db*/`、`.env` 没在仓库里？**
这些是体积大或含敏感信息的本地生成物，已被 `.gitignore` 忽略，clone 后需自行下载模型、构建向量库、配置 `.env`。

## 许可证

[MIT](LICENSE) © 2026 chino-y
