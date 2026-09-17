# RAG 知识库问答系统

基于 **LangChain + ChromaDB + DeepSeek** 的检索增强生成（Retrieval-Augmented Generation）学习与演示项目。

## 功能

- 📄 多格式文档加载：TXT / Markdown / PDF / DOCX
- ✂️ 文本分割（RecursiveCharacterTextSplitter，可调 chunk_size / chunk_overlap）
- 🧬 向量化与持久化：HuggingFace Embedding（`BAAI/bge-m3`）+ ChromaDB
- 🔍 相似度检索
- 🤖 RAG 增强生成：检索 → 拼接 Prompt → DeepSeek 生成
- 🌐 FastAPI Web 服务，提供 HTTP 问答接口

## 目录结构

```
rag-study/
├── 02-本地大模型.py            # 用 ModelScope 加载本地 Qwen3 模型推理
├── 03-openai框架部署本地大模型.py # 用 OpenAI 兼容框架调用 Ollama 本地模型
├── 04-embadding模型下载.py      # 下载 BAAI/bge-m3 嵌入模型到 models/embeddings
├── 05-RAG.py                  # RAG 完整流程（核心），导出 rag() 供服务调用
├── main.py                    # 独立的 RAG 演示脚本
├── server.py                  # FastAPI Web 服务入口
├── requirements.txt           # 依赖清单
├── .env.example               # 环境变量模板（复制为 .env 使用）
├── docs/                      # 语料目录（放入你自己的文档）
│   ├── rag_intro.txt
│   └── langchain_guide.md
├── models/                    # 本地模型（不提交，需自行下载）
└── chroma_db_docs/            # 向量库（不提交，由 docs 构建）
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
cp .env.example .env   # Windows: copy .env.example .env
```

编辑 `.env`，填入你的 DeepSeek API Key：

```
DEEPSEEK_API_KEY=sk-你的密钥
```

### 3. 准备嵌入模型

```bash
python 04-embadding模型下载.py   # 下载 BAAI/bge-m3 到 models/embeddings
```

### 4. 准备语料

把你的 TXT / Markdown / PDF / DOCX 文件放到 `docs/` 目录即可。

> 说明：`docs/emotion.txt`（心理疏导对话语料）因涉及隐私未随仓库提交，请自行准备语料。

### 5. 构建知识库并测试

```bash
python 05-RAG.py   # 加载文档 → 构建向量库 → 检索测试
```

### 6. 启动 Web 服务

```bash
python server.py
# 或
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```

测试接口：

```
http://localhost:8000/ai_chat?query=什么是RAG？
http://localhost:8000/docs   # Swagger 文档
```

## 环境变量

| 变量 | 说明 | 默认值 |
|---|---|---|
| `DEEPSEEK_API_KEY` | DeepSeek API 密钥（必填） | — |
| `DEEPSEEK_BASE_URL` | DeepSeek API 地址 | `https://api.deepseek.com` |
| `DEEPSEEK_MODEL_NAME` | 模型名称 | `deepseek-chat` |
| `HF_ENDPOINT` | HuggingFace 镜像 | `https://hf-mirror.com` |

## 注意事项

- **不要提交 `.env`**：密钥已被 `.gitignore` 忽略，请勿把真实密钥写进代码。
- **模型与向量库不提交**：`models/`（数 GB）和 `chroma_db*/` 均为本地生成物，clone 后需自行下载模型、构建向量库。
