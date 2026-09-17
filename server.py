"""
RAG 知识库问答系统 —— FastAPI Web 服务器

使用 FastAPI 框架，实现一个简单的 Web 服务器
使用 uvicorn 启动，修改代码后会自动重启（reload=True）

启动方式：
    python server.py
    或
    uvicorn server:app --host 0.0.0.0 --port 8000 --reload

测试接口：
    http://localhost:8000/ai_chat?query=什么是RAG？
"""

import os
import sys
import importlib

from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware


# 确保能导入同目录下的 05-RAG 模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 导入 RAG 模块中的 rag 函数（文件名以数字开头，用 importlib）
rag_module = importlib.import_module("05-RAG")
rag = rag_module.rag

# ============================================================
# 创建 FastAPI 应用
# ============================================================
app = FastAPI(
    title="RAG 知识库问答系统",
    description="基于 LangChain + ChromaDB + DeepSeek 的检索增强生成系统",
    version="1.0.0"
)

# 跨域配置（允许所有来源访问）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # 生产环境建议改为具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# 注册路由
# ============================================================

@app.get("/")
def root():
    """健康检查 / 首页"""
    return {
        "status": "ok",
        "service": "RAG 知识库问答系统",
        "usage": "GET /ai_chat?query=你的问题"
    }


@app.get("/ai_chat")
def ai_chat(query: str = ""):
    """
    RAG 问答接口

    参数:
        query: 用户提问内容

    返回:
        {"ai_text": "LLM 生成的回答内容"}
    """
    if not query.strip():
        return {"ai_text": "请提供 query 参数，例如: /ai_chat?query=什么是RAG？"}

    print(f"[请求] 用户提问: {query}")
    ai_text = rag(query)
    print(f"[回答] {ai_text[:100]}...")
    return {"ai_text": ai_text}


# ============================================================
# 启动服务器
# ============================================================
if __name__ == "__main__":
    import uvicorn

    print("="*50)
    print("  RAG 知识库问答系统 —— Web 服务")
    print("="*50)
    print(f"  接口地址: http://localhost:8000/ai_chat?query=你的问题")
    print(f"  文档地址: http://localhost:8000/docs")
    print("="*50)

    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=True      # 代码修改后自动重启
    )
