# 0.导入依赖库
from pathlib import Path
import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    UnstructuredMarkdownLoader,
    Docx2txtLoader,
)

# 4.设置文档存放目录
DOCS_DIR = Path("./docs") #语料存放路径
CHROMA_DIR = Path("./chroma_db_docs") #词向量数据库路径

# 环境变量配置（从 .env 加载，密钥不要硬编码）
load_dotenv(dotenv_path=".env", override=True)
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL_NAME = os.getenv("DEEPSEEK_MODEL_NAME", "deepseek-chat")

# 5.创建示例文档
# def create_sample_docs():
#     """
#     为了方便教学，没有文档，使用这段代码自动创建两个示例文件。
#     在实际项目中，会直接使用自己的 PDF、DOCX、Markdown 等文件。
#     """
#     DOCS_DIR.mkdir(exist_ok=True)
#     # --- 示例 1：纯文本文件（.txt）---
#     # 这是最简单、最常见的文档格式。
#     txt_content = """RAG（检索增强生成）技术详解
#
# RAG 全称是 Retrieval-Augmented Generation，由 Meta AI 在 2020 年提出。
#
# 核心思想：
# RAG 将信息检索系统与大型语言模型结合。当用户提问时，系统先从知识库中检索相关文档，
# 然后将这些文档作为上下文一起提供给 LLM，让 LLM 基于这些私有知识生成答案。
#
# RAG 的三大优势：
# 1. 知识实时更新：无需重新训练模型，只需更新知识库即可
# 2. 减少幻觉：LLM 基于真实文档回答，大幅减少编造内容的概率
# 3. 可溯源：每段回答都能追溯到具体的来源文档
#
# ChromaDB 简介：
# ChromaDB 是一个开源的向量数据库，专为 AI 应用设计。它支持：
# - 多种 Embedding 模型
# - 高效的近似最近邻搜索（ANN）
# - 元数据过滤
# - 本地持久化存储
# """
#     with open(DOCS_DIR / "rag_intro.txt", "w", encoding="utf-8") as f:
#         f.write(txt_content)
#
#     # --- 示例 2：Markdown 文件（.md）---
#     # Markdown 是技术文档、wiki 的常见格式，LangChain 可以正确解析标题层级。
#     md_content = """# LangChain 框架入门
#
# ## 什么是 LangChain？
#
# LangChain 是一个用于构建 LLM 应用的框架，提供了模块化的组件来简化开发。
#
# ## 核心组件
#
# | 组件 | 说明 |
# |------|------|
# | Document Loaders | 从各种来源加载文档 |
# | Text Splitters | 将文档分割成合适大小的块 |
# | Embeddings | 将文本转换为向量 |
# | Vector Stores | 存储和检索向量 |
# | Retrievers | 从向量库中检索相关文档 |
# | Chains | 将多个组件串联成 pipeline |
#
# ## RAG 在 LangChain 中的实现
#
# ```python
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_chroma import Chroma
#
# # 1. 加载文档
# loader = TextLoader("docs/rag_intro.txt")
# docs = loader.load()
#
# # 2. 分割文本
# splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
# chunks = splitter.split_documents(docs)
#
# # 3. 构建向量库
# vectorstore = Chroma.from_documents(chunks, embeddings)
# ```
# """
#     with open(DOCS_DIR / "langchain_guide.md", "w", encoding="utf-8") as f:
#         f.write(md_content)
#
#     print(">>> 已创建示例文档:")
#     print("提示：你也可以直接把文件放入 ./docs 目录")
#create_sample_docs()

# 6.使用文档加载器 —— 把读取文件为Document 对象
def load_all_documents():
    """
    遍历 ./docs 目录，根据文件后缀选择合适的 Loader 加载。
    所有 Loader 返回 LangChain 的 Document 对象，包含：
      - page_content: 文档的文本内容
      - metadata: 元数据（来源文件、页码等）
    """

    all_docs = []

    # --- TXT 文件：最基础，直接读取 ---
    for file in DOCS_DIR.glob("*.txt"):
        loader = TextLoader(str(file), encoding="utf-8")
        content = loader.load()
        all_docs.extend(loader.load())

    # --- Markdown 文件：保留标题层级结构 ---
    for file in DOCS_DIR.glob("*.md"):
        loader = UnstructuredMarkdownLoader(str(file))
        content=loader.load()
        all_docs.extend(loader.load())

    x=f"有{len(all_docs)}个Document资料    第一个的内容:{all_docs[0].page_content[:20]}   第一个源信息:{all_docs[0].metadata}"
    print(x)
    return all_docs
# load_all_documents()  # 已移到 if __name__ == "__main__" 中统一调用


# 8.构建持久化向量知识库
def build_vectorstore(documents, chunk_size=400, chunk_overlap=80):
    print("构建持久化向量知识库")
    # 加载 Embedding 模型
    print("Step 1/3: 加载 Embedding 模型...")
    embeddings = HuggingFaceEmbeddings(
        model_name="./models/embeddings",
        cache_folder="./models/embeddings",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )


    # 分割文档
    print("Step 2/3: 分割文档...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "，", " ", ""]
    )
    chunks = splitter.split_documents(documents)
    print(f" 完成。{len(documents)} 个文档 → {len(chunks)} 个文本块")

    # 预览前几个块
    print("--- 前 6 个文本块预览 ---")
    for chunk in chunks[:6]:
        source = Path(chunk.metadata.get("source", "")).name
        print(f"  [{source}] {chunk.page_content[:60]}...  ({len(chunk.page_content)} 字符)")
    if len(chunks) > 6:
        print(f"  ... (还有 {len(chunks) - 6} 个)")

    # 4.3 向量化 & 存储
    print("Step 3/3: 向量化并存入 ChromaDB...")
    # (每个文本块 → Embedding 模型 → 512维向量 → 写入磁盘)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="teaching_docs",  # 知识库名称，加载时要用一样的名字
        persist_directory=str(CHROMA_DIR)  # 持久化路径（与加载时保持一致）
    )
    print(f"完成！存储位置: {CHROMA_DIR.absolute()}")
    print(f"知识库: teaching_docs | 向量数: {vectorstore._collection.count()}")
    return vectorstore
# build_vectorstore(load_all_documents())



# 9.检索测试
def test_retrieval(vectorstore):
    """用几个典型问题测试知识库的检索效果"""
    print("检索效果测试")
    query = "最近我感到很焦虑，难以入睡。"
    results = vectorstore.similarity_search(query, k=5)
    print(results)


# ============================================================
# 10. 加载已有知识库 —— 生产模式（构建一次，反复使用）
# ============================================================
def load_existing_vectorstore():
    """
    这是生产环境中最重要的模式：
    知识库构建非常耗时（大文档可能跑几小时），不能每次都重建。
    通过 persist_directory + collection_name 直接加载已有数据。

    【关键认知】
      - 知识库只需要构建一次！
      - 之后每次启动程序，直接用 Chroma(...) 加载磁盘上的数据。
      - 这和 MySQL 的启动逻辑一样——数据在磁盘，启动时加载即可。

    注意：这里用的是 Chroma(...) 而不是 from_documents(...)
      - from_documents = 创建新的
      - Chroma(...)    = 加载已有的
    """
    print("\n" + "="*60)
    print("加载已有知识库（生产模式）")
    print("="*60)

    if not CHROMA_DIR.exists():
        print("[错误] 知识库目录不存在，请先运行构建流程。")
        return None

    embeddings = HuggingFaceEmbeddings(
        model_name="./models/embeddings",
        cache_folder="./models/embeddings",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    # 注意：这里用的是 Chroma(...) 而不是 from_documents(...)
    vectorstore = Chroma(
        embedding_function=embeddings,
        collection_name="teaching_docs",      # 必须和构建时的名字一致
        persist_directory=str(CHROMA_DIR)     # 必须和构建时的路径一致
    )

    print(f">>> 成功加载已有知识库！向量数量: {vectorstore._collection.count()}")
    return vectorstore


# ============================================================
# 11. 对比不同文本分割策略
# ============================================================
def compare_split_strategies(documents):
    """
    对比不同的 chunk_size 和 chunk_overlap 组合，
    帮助理解参数如何影响分割结果，并推荐最佳策略。

    策略参考：
      - 精准事实检索 → 小块 (100-200)
      - 通用问答       → 中块 (300-500)  ← 推荐
      - 长篇摘要       → 大块 (800-1500)
    """
    print("\n" + "="*60)
    print("对比文本分割策略")
    print("="*60)

    strategies = [
        {"name": "精准检索(小块)",   "size": 150, "overlap": 30},
        {"name": "通用问答(推荐)",   "size": 400, "overlap": 80},
        {"name": "长篇摘要(大块)",   "size": 1000, "overlap": 200},
    ]

    best = {"size": 400, "overlap": 80}  # 默认推荐

    for s in strategies:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=s["size"],
            chunk_overlap=s["overlap"],
            separators=["\n\n", "\n", "。", "，", " ", ""]
        )
        chunks = splitter.split_documents(documents)
        avg_len = sum(len(c.page_content) for c in chunks) / max(len(chunks), 1)
        print(f"  {s['name']}: chunk_size={s['size']}, overlap={s['overlap']}")
        print(f"    → {len(chunks)} 个文本块, 平均长度 {avg_len:.0f} 字符")

    print(f"\n>>> 推荐策略: chunk_size={best['size']}, chunk_overlap={best['overlap']}")
    return best


# ============================================================
# 12. 增强生成 —— RAG + LLM 生成答案
# ============================================================
# def demo_rag_generation(vectorstore):
#     """完整的 RAG 流程：检索 + 拼接 Prompt + LLM 生成"""
#     print("\n" + "="*60)
#     print("增强生成（RAG + DeepSeek）—— 最后一公里")
#     print("="*60)
#
#     query = "什么是 RAG？它有什么优势？"
#
#     # --- Step 1: 检索 ---
#     print("\n[Step 1] 检索相关文档...")
#     docs = vectorstore.similarity_search(query, k=3)
#
#     # 显示检索结果
#     print("--- 检索到的参考资料 ---")
#     for i, doc in enumerate(docs):
#         source = Path(doc.metadata.get("source", "")).name
#         print(f"  [{i+1}] [{source}] {doc.page_content[:80]}...")
#
#     context = "\n\n".join([f"【来源 {i+1}】\n{doc.page_content}" for i, doc in enumerate(docs)])
#
#     # --- Step 2: 拼接 Prompt ---
#     prompt = f"""你是一个知识问答助手。请根据以下参考资料回答用户问题。
#
# 【参考资料】
# {context}
#
# 【用户问题】
# {query}
#
# 【要求】
# 1. 仅使用参考资料中的信息回答
# 2. 如果资料中没有相关信息，请如实告知
# 3. 回答要结构清晰，列出要点"""
#
#     print(f"\n[Step 2] 拼接 Prompt（共 {len(prompt)} 字符）")
#
#     # --- Step 3: 调用 LLM 生成答案 ---
#     print("[Step 3] 调用 DeepSeek 生成答案...\n")
#     llm = ChatOpenAI(
#         model=DEEPSEEK_MODEL_NAME,
#         api_key=DEEPSEEK_API_KEY,
#         base_url=DEEPSEEK_BASE_URL,
#         temperature=0.3,
#     )
#     response = llm.invoke([HumanMessage(content=prompt)])
#     print(">>> LLM 回答：")
#     print("-" * 40)
#     print(response.content)
#     print("-" * 40)
#

# ============================================================
# 13. 完整 RAG 函数 —— 供外部调用（如 Web 服务器）
# ============================================================
# 全局缓存：避免每次请求都重新加载模型
_vectorstore_cache = None
_llm_cache = None


def rag(query: str) -> str:
    """
    完整的 RAG 查询函数，供 FastAPI 等外部服务调用。
    内部使用缓存避免重复加载模型。
    """
    global _vectorstore_cache, _llm_cache

    # 懒加载向量库（只加载一次）
    if _vectorstore_cache is None:
        print("[RAG] 首次加载向量库...")
        _vectorstore_cache = load_existing_vectorstore()
        if _vectorstore_cache is None:
            # 回退：知识库不存在时，用 ./docs 下的文档现场构建
            print("[RAG] 向量库不存在，尝试从 ./docs 构建...")
            docs = load_all_documents()
            _vectorstore_cache = build_vectorstore(docs)

    # 懒加载 LLM（只初始化一次）
    if _llm_cache is None:
        print("[RAG] 首次加载 LLM...")
        _llm_cache = ChatOpenAI(
            model=DEEPSEEK_MODEL_NAME,
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL,
            temperature=0.3,
        )

    # 检索
    docs = _vectorstore_cache.similarity_search(query, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])

    # 拼接 Prompt
    prompt = f"""你是一个知识问答助手。请根据以下参考资料回答用户问题。

【参考资料】
{context}

【用户问题】
{query}

【要求】
1. 优先使用参考资料中的信息回答
2. 如果资料中没有相关信息，如实告知并基于你的知识回答
3. 回答要结构清晰"""

    # 生成
    response = _llm_cache.invoke([HumanMessage(content=prompt)])
    return response.content


# ============================================================
# 主流程入口
# ============================================================
if __name__ == "__main__":
    print("="*60)
    print("  RAG 知识库问答系统 —— 完整演示")
    print("="*60)

    # 1. 用 Loader 加载所有文档
    print("\n[1/3] 加载文档...")
    raw_docs = load_all_documents()
    print(f">>> 共加载 {len(raw_docs)} 个原始文档")

    if not raw_docs:
        print("[错误] 没有找到任何可加载的文档！请检查 ./docs 目录。")
        exit()

    # 2. 用默认参数构建持久化知识库（chunk_size=400, chunk_overlap=80）
    print("\n[2/3] 构建持久化知识库...")
    vectorstore = build_vectorstore(raw_docs)

    # 3. 测试检索效果
    print("\n[3/3] 测试检索...")
    test_retrieval(vectorstore)

    print("\n>>> 演示完成！知识库已持久化到磁盘，下次可用 load_existing_vectorstore() 直接加载。")
    print("    完整 RAG 问答请运行 server.py 启动 Web 服务，或调用 rag() 函数。")


"""
=================================================================
 本课核心知识总结
=================================================================

Document Loaders：LangChain 支持 100+ 种数据源
  所有 Loader 都返回统一的 Document 对象

Text Splitting：chunk_size 和 chunk_overlap 直接影响检索质量
  精准事实检索 → 小块 (100-200)
  通用问答       → 中块 (300-500)  ← 推荐
  长篇摘要       → 大块 (800-1500)

Persistence：知识库持久化到磁盘，构建一次、反复使用
  from_documents() → 构建
  Chroma(...)      → 加载

Collection：一个知识库 = 一个 collection
  可以同时管理多个知识库（如：公司制度、产品文档、技术文档）
=================================================================
"""