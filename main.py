# 1.设置环境变量(一般用下面第2种方法)
import os

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv


# 从 .env 文件加载环境变量（密钥不要硬编码在代码里，避免泄露）
load_dotenv(dotenv_path=".env", override=True)

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# 读取环境变量
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL_NAME = os.getenv("DEEPSEEK_MODEL_NAME", "deepseek-chat")

def load_llm():
    llm=ChatOpenAI(model=DEEPSEEK_MODEL_NAME,api_key=DEEPSEEK_API_KEY,base_url=DEEPSEEK_BASE_URL)
    return llm
# if __name__ == "__main__":
#     llm = load_llm()
#     ai_res=llm.invoke("你是谁?")
#     print(ai_res)


# 9.加载持久化的向量库
def load_vectorstore():
    """从磁盘加载之前构建好的 ChromaDB 向量库"""
    embeddings = HuggingFaceEmbeddings(
        model_name="./models/embeddings",
        cache_folder="./models/embeddings",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    vectorstore = Chroma(
        collection_name="teaching_docs",
        embedding_function=embeddings,
        persist_directory="./chroma_db"
    )
    print(f"已加载向量库，向量数: {vectorstore._collection.count()}")
    return vectorstore

# 10.增强生成 —— RAG + llm 生成答案
def demo_rag_generation(vectorstore):
    print("增强生成（RAG + DeepSeek）—— 最后一公里")
    query = "最近我感到很焦虑，难以入睡。"

    # --- Step 1: 检索 ---
    docs = vectorstore.similarity_search(query, k=3)
    context = "\n".join([f"- {doc.page_content[:200]}..." for doc in docs])

    # --- Step 2: 拼接 Prompt ---
    prompt = f"""你是一个知识问答助手。请根据以下参考资料回答用户问题。

【参考资料】
{context}

【用户问题】
{query}

【要求】
1. 仅使用参考资料中的信息回答
2. 如果资料中没有相关信息，请如实告知"""

    #  【检索到的参考资料】
    for i, doc in enumerate(docs):
        print(f"  [{i+1}] {doc.page_content[:100]}...")
    #  【发送给 LLM 的完整 Prompt】
    print(prompt)

    # --- Step 3: 调用 llm 生成答案 ---
    llm = ChatOpenAI(
        model="deepseek-chat",
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
        temperature=0.3,
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    print(">>> llm 回答：")
    print(response.content)
demo_rag_generation(load_vectorstore())