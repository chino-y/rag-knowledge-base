import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from langchain_huggingface import HuggingFaceEmbeddings

print("开始下载")
embedding_model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3", #第一次会自动下载
        cache_folder="./models/embeddings", # 下载路径
        model_kwargs={"device": "cpu"}, # cpu 或 cuda
        encode_kwargs={"normalize_embeddings": True} # 归一化：让向量长度=1，检索更准
    )
print("结束下载")