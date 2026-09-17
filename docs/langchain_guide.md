# LangChain 框架入门

## 什么是 LangChain？

LangChain 是一个用于构建 LLM 应用的框架，提供了模块化的组件来简化开发。

## 核心组件

| 组件 | 说明 |
|------|------|
| Document Loaders | 从各种来源加载文档 |
| Text Splitters | 将文档分割成合适大小的块 |
| Embeddings | 将文本转换为向量 |
| Vector Stores | 存储和检索向量 |
| Retrievers | 从向量库中检索相关文档 |
| Chains | 将多个组件串联成 pipeline |

## RAG 在 LangChain 中的实现

```python
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# 1. 加载文档
loader = TextLoader("docs/rag_intro.txt")
docs = loader.load()

# 2. 分割文本
splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# 3. 构建向量库
vectorstore = Chroma.from_documents(chunks, embeddings)
```
