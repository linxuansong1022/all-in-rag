import os
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.llms import MockLLM

# 1. 配置全局嵌入模型
# 使用 MockLLM 绕过 LLM 检查，专注于检索
Settings.embed_model = HuggingFaceEmbedding("BAAI/bge-small-zh-v1.5")
Settings.llm = MockLLM() 

# 2. 创建示例文档
texts = [
    "张三是法外狂徒",
    "LlamaIndex是一个用于构建和查询私有或领域特定数据的框架。",
    "它提供了数据连接、索引和查询接口等工具。"
]
docs = [Document(text=t) for t in texts]

# 3. 创建索引并持久化到本地
index = VectorStoreIndex.from_documents(docs)
persist_path = "./llamaindex_index_store"
index.storage_context.persist(persist_dir=persist_path)
print(f"LlamaIndex 索引已保存至: {persist_path}")

# 4. 执行检索 (只找文档，不调用 AI 回答)
# similarity_top_k=1 表示只找最像的一个
retriever = index.as_retriever(similarity_top_k=1)
results = retriever.retrieve("谁是法外狂徒")

print("\n--- 检索结果 ---")
for node in results:
    print(f"内容: {node.text}")
    print(f"相似度分数: {node.score:.4f}")
