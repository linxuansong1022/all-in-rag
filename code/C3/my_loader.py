from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.llms import MockLLM

# 1. 配置必要的模型 (因为加载索引时需要用同样的模型来把你的问题变成向量)
print("正在加载模型...")
Settings.embed_model = HuggingFaceEmbedding("BAAI/bge-small-zh-v1.5")
Settings.llm = MockLLM() # 这里我们不需要 LLM，只要检索

# 2. 从本地目录加载索引
persist_path = "./llamaindex_index_store"
print(f"从 {persist_path} 加载索引...")

# 重建存储上下文
storage_context = StorageContext.from_defaults(persist_dir=persist_path)
# 加载索引
loaded_index = load_index_from_storage(storage_context)

print("索引加载成功！")

# 3. 执行检索
query = "谁是法外狂徒"
print(f"\n开始检索: '{query}'")

retriever = loaded_index.as_retriever(similarity_top_k=1)
results = retriever.retrieve(query)

print("\n--- 检索结果 ---")
for node in results:
    print(f"内容: {node.text}")
    print(f"相似度: {node.score:.4f}")
