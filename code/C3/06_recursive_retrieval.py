# ==============================================================================
# 【笔记：06 递归检索 - 代码生成版】
# 核心思想：
#   这是“递归检索 (Recursive Retrieval)”的演示。
#   系统建立了两层结构：
#   1. 第一层（摘要）：用来“路由”。比如“这是1994年的表”。
#   2. 第二层（引擎）：用来“执行”。这里挂载的是 PandasQueryEngine。
#
# 绝招 (PandasQueryEngine)：
#   利用 LLM 强大的代码生成能力，将用户的自然语言问题（如“评分最少是多少”）
#   转化为 Python/Pandas 代码并在内存中执行。
#
# 适用场景：
#   需要对表格数据进行“计算、排序、统计”等复杂操作时（例如：求平均值、找最大值）。
#   普通的文本检索做不到数学计算，必须用这种方法。
# ==============================================================================

import os
import pandas as pd
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.schema import IndexNode
from llama_index.experimental.query_engine import PandasQueryEngine
from llama_index.core.retrievers import RecursiveRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.llms.deepseek import DeepSeek
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# 1. 配置模型 (使用你的 DeepSeek Key)
# 生产环境建议使用 os.getenv("DEEPSEEK_API_KEY")
Settings.llm = DeepSeek(model="deepseek-chat", api_key="sk-17d7a71fae104eaea98f4d2a5e619082")
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-zh-v1.5")

# 2. 加载数据并为每个工作表创建查询引擎和摘要节点
excel_file = '../../data/C3/excel/movie.xlsx'
xls = pd.ExcelFile(excel_file)

df_query_engines = {}
all_nodes = []

print("正在分析 Excel 工作表并构建引擎...")
for sheet_name in xls.sheet_names:
    df = pd.read_excel(xls, sheet_name=sheet_name)
    
    # 为当前工作表（DataFrame）创建一个 PandasQueryEngine
    # 这里的关键：让 DeepSeek 来写代码查询表格
    query_engine = PandasQueryEngine(df=df, llm=Settings.llm, verbose=True)
    
    # 为当前工作表创建一个摘要节点（IndexNode）
    year = sheet_name.replace('年份_', '')
    summary = f"这个表格包含了年份为 {year} 的电影信息，可以用来回答关于这一年电影的具体问题。"
    node = IndexNode(text=summary, index_id=sheet_name)
    all_nodes.append(node)
    
    # 存储工作表名称到其查询引擎的映射
    df_query_engines[sheet_name] = query_engine

# 3. 创建顶层索引（只包含摘要节点）
vector_index = VectorStoreIndex(all_nodes)

# 4. 创建递归检索器
# 4.1 创建顶层检索器，用于在摘要节点中检索
vector_retriever = vector_index.as_retriever(similarity_top_k=1)

# 4.2 创建递归检索器
recursive_retriever = RecursiveRetriever(
    "vector",
    retriever_dict={"vector": vector_retriever},
    query_engine_dict=df_query_engines,
    verbose=True,
)

# 5. 创建查询引擎
query_engine = RetrieverQueryEngine.from_args(recursive_retriever)

# 6. 执行查询
query = "1994年评分人数最少的电影是哪一部？"
print(f"\n查询: {query}")
response = query_engine.query(query)

print("\n" + "="*50)
print(f"最终回答: {response}")
