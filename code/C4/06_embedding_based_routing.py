import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables import RunnableLambda
from langchain_community.utils.math import cosine_similarity
import numpy as np

# 加载环境变量 (读取 .env 文件中的 KEY)
load_dotenv()

"""
=== 基于向量相似度 (Embedding) 的路由演示 ===

【核心思想】：
这就好比我们在图书馆找书。
1. 我们预先给每个“分区”（比如“川菜区”、“粤菜区”）写如果不看书名，只看简介。
2. 我们把这两个区的简介内容转换成“向量”（一串代表含义的数字）。
3. 当读者问“怎么做麻辣鱼？”时，我们也把这句话转换成“向量”。
4. 计算读者问题的向量和每个分区的向量的“相似度”（距离）。
5. 发现“麻辣鱼”和“川菜区”的向量距离最近，于是就把读者指引到川菜区。

【流程解析】：
1. 定义路由简介：为每个处理分支（Chain）写一段描述性文字。
2. 向量化路由简介：使用 Embedding 模型将这些描述文字转换为向量，并存储起来。
3. 定义处理分支：创建实际执行任务的 Chain（例如川菜大厨 Chain）。
4. 路由逻辑 (Route Function)：
    a. 接收用户问题，将其向量化。
    b. 计算 问题向量 vs 所有路由简介向量 的余弦相似度。
    c. 选出相似度最高的那一个路由。
    d. 将问题转发给该路由对应的 Chain 执行。
"""

# ==========================================
# 1. 定义路由描述 (Route Descriptions)
# ==========================================
# 这里我们定义了两个“专家”的画像。
# Embedding 模型会根据这些文字来理解每个路由是干什么的。
sichuan_route_prompt = "你是一位处理川菜的专家。用户的问题是关于麻辣、辛香、重口味的菜肴，例如水煮鱼、麻婆豆腐、鱼香肉丝、宫保鸡丁、花椒、海椒等。"
cantonese_route_prompt = "你是一位处理粤菜的专家。用户的问题是关于清淡、鲜美、原汁原味的菜肴，例如白切鸡、老火靓汤、虾饺、云吞面等。"

route_prompts = [sichuan_route_prompt, cantonese_route_prompt]
route_names = ["川菜", "粤菜"]

# ==========================================
# 2. 向量化路由描述 (Embed Route Descriptions)
# ==========================================
# 初始化嵌入模型 (Embedding Model)
# 这里使用 BAAI/bge-small-zh-v1.5，这是一个非常强大的中文嵌入模型。
# 它能把文字转换成 512 维的向量。
print("正在初始化嵌入模型...")
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")

# 将上面的“川菜简介”和“粤菜简介”转换成向量
# route_prompt_embeddings 是一个列表，包含两个向量
route_prompt_embeddings = embeddings.embed_documents(route_prompts)
print(f"已定义 {len(route_names)} 个路由: {', '.join(route_names)}")

# ==========================================
# 3. 定义不同路由的目标链 (Target Chains)
# ==========================================
# 初始化 LLM (这里使用 Google Gemini)
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash", 
    temperature=0, 
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# 定义【川菜处理链】：如果路由选择了川菜，就会执行这个 Chain
sichuan_chain = (
    PromptTemplate.from_template("你是一位川菜大厨。请用正宗的川菜做法，回答关于「{query}」的问题。")
    | llm
    | StrOutputParser()
)

# 定义【粤菜处理链】：如果路由选择了粤菜，就会执行这个 Chain
cantonese_chain = (
    PromptTemplate.from_template("你是一位粤菜大厨。请用经典的粤菜做法，回答关于「{query}」的问题。")
    | llm
    | StrOutputParser()
)

# 建立 路由名称 -> 处理链 的映射字典
route_map = { "川菜": sichuan_chain, "粤菜": cantonese_chain }
print("川菜和粤菜的处理链创建成功。\n")

# ==========================================
# 4. 创建路由函数 (The Routing Logic)
# ==========================================
def route(info):
    """
    这是路由的核心逻辑函数。
    它接收用户的输入信息 (info)，决定该去哪个 Chain，并执行它。
    """
    # a. 对用户查询进行嵌入 (Vectorize Query)
    # 将用户的问题转换成向量
    query_embedding = embeddings.embed_query(info["query"])
    
    # b. 计算相似度 (Compute Similarity)
    # 计算 用户问题向量 与 预先计算好的路由描述向量 之间的余弦相似度
    # cosine_similarity 返回的是一个矩阵，我们取第一行（因为只有一个查询）
    similarity_scores = cosine_similarity([query_embedding], route_prompt_embeddings)[0]
    
    # c. 找到最相似的路由 (Find Best Match)
    # np.argmax 返回最大值的索引 (比如是第0个还是第1个)
    chosen_route_index = np.argmax(similarity_scores)
    chosen_route_name = route_names[chosen_route_index]
    
    # 打印决策过程，方便调试
    print(f"路由决策: 检测到问题与“{chosen_route_name}”最相似 (相似度: {similarity_scores[chosen_route_index]:.4f})。")
    
    # d. 获取对应的处理链 (Select Chain)
    chosen_chain = route_map[chosen_route_name]
    
    # e. 执行选中的链并返回结果 (Execute)
    return chosen_chain.invoke(info)

# 将路由函数包装成 Runnable，这样它可以像 Chain 一样被调用
full_chain = RunnableLambda(route)


# ==========================================
# 5. 运行演示查询 (Run Demo)
# ==========================================
demo_queries = [
    "水煮鱼怎么做才嫩？",        # 应该路由到川菜
    "如何做一碗清淡的云吞面？",    # 应该路由到粤菜
    "麻婆豆腐的核心调料是什么？",  # 应该路由到川菜
]

for i, query in enumerate(demo_queries, 1):
    print(f"\n--- 问题 {i}: {query} ---")
    try:
        # 传入字典，full_chain 会自动调用 route 函数
        result = full_chain.invoke({"query": query})
        print(f"回答: {result}")
    except Exception as e:
        print(f"执行错误: {e}")
