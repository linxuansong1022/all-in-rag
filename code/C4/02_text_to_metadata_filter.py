import os
import logging
from langchain.schema import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from llama_index.llms.google_genai import GoogleGenAI

logging.basicConfig(level=logging.INFO)

# 1. 构造本地 Mock 数据 (关于龙的数据集)
# 我们直接手动创建 Document 对象，模拟数据库里的内容
docs = [
    Document(
        page_content="一条金色的中华龙在祥云间盘旋，它身形矫健，龙须飘逸，象征着皇权与吉祥。",
        metadata={
            "name": "中华金龙",
            "category": "chinese_dragon",
            "location": "Forbidden City",
            "environment": "sky",
            "time_of_day": "day",
            "danger_level": 1
        }
    ),
    Document(
        page_content="一头巨大的红色西方巨龙，栖息在火山熔岩旁，守护着它的财宝。它会喷吐烈焰，性格暴躁。",
        metadata={
            "name": "熔岩红龙",
            "category": "western_dragon",
            "location": "Volcano",
            "environment": "cave",
            "time_of_day": "night",
            "danger_level": 9
        }
    ),
    Document(
        page_content="无牙仔是一只夜煞，通体漆黑，飞行速度极快。它虽然是龙，但性格像小狗一样忠诚可爱。",
        metadata={
            "name": "无牙仔",
            "category": "movie_character",
            "location": "Berk Island",
            "environment": "coast",
            "time_of_day": "night",
            "danger_level": 3
        }
    ),
    Document(
        page_content="白龙马是西海龙王三太子变的，它通常以白马的形态出现，驮着唐僧去西天取经。",
        metadata={
            "name": "白龙马",
            "category": "chinese_dragon",
            "location": "Journey to West",
            "environment": "land",
            "time_of_day": "day",
            "danger_level": 2
        }
    ),
    Document(
        page_content="冰霜巨龙苏醒了，它吐出的不是火焰而是寒冰气息，所到之处万物冻结。",
        metadata={
            "name": "辛达苟萨",
            "category": "western_dragon",
            "location": "Northrend",
            "environment": "ice",
            "time_of_day": "night",
            "danger_level": 10
        }
    ),
]

print(f"--> 已加载 {len(docs)} 条龙的文档数据。")

# 2. 创建向量存储
# 这一步会把上面那些 page_content 转换成向量存起来
embed_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
vectorstore = Chroma.from_documents(docs, embed_model)

# 3. 配置元数据字段信息 (Schema)
# 这一步至关重要！它相当于告诉 LLM：“数据库里有哪些字段，每个字段是什么意思，有哪些取值”。
# LLM 会根据这个说明书，把用户的自然语言翻译成过滤代码。
metadata_field_info = [
    AttributeInfo(
        name="category",
        description="龙的种类，可选值: ['chinese_dragon', 'western_dragon', 'movie_character']",
        type="string", 
    ),
    AttributeInfo(
        name="location",
        description="龙出现的地点",
        type="string",
    ),
    AttributeInfo(
        name="environment",
        description="环境类型，如 sky(天空), cave(洞穴), land(陆地), ice(冰原)",
        type="string",
    ),
    AttributeInfo(
        name="time_of_day",
        description="出现的时间，'day' 或 'night'",
        type="string"
    ),
    AttributeInfo(
        name="danger_level",
        description="危险等级，1-10的整数，10最危险",
        type="integer"
    )
]

# 4. 创建自查询检索器
# 使用 Google Gemini 模型
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash", 
    temperature=0, 
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

retriever = SelfQueryRetriever.from_llm(
    llm=llm,
    vectorstore=vectorstore,
    document_contents="关于各种龙的描述信息，包含中西方神话和电影角色",
    metadata_field_info=metadata_field_info,
    enable_limit=True,
    verbose=True # 开启 verbose，能在控制台看到 LLM 生成的查询语句！
)

# 5. 执行查询示例
# 这里的查询都是自然语言，但实际上会触发复杂的后台逻辑
queries = [
    "找一种最危险的龙",                  # 测试排序/比较
    "生活在晚上的电影角色",              # 测试多条件组合 (AND)
    "不是西方龙，也不是中国龙，而且在晚上出现", # 测试否定逻辑 (NOT) 和复杂组合
    "危险等级超过5的西方龙"               # 测试数值比较
]

print(f"\n{'='*20} 开始元数据过滤测试 {'='*20}")

for query in queries:
    print(f"\n❓ 用户提问: '{query}'")
    try:
        # invoke 会自动触发 LLM -> 生成 Filter -> 查询向量库
        results = retriever.invoke(query)
        
        if results:
            print(f"✅ 找到 {len(results)} 个结果:")
            for doc in results:
                meta = doc.metadata
                print(f"   - [{meta.get('name')}] ({meta.get('category')})")
                print(f"     场景: {meta.get('time_of_day')}, 危险度: {meta.get('danger_level')}")
                print(f"     描述: {doc.page_content[:30]}...")
        else:
            print("❌ 未找到匹配的结果")
            
    except Exception as e:
        print(f"⚠️ 查询执行出错: {e}")

# 清理
vectorstore.delete_collection()