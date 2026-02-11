import os
from dotenv import load_dotenv
from llama_index.core.node_parser import SentenceWindowNodeParser, SentenceSplitter
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.postprocessor import MetadataReplacementPostProcessor

# 加载环境变量
load_dotenv()

# 1. 配置模型
Settings.llm = GoogleGenAI(model="models/gemini-2.0-flash", api_key=os.getenv("GOOGLE_API_KEY"))
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en")

# 2. 加载文档
print("正在加载 PDF 文档...")
documents = SimpleDirectoryReader(
    input_files=["../../data/C3/pdf/IPCC_AR6_WGII_Chapter03.pdf"]
).load_data()

# 3. 创建节点与构建索引
print("正在构建索引（这可能需要一点时间）...")
# 3.1 句子窗口索引
node_parser = SentenceWindowNodeParser.from_defaults(
    window_size=3,#记住前面3句和后面3句 一共7句话 的窗口
    window_metadata_key="window", #把这7句话存在metadata 的 "window" 字段里
    original_text_metadata_key="original_text",
)
#这个索引用了上面那个“带口袋”的解析器
sentence_nodes = node_parser.get_nodes_from_documents(documents)
sentence_index = VectorStoreIndex(sentence_nodes)

# 3.2 常规分块索引 (基准) 每512字一切 作为基准对照组
base_parser = SentenceSplitter(chunk_size=512)
base_nodes = base_parser.get_nodes_from_documents(documents)
base_index = VectorStoreIndex(base_nodes)

# 4. 构建查询引擎
# 注意：我们这里主要看检索出来的原文，所以配置为显示源节点
sentence_query_engine = sentence_index.as_query_engine(
    similarity_top_k=1,
    node_postprocessors=[ #后处理器，当检索系统通过向量匹配到最像的那一句话（Node）后，在把这个 Node 交给 AI 之前，这个处理器会横插一杠子。它会去 Node 的口袋（metadata）里找 window 这个 key，把里面的长段落拿出来，直接覆盖掉原来的那一句话
        MetadataReplacementPostProcessor(target_metadata_key="window")
    ],
)
base_query_engine = base_index.as_query_engine(similarity_top_k=1)

# 5. 执行查询并对比结果
query = "What are the concerns surrounding the AMOC?"
print(f"\n查询问题: {query}")

print("\n" + "="*50)
print("【1. 句子窗口检索效果】")
window_response = sentence_query_engine.query(query)
# 打印检索到的“窗口”内容
source_text = window_response.source_nodes[0].node.get_content()
print(f"检索到的完整窗口内容（包含上下文）:\n\n{source_text}")

print("\n" + "="*50)
print("【2. 常规分块检索效果】")
base_response = base_query_engine.query(query)
# 打印普通分块内容
base_text = base_response.source_nodes[0].node.get_content()
print(f"检索到的普通分块内容:\n\n{base_text[:500]}...")
#   2. 句子窗口检索（Sentence Window Retrieval）
#    * 做法：
#        1. 切分（细切）：先把文章按照“句号”切成一句话一句话。比如：“AMOC 将在 21 世纪衰退。”
#        2. 向量化（精细化）：只拿这一句话去算向量。
#            * 好处：这句话的向量非常纯粹，就是关于“AMOC 衰退”的，检索命中率极高！
#        3. 偷藏私货（Metadata）：在存这句话的时候，系统会在它的口袋里（Metadata）偷偷塞一张纸条。
#            * 纸条上写着：“这一句的前面 3 句是 A、B、C；后面 3 句是 X、Y、Z。”
#        4. 检索与替换（变身）：
#            * 当用户问“AMOC 会怎样？”时，系统精准地找到了那句“AMOC 将在 21 世纪衰退”。
#            * 关键一步：在把这句话交给 AI 之前，系统触发了 `MetadataReplacementPostProcessor`（后处理器）。
#            * 它把找到的那短短的一句话扔掉，替换成了纸条上记录的整整 7 句话（前3 + 本句 + 后3）。