import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_google_genai import ChatGoogleGenerativeAI

# 导入ColBERT重排器需要的模块
from langchain.retrievers.document_compressors.base import BaseDocumentCompressor
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain_core.documents import Document
from typing import Sequence
import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F

# 加载环境变量
load_dotenv()

"""
=== Rerank (重排序) 与 Refine (精炼) 流程演示 ===

【核心思想】：
这是 RAG 系统中提升准确率的“进阶连招”。
普通的检索就像大海捞针，捞上来 20 根针，可能只有 5 根是有用的，而且每面上还带着很多海草（无关信息）。
这个脚本演示了如何把这些“针”洗干净，并且挑出最锋利的那几根。

【流程解析】：
1. 基础检索 (Base Retrieval):
   先用向量检索（FAISS + BGE）快速捞出 Top 20 篇文档。这步追求速度，但也允许有一定的误检。

2. 重排序 (Reranking - ColBERT):
   使用 ColBERT 模型对这 20 篇文档进行“精读”。
   ColBERT 会仔细比对查询词和文档词的交互，给每个文档重新打分。
   我们只保留分数最高的 Top 5。这步追求精度。

3. 精炼 (Refining/Compression - LLM):
   把这 Top 5 篇文档丢给 LLM (Gemini)。
   要求 LLM：“只保留和问题相关的句子，删掉无关的废话。”
   这一步能极大减少最终输入给大模型的上下文长度（节约 Token），并减少干扰信息（减少幻觉）。

4. 最终生成 (Final Generation):
   拿着这 5 篇“洗干净”的精华文档，去回答用户的问题。
"""

# ==========================================
# 1. 自定义 ColBERT 重排器 (The Reranker)
# ==========================================
class ColBERTReranker(BaseDocumentCompressor):
    """
    ColBERT 重排器
    这是一个自定义的 LangChain 组件。它利用 ColBERT 模型计算 查询(Query) 和 文档(Document) 的精细相似度。
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # 加载 BERT 模型作为 ColBERT 的底层
        # ColBERT 的核心就是通过 BERT 对每个 Token 生成向量
        model_name = "bert-base-uncased" # 这里使用英文模型演示，中文建议换用中文模型
        
        # 加载模型和分词器
        # 注意：这里使用 object.__setattr__ 是为了绕过 Pydantic 的验证机制
        object.__setattr__(self, 'tokenizer', AutoTokenizer.from_pretrained(model_name))
        object.__setattr__(self, 'model', AutoModel.from_pretrained(model_name))
        self.model.eval() # 设置为评估模式，不进行训练
        print(f"ColBERT模型加载完成")

    def encode_text(self, texts):
        """
        文本编码函数：将文本转换成 ColBERT 需要的张量格式
        """
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",  # 返回 PyTorch 张量
            padding=True,         # 填充至最长长度
            truncation=True,      # 超长截断
            max_length=128        # 最大长度限制
        )

        with torch.no_grad(): # 不计算梯度，节省内存
            outputs = self.model(**inputs)

        # 获取最后一层隐藏状态作为嵌入 (Embeddings)
        embeddings = outputs.last_hidden_state
        # 归一化 (Normalize)，让计算余弦相似度更方便
        embeddings = F.normalize(embeddings, p=2, dim=-1)

        return embeddings

    def calculate_colbert_similarity(self, query_emb, doc_embs, query_mask, doc_masks):
        """
        ColBERT 相似度计算核心 (MaxSim 操作)
        这是 ColBERT 比普通向量检索准的关键！
        它不是算两个句子的整体相似度，而是算 Query 中每个 Token 和 Document 中每个 Token 的最大相似度之和。
        """
        scores = []

        for i, doc_emb in enumerate(doc_embs):
            doc_mask = doc_masks[i:i+1]

            # 计算相似度矩阵 (Query Token x Doc Token)
            # 这一步计算了 Query 中每个词和 Doc 中每个词的相似度
            similarity_matrix = torch.matmul(query_emb, doc_emb.unsqueeze(0).transpose(-2, -1))

            # 应用掩码 (Mask)，忽略填充(Padding)部分的计算
            doc_mask_expanded = doc_mask.unsqueeze(1)
            similarity_matrix = similarity_matrix.masked_fill(~doc_mask_expanded.bool(), -1e9)

            # MaxSim 操作：对于 Query 中的每个词，找到它在 Doc 中最相似的那个词的得分
            max_sim_per_query_token = similarity_matrix.max(dim=-1)[0]

            # 再次应用查询掩码
            query_mask_expanded = query_mask.unsqueeze(0)
            max_sim_per_query_token = max_sim_per_query_token.masked_fill(~query_mask_expanded.bool(), 0)

            # 求和得到最终分数：所有 Query Token 的最大相似度之和
            colbert_score = max_sim_per_query_token.sum(dim=-1).item()
            scores.append(colbert_score)

        return scores

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks=None,
    ) -> Sequence[Document]:
        """
        LangChain 标准接口：接收一堆文档，返回重排后的文档
        """
        if len(documents) == 0:
            return documents

        # 1. 编码查询 (Encode Query)
        query_inputs = self.tokenizer(
            [query],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        )

        with torch.no_grad():
            query_outputs = self.model(**query_inputs)
            query_embeddings = F.normalize(query_outputs.last_hidden_state, p=2, dim=-1)

        # 2. 编码文档 (Encode Documents)
        doc_texts = [doc.page_content for doc in documents]
        doc_inputs = self.tokenizer(
            doc_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        )

        with torch.no_grad():
            doc_outputs = self.model(**doc_inputs)
            doc_embeddings = F.normalize(doc_outputs.last_hidden_state, p=2, dim=-1)

        # 3. 计算分数 (Calculate Scores)
        scores = self.calculate_colbert_similarity(
            query_embeddings,
            doc_embeddings,
            query_inputs['attention_mask'],
            doc_inputs['attention_mask']
        )

        # 4. 排序并截取 (Sort and Top-K)
        # 将文档和分数打包
        scored_docs = list(zip(documents, scores))
        # 按分数降序排列
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        # 只保留前 5 个 (Top-5)
        # 这里你可以修改切片 [:5] 来决定保留多少文档
        reranked_docs = [doc for doc, _ in scored_docs[:5]]

        return reranked_docs


# ==========================================
# 2. 初始化配置 (Setup)
# ==========================================

# 向量嵌入模型 (用于第一阶段检索)
hf_bge_embeddings = HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-large-zh-v1.5"
)

# LLM (用于第二阶段精炼/压缩)
# 切换为 Google Gemini
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash", 
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0
)

# ==========================================
# 3. 准备数据 (Data Preparation)
# ==========================================
# 加载示例文本
print("正在加载并切分文档...")
loader = TextLoader("../../data/C4/txt/ai.txt", encoding="utf-8")
documents = loader.load()
# 切分成 500 字的小块
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
docs = text_splitter.split_documents(documents)

# ==========================================
# 4. 构建检索管道 (Build Pipeline)
# ==========================================

# 4.1 基础检索器 (Base Retriever)
# 使用 FAISS 向量库，检索 Top 20
vectorstore = FAISS.from_documents(docs, hf_bge_embeddings)
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 20})

# 4.2 重排序器 (Reranker)
# 使用上面定义的 ColBERT
reranker = ColBERTReranker()

# 4.3 压缩器/精炼器 (Compressor/Refiner)
# 使用 LLM 提取关键信息
compressor = LLMChainExtractor.from_llm(llm)

# 4.4 组装管道 (Pipeline)
# 顺序：ColBERT 重排 (20 -> 5) -> LLM 精炼 (提取关键句)
pipeline_compressor = DocumentCompressorPipeline(
    transformers=[reranker, compressor]
)

# 4.5 最终检索器 (Final Retriever)
final_retriever = ContextualCompressionRetriever(
    base_compressor=pipeline_compressor,
    base_retriever=base_retriever
)

# ==========================================
# 5. 执行查询并对比 (Execute & Compare)
# ==========================================
query = "AI还有哪些缺陷需要克服？"
print(f"\n{'='*20} 开始执行查询 {'='*20}")
print(f"查询问题: {query}\n")

# 5.1 基础检索结果 (不重排)
print(f"--- (1) 基础检索结果 (Top 20 的前 3 个) ---")
base_results = base_retriever.invoke(query)
for i, doc in enumerate(base_results[:3]): # 只打印前3个示意
    print(f"  [{i+1}] {doc.page_content[:100].replace(chr(10), ' ')}... (长度: {len(doc.page_content)})")

print("\n" + "-"*50 + "\n")

# 5.2 最终检索结果 (重排 + 精炼)
print(f"--- (2) 最终管道结果 (ColBERT 重排 + LLM 精炼) ---")
# 注意：这一步会比较慢，因为涉及复杂的计算和 LLM 调用
final_results = final_retriever.invoke(query)

if not final_results:
    print("没有找到相关结果 (可能是 LLM 认为没有相关内容并过滤掉了)")
else:
    for i, doc in enumerate(final_results):
        print(f"  [{i+1}] {doc.page_content.replace(chr(10), ' ')} (长度: {len(doc.page_content)})")
        # 你会发现这里的长度比基础检索的短很多，因为只保留了精华

