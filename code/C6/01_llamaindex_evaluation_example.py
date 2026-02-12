import os
import asyncio
from dotenv import load_dotenv, find_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.node_parser import SentenceWindowNodeParser, SentenceSplitter
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from llama_index.core.evaluation import (
    FaithfulnessEvaluator,
    RelevancyEvaluator,
    BatchEvalRunner,
)
from llama_index.core.evaluation import DatasetGenerator, QueryResponseDataset

"""
=== RAG 效果评估演示 (RAG Evaluation) ===
... (comments omitted for brevity) ...
"""

# 加载环境变量
# 首先尝试加载当前目录的 .env，如果找不到则自动向上搜索
load_dotenv() 

# 如果上面的没加载到 Key，我们再强制尝试加载一次根目录的（双重保险）
if not os.getenv("GOOGLE_API_KEY"):
    load_dotenv("../../.env")

# 检查 API Key 是否存在
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("❌ 错误: 未找到 GOOGLE_API_KEY 环境变量。")
    print("请确保 .env 文件存在于当前目录 (code/C6) 或项目根目录。")
    print("且文件中包含: GOOGLE_API_KEY=你的Key")
    exit(1)
else:
    print(f"✅ 已成功加载 GOOGLE_API_KEY (前5位: {api_key[:5]}...)")

# ==========================================
# 0. 全局配置 (Global Settings)
# ==========================================
# 配置 LLM 为 Gemini
Settings.llm = GoogleGenAI(
    model="models/gemini-2.0-flash", 
    api_key=api_key,
    temperature=0
)
# 配置 Embedding 模型
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")


async def main():
    # ==========================================
    # 1. 准备数据 (Prepare Data)
    # ==========================================
    print("正在加载文档...")
    try:
        reader = SimpleDirectoryReader(input_files=["../../data/C3/pdf/IPCC_AR6_WGII_Chapter03.pdf"])
        documents = reader.load_data()
    except Exception as e:
        print(f"文档加载失败，请检查路径: {e}")
        return

    # ==========================================
    # 2. 生成考题 (Generate Evaluation Dataset)
    # ==========================================
    # 如果本地已经有生成的考题集，就直接加载，否则现场生成
    dataset_path = "./c6_response_eval_dataset.json"
    
    if os.path.exists(dataset_path):
        print("加载现有的评估数据集...")
        response_eval_dataset = QueryResponseDataset.from_json(dataset_path)
    else:
        print("正在生成评估数据集 (这可能需要一分钟)...")
        # 为了演示快一点，我们只取前 4 页文档
        dataset_generator = DatasetGenerator.from_documents(
            documents[:4], 
            llm=Settings.llm
        )
        # 生成 10 个问题
        response_eval_dataset = await dataset_generator.agenerate_dataset_from_nodes(num=10)
        response_eval_dataset.save_json(dataset_path)
        print("数据集生成完毕并保存。")

    queries = response_eval_dataset.queries
    print(f"准备了 {len(queries)} 个测试问题。")

    # ==========================================
    # 3. 准备两个考生 (Build 2 Query Engines)
    # ==========================================
    
    # --- 考生 A：句子窗口检索 (Sentence Window Retrieval) ---
    # 这种策略会切分得很细，但在检索时会把周围的句子带出来，通常效果更好
    print("初始化考生 A (句子窗口检索)...")
    sentence_parser = SentenceWindowNodeParser.from_defaults(
        window_size=5,
        window_metadata_key="window",
        original_text_metadata_key="original_text",
    )
    sentence_nodes = sentence_parser.get_nodes_from_documents(documents)
    sentence_index = VectorStoreIndex(sentence_nodes)

    sentence_query_engine = sentence_index.as_query_engine(
        similarity_top_k=2,
        node_postprocessors=[
            MetadataReplacementPostProcessor(target_metadata_key="window")
        ],
    )

    # --- 考生 B：基础检索 (Base Retrieval) ---
    # 这种是普通的切分策略，每块 512 个字
    print("初始化考生 B (常规分块检索)...")
    base_parser = SentenceSplitter(chunk_size=512)
    base_nodes = base_parser.get_nodes_from_documents(documents)
    base_index = VectorStoreIndex(base_nodes)
    
    base_query_engine = base_index.as_query_engine(similarity_top_k=2)

    # ==========================================
    # 4. 裁判入场 (Initialize Evaluators)
    # ==========================================
    faithfulness_evaluator = FaithfulnessEvaluator(llm=Settings.llm)
    relevancy_evaluator = RelevancyEvaluator(llm=Settings.llm)
    
    evaluators = {
        "faithfulness": faithfulness_evaluator, 
        "relevancy": relevancy_evaluator
    }

    # ==========================================
    # 5. 考试开始 (Run Evaluation)
    # ==========================================
    print("\n=== 考生 A (句子窗口) 开始答题并接受评估 ===")
    runner = BatchEvalRunner(evaluators, workers=2, show_progress=True)
    sentence_results = await runner.aevaluate_queries(
        queries=queries, query_engine=sentence_query_engine
    )

    print("\n=== 考生 B (常规分块) 开始答题并接受评估 ===")
    base_results = await runner.aevaluate_queries(
        queries=queries, query_engine=base_query_engine
    )

    # ==========================================
    # 6. 公布成绩 (Print Results)
    # ==========================================
    def print_eval_details(runner_name, eval_results, queries_dict):
        """打印扣分项的详细理由"""
        print(f"\n--- {runner_name} 扣分项详细分析 ---")
        # eval_results 是一个字典: {"faithfulness": [results], "relevancy": [results]}
        # 对应的是 queries 列表中的顺序
        # 因为我们无法直接从 EvaluatorResponse 获取原始问题，所以需要遍历
        for metric, results in eval_results.items():
            for i, res in enumerate(results):
                if res.score < 1.0:
                    # 尝试从 queries 列表中获取对应的问题
                    query_list = list(queries_dict.values())
                    query_text = query_list[i] if i < len(query_list) else "未知问题"
                    print(f"\n❌ [{metric.upper()}] 扣分题目: {query_text}")
                    print(f"💡 扣分理由: {res.feedback}")
                    # print(f"📝 参考证据: {res.contexts}") # 如果需要查看检索到的上下文可以开启

    def get_score(results, metric_name):
        """计算平均分"""
        scores = results[metric_name]
        total_score = sum(result.score for result in scores)
        return total_score / len(scores)

    print("\n" + "="*40)
    print("🏆 最终成绩单")
    print("="*40)

    # 考生 A 成绩
    s_faith = get_score(sentence_results, "faithfulness")
    s_rel = get_score(sentence_results, "relevancy")
    print(f"考生 A (句子窗口):")
    print(f"  - 忠实度 (不瞎编): {s_faith:.1%}")
    print(f"  - 相关性 (不跑题): {s_rel:.1%}")
    if s_faith < 1.0 or s_rel < 1.0:
        print_eval_details("考生 A", sentence_results, queries)

    # 考生 B 成绩
    b_faith = get_score(base_results, "faithfulness")
    b_rel = get_score(base_results, "relevancy")
    print(f"\n考生 B (常规分块):")
    print(f"  - 忠实度 (不瞎编): {b_faith:.1%}")
    print(f"  - 相关性 (不跑题): {b_rel:.1%}")
    if b_faith < 1.0 or b_rel < 1.0:
        print_eval_details("考生 B", base_results, queries)

    # 总结
    print("\n[总结]:")
    if s_faith >= b_faith and s_rel >= b_rel:
        print("✅ 句子窗口检索 (考生A) 完胜！")
    elif s_faith < b_faith and s_rel < b_rel:
        print("❌ 常规检索 (考生B) 竟然赢了？可能是文档太简单了。")
    else:
        print("⚖️ 两者互有胜负。")


if __name__ == "__main__":
    asyncio.run(main())

