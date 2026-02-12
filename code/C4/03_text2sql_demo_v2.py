import os
import json
import sqlite3
import numpy as np
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pymilvus import connections, MilvusClient, FieldSchema, CollectionSchema, DataType, Collection


class BGESmallEmbeddingFunction:
    """
    封装 BGE-Small 中文嵌入模型 (BAAI/bge-small-zh-v1.5)。
    该类用于生成文本的向量表示 (Embeddings)。
    
    BGE (BAAI General Embedding) 是目前效果较好的开源中文 Embedding 模型之一。
    """
    
    def __init__(self, model_name="BAAI/bge-small-zh-v1.5", device="cpu"):
        self.model_name = model_name
        self.device = device
        # 使用 SentenceTransformer 加载预训练模型
        self.model = SentenceTransformer(model_name, device=device)
        # 获取模型的输出维度 (通常是 512)
        self.dense_dim = self.model.get_sentence_embedding_dimension()
    
    def encode_text(self, texts):
        """
        核心方法：将输入文本 (单个字符串或字符串列表) 转换为密集向量。
        """
        if isinstance(texts, str):
            texts = [texts]
        
        # normalize_embeddings=True 能够让之后的点积计算等价于余弦相似度
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=16,
            convert_to_numpy=True
        )
        
        return embeddings
    
    @property
    def dim(self):
        """返回向量维度，用于定义 Milvus schema"""
        return self.dense_dim


class SimpleKnowledgeBase:
    """
    Text2SQL 的知识库管理类。
    
    功能：
    1. 管理 Milvus 向量数据库连接
    2. 存储两类知识：
       - SQL 示例 (Few-shot examples): 帮助 LLM 学习如何将特定问题转换为 SQL。
       - 表结构信息 (Schema): 帮助检索相关的表和字段信息。
    3. 提供基于语义相似度的检索功能 (RAG)。
    """
    
    def __init__(self, milvus_uri: str = "http://localhost:19530"):
        self.milvus_uri = milvus_uri
        self.collection_name = "text2sql_knowledge_base"
        self.milvus_client = None
        self.collection = None
        
        # 初始化 Embedding 模型
        self.embedding_function = BGESmallEmbeddingFunction(
            model_name="BAAI/bge-small-zh-v1.5",
            device="cpu"
        )
        
        self.sql_examples = []
        self.table_schemas = []
        self.data_loaded = False
    
    def connect_milvus(self):
        """连接到本地 Milvus 实例"""
        connections.connect(uri=self.milvus_uri)
        self.milvus_client = MilvusClient(uri=self.milvus_uri)
        return True
    
    def create_collection(self):
        """
        定义并创建 Milvus 集合 (Collection)，相当于关系型数据库中的表。
        """
        if not self.milvus_client:
            self.connect_milvus()
        
        # 如果集合已存在，先删除，保持环境纯净 (仅用于演示)
        if self.milvus_client.has_collection(self.collection_name):
            self.milvus_client.drop_collection(self.collection_name)
        
        # 定义字段结构 (Schema)
        fields = [
            FieldSchema(name="pk", dtype=DataType.VARCHAR, is_primary=True, auto_id=True, max_length=100),
            FieldSchema(name="content_type", dtype=DataType.VARCHAR, max_length=50),  # 类型: sql_example 或 table_schema
            FieldSchema(name="question", dtype=DataType.VARCHAR, max_length=1000),      # 问题文本
            FieldSchema(name="sql", dtype=DataType.VARCHAR, max_length=2000),           # 对应的 SQL
            FieldSchema(name="description", dtype=DataType.VARCHAR, max_length=1000),   # 描述
            FieldSchema(name="table_name", dtype=DataType.VARCHAR, max_length=100),     # 表名
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.embedding_function.dim) # 向量字段
        ]
        
        schema = CollectionSchema(fields, description="Text2SQL知识库")
        self.collection = Collection(name=self.collection_name, schema=schema, consistency_level="Strong")
        
        # 为向量字段创建索引，加速检索
        index_params = {"index_type": "AUTOINDEX", "metric_type": "IP", "params": {}}
        self.collection.create_index("embedding", index_params)
        
        return True
    
    def load_data(self):
        """
        主控方法：从 JSON 文件加载数据，向量化，并存入 Milvus。
        """        
        data_dir = os.path.join(os.path.dirname(__file__), "data")
        
        self.load_sql_examples(data_dir)
        self.load_table_schemas(data_dir)
        self.vectorize_and_store()
        
        self.data_loaded = True
    
    def load_sql_examples(self, data_dir: str):
        """从 qsql_examples.json 加载 SQL 示例数据，如果不存在则创建默认数据"""
        sql_examples_path = os.path.join(data_dir, "qsql_examples.json")
        
        default_examples = [
            {"question": "查询所有用户信息", "sql": "SELECT * FROM users", "description": "获取用户记录", "database": "sqlite"},
            {"question": "年龄大于30的用户", "sql": "SELECT * FROM users WHERE age > 30", "description": "年龄筛选", "database": "sqlite"},
            {"question": "统计用户总数", "sql": "SELECT COUNT(*) as user_count FROM users", "description": "用户计数", "database": "sqlite"},
            {"question": "查询库存不足的产品", "sql": "SELECT * FROM products WHERE stock < 50", "description": "库存筛选", "database": "sqlite"},
            {"question": "查询用户订单信息", "sql": "SELECT u.name, p.name, o.quantity FROM orders o JOIN users u ON o.user_id = u.id JOIN products p ON o.product_id = p.id", "description": "订单详情", "database": "sqlite"},
            {"question": "按城市统计用户", "sql": "SELECT city, COUNT(*) as count FROM users GROUP BY city", "description": "城市分组", "database": "sqlite"}
        ]
        
        if os.path.exists(sql_examples_path):
            with open(sql_examples_path, 'r', encoding='utf-8') as f:
                self.sql_examples = json.load(f)
        else:
            self.sql_examples = default_examples
            os.makedirs(data_dir, exist_ok=True)
            with open(sql_examples_path, 'w', encoding='utf-8') as f:
                json.dump(self.sql_examples, f, ensure_ascii=False, indent=2)
    
    def load_table_schemas(self, data_dir: str):
        """从 table_schemas.json 加载数据库表结构描述"""
        schema_path = os.path.join(data_dir, "table_schemas.json")
        
        default_schemas = [
            # ... (这里省略了一些默认 Schema 的详细定义，实际代码中会包含) ...
            {
                "table_name": "users",
                "description": "用户信息表",
                "columns": [
                    {"name": "id", "type": "INTEGER", "description": "用户ID"},
                    {"name": "name", "type": "VARCHAR", "description": "用户姓名"},
                    {"name": "age", "type": "INTEGER", "description": "用户年龄"},
                    {"name": "email", "type": "VARCHAR", "description": "邮箱地址"},
                    {"name": "city", "type": "VARCHAR", "description": "所在城市"},
                    {"name": "created_at", "type": "DATETIME", "description": "创建时间"}
                ]
            },
            {
                "table_name": "products",
                "description": "产品信息表",
                "columns": [
                    {"name": "id", "type": "INTEGER", "description": "产品ID"},
                    {"name": "product_name", "type": "VARCHAR", "description": "产品名称"},
                    {"name": "category", "type": "VARCHAR", "description": "产品类别"},
                    {"name": "price", "type": "DECIMAL", "description": "产品价格"},
                    {"name": "stock", "type": "INTEGER", "description": "库存数量"},
                    {"name": "description", "type": "TEXT", "description": "产品描述"}
                ]
            },
            {
                "table_name": "orders",
                "description": "订单信息表",
                "columns": [
                    {"name": "id", "type": "INTEGER", "description": "订单ID"},
                    {"name": "user_id", "type": "INTEGER", "description": "用户ID"},
                    {"name": "product_id", "type": "INTEGER", "description": "产品ID"},
                    {"name": "quantity", "type": "INTEGER", "description": "购买数量"},
                    {"name": "total_price", "type": "DECIMAL", "description": "总价格"},
                    {"name": "order_date", "type": "DATETIME", "description": "订单日期"}
                ]
            }
        ]
        
        if os.path.exists(schema_path):
            with open(schema_path, 'r', encoding='utf-8') as f:
                self.table_schemas = json.load(f)
        else:
            self.table_schemas = default_schemas
            os.makedirs(data_dir, exist_ok=True)
            with open(schema_path, 'w', encoding='utf-8') as f:
                json.dump(self.table_schemas, f, ensure_ascii=False, indent=2)
    
    def vectorize_and_store(self):
        """
        核心步骤：
        1. 将 SQL 示例和 Schema 描述转换为文本字符串
        2. 调用 Embedding 模型生成向量
        3. 将 Metadata 和 向量 存入 Milvus
        """
        self.create_collection()
        
        all_texts = []
        all_metadata = []
        
        # 1. 处理 SQL 示例
        for example in self.sql_examples:
            # 构造用于 Embedding 的文本，包含问题、SQL 和描述
            text = f"问题: {example['question']} SQL: {example['sql']} 描述: {example.get('description', '')}"
            all_texts.append(text)
            all_metadata.append({
                "content_type": "sql_example",
                "question": example['question'],
                "sql": example['sql'],
                "description": example.get('description', ''),
                "table_name": ""
            })
        
        # 2. 处理表结构
        for schema in self.table_schemas:
            columns_desc = ", ".join([f"{col['name']} ({col['type']}): {col.get('description', '')}" 
                                    for col in schema['columns']])
            text = f"表 {schema['table_name']}: {schema['description']} 字段: {columns_desc}"
            all_texts.append(text)
            all_metadata.append({
                "content_type": "table_schema",
                "question": "",
                "sql": "",
                "description": schema['description'],
                "table_name": schema['table_name']
            })
        
        # 3. 批量生成 Embedding
        embeddings = self.embedding_function.encode_text(all_texts)
        
        # 4. 准备插入数据
        insert_data = []
        for i, (embedding, metadata) in enumerate(zip(embeddings, all_metadata)):
            insert_data.append([
                metadata["content_type"],
                metadata["question"],
                metadata["sql"],
                metadata["description"],
                metadata["table_name"],
                embedding.tolist()
            ])
        
        # 5. 插入 Milvus 并刷新
        self.collection.insert(insert_data)
        self.collection.flush()
        self.collection.load()
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        在知识库中搜索与 query 语义最相似的内容。
        """
        if not self.data_loaded:
            self.load_data()
        
        # 1. 编码查询语句
        query_embedding = self.embedding_function.encode_text([query])[0]
        
        # 2. 在 Milvus 中进行向量搜索
        search_params = {"metric_type": "IP", "params": {}}
        results = self.collection.search(
            [query_embedding.tolist()],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["content_type", "question", "sql", "description", "table_name"]
        )[0]
        
        # 3. 格式化结果
        formatted_results = []
        for hit in results:
            result = {
                "score": float(hit.distance),
                "content_type": hit.entity.get("content_type"),
                "question": hit.entity.get("question"),
                "sql": hit.entity.get("sql"),
                "description": hit.entity.get("description"),
                "table_name": hit.entity.get("table_name")
            }
            formatted_results.append(result)
        
        return formatted_results
    
    def _fallback_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """
        降级搜索方法：
        如果 Milvus 不可用，可以使用此方法进行简单的关键词匹配搜索。
        """
        results = []
        query_lower = query.lower()
        
        for example in self.sql_examples:
            question_lower = example['question'].lower()
            sql_lower = example['sql'].lower()
            
            score = 0
            for word in query_lower.split():
                if word in question_lower:
                    score += 2
                if word in sql_lower:
                    score += 1
            
            if score > 0:
                results.append({
                    "score": score,
                    "content_type": "sql_example",
                    "question": example['question'],
                    "sql": example['sql'],
                    "description": example.get('description', ''),
                    "table_name": ""
                })
        
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_k]
    
    def add_sql_example(self, question: str, sql: str, description: str = ""):
        """动态添加新的 SQL 示例到知识库"""
        new_example = {
            "question": question,
            "sql": sql,
            "description": description,
            "database": "sqlite"
        }
        self.sql_examples.append(new_example)
        
        # 更新 JSON 文件
        data_dir = os.path.join(os.path.dirname(__file__), "data")
        sql_examples_path = os.path.join(data_dir, "qsql_examples.json")
        
        with open(sql_examples_path, 'w', encoding='utf-8') as f:
            json.dump(self.sql_examples, f, ensure_ascii=False, indent=2)
        
        # 实时插入 Milvus
        if self.collection and self.data_loaded:
            text = f"问题: {question} SQL: {sql} 描述: {description}"
            embedding = self.embedding_function.encode_text([text])[0]
            
            insert_data = [[
                "sql_example",
                question,
                sql,
                description,
                "",
                embedding.tolist()
            ]]
            
            self.collection.insert(insert_data)
            self.collection.flush()
    
    def cleanup(self):
        """资源清理：释放 Collection，删除 Milvus 生成的数据"""
        if self.collection:
            self.collection.release()
        
        if self.milvus_client and self.milvus_client.has_collection(self.collection_name):
            self.milvus_client.drop_collection(self.collection_name)


def demo():
    """
    独立演示函数：
    不依赖上面的 SimpleKnowledgeBase 类，
    而是展示：
    1. BGE Model 自定义调用
    2. 基于 SQLite 的原生 SQL 查询操作 (模拟业务数据库)
    """
    # 1. 模型测试：演示如何将自然语言转化为向量
    embedding_function = BGESmallEmbeddingFunction()
    test_texts = ["查询用户", "统计数据"]
    embeddings = embedding_function.encode_text(test_texts)
    print(f"向量维度: {embeddings.shape}")
    
    # 2. 数据库查询演示：创建一个临时的 SQLite 数据库
    db_path = "demo.db"
    
    if os.path.exists(db_path):
        os.remove(db_path)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 创建简单用户表
    cursor.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, age INTEGER, city TEXT)")
    
    # 插入测试数据
    users_data = [(1, '张三', 25, '北京'), (2, '李四', 32, '上海'), (3, '王五', 35, '深圳')]
    cursor.executemany("INSERT INTO users VALUES (?, ?, ?, ?)", users_data)
    
    conn.commit()
    
    # 定义测试用的 SQL 语句 (注意：这里的 SQL 是硬编码的，不是由 AI 生成的)
    test_sqls = [
        ("查询所有用户", "SELECT * FROM users"),
        ("年龄大于30的用户", "SELECT * FROM users WHERE age > 30"),
        ("统计用户总数", "SELECT COUNT(*) FROM users")
    ]
    
    for i, (question, sql) in enumerate(test_sqls, 1):
        print(f"\n问题 {i}: {question}")
        print("-" * 40)
        print(f"SQL: {sql}")
        
        # 执行查询
        cursor.execute(sql)
        rows = cursor.fetchall()
        
        if rows:
            print(f"返回 {len(rows)} 行数据")
            for j, row in enumerate(rows[:2], 1):
                print(f"  {j}. {row}")
            
            if len(rows) > 2:
                print(f"  ... 还有 {len(rows) - 2} 行")
        else:
            print("无数据返回")
    
    conn.close()
    os.remove(db_path)


if __name__ == "__main__":
    demo()