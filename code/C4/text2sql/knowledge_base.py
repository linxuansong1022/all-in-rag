import json
import os
from typing import List, Dict, Any
from pymilvus import MilvusClient, FieldSchema, CollectionSchema, DataType
from pymilvus.model.hybrid import BGEM3EmbeddingFunction


class SimpleKnowledgeBase:
    """
    知识库 (Knowledge Base)
    核心作用：Schema Linking (模式链接) 的基础设施。
    它不存业务数据（如订单记录），而是存元数据（表结构、字段含义、SQL案例）。
    目标：当用户提问时，通过向量检索快速找到相关的表和字段，而不是把几百张表全塞给 LLM。
    """
    
    def __init__(self, milvus_uri: str = "http://localhost:19530"):
        self.milvus_uri = milvus_uri
        self.client = MilvusClient(uri=milvus_uri)
        # 使用 BGE-M3 模型，它对中文语义理解很好，适合做 Schema 匹配
        self.embedding_function = BGEM3EmbeddingFunction(use_fp16=False, device="cpu")
        self.collection_name = "text2sql_kb"
        self._setup_collection()
    
    def _setup_collection(self):
        """设置集合"""
        if self.client.has_collection(self.collection_name):
            self.client.drop_collection(self.collection_name)
        
        # 定义字段
        fields = [
            FieldSchema(name="pk", dtype=DataType.VARCHAR, is_primary=True, auto_id=True, max_length=100),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=4096), # 存储 DDL、字段描述或 SQL 案例的文本
            FieldSchema(name="type", dtype=DataType.VARCHAR, max_length=32),  # 类型标签：ddl, qsql, description
            FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=self.embedding_function.dim["dense"]) # 向量字段
        ]
        
        schema = CollectionSchema(fields, description="Text2SQL知识库")
        
        # 创建集合
        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            consistency_level="Strong"
        )
        
        # 创建索引 (AUTOINDEX 自动选择最佳索引算法)
        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="dense_vector",
            index_type="AUTOINDEX",
            metric_type="IP"
        )
        
        self.client.create_index(
            collection_name=self.collection_name,
            index_params=index_params
        )
    
    def load_data(self):
        """加载所有知识库数据"""
        data_dir = os.path.join(os.path.dirname(__file__), "data")
        
        # 1. 加载 DDL (Create Table 语句)
        # 作用：让 LLM 知道有哪些表，以及字段的数据类型（int, varchar）。
        ddl_path = os.path.join(data_dir, "ddl_examples.json")
        if os.path.exists(ddl_path):
            with open(ddl_path, 'r', encoding='utf-8') as f:
                ddl_data = json.load(f)
            self._add_ddl_data(ddl_data)
        
        # 2. 加载 Few-shot Examples (Q->SQL 对)
        # 作用：提供“解题思路”。遇到复杂查询（如环比增长），检索出类似的 SQL 例子给 LLM 参考。
        qsql_path = os.path.join(data_dir, "qsql_examples.json")
        if os.path.exists(qsql_path):
            with open(qsql_path, 'r', encoding='utf-8') as f:
                qsql_data = json.load(f)
            self._add_qsql_data(qsql_data)
        
        # 3. 加载 Description (业务描述)
        # 作用：解决“同义词”问题。比如用户说“营业额”，数据库里叫 `gmv`。
        # 通过存储 `gmv: 商品交易总额，即营业额`，向量检索能把这两个词联系起来。
        desc_path = os.path.join(data_dir, "db_descriptions.json")
        if os.path.exists(desc_path):
            with open(desc_path, 'r', encoding='utf-8') as f:
                desc_data = json.load(f)
            self._add_description_data(desc_data)
        
        # 加载集合到内存
        self.client.load_collection(collection_name=self.collection_name)
        print("知识库数据加载完成")
    
    def _add_ddl_data(self, data: List[Dict]):
        """添加DDL数据"""
        contents = []
        types = []
        
        for item in data:
            content = f"表名: {item.get('table_name', '')}\n"
            content += f"DDL: {item.get('ddl_statement', '')}\n"
            content += f"描述: {item.get('description', '')}"
            
            contents.append(content)
            types.append("ddl")
        
        self._insert_data(contents, types)
    
    def _add_qsql_data(self, data: List[Dict]):
        """添加Q->SQL数据"""
        contents = []
        types = []
        
        for item in data:
            content = f"问题: {item.get('question', '')}\n"
            content += f"SQL: {item.get('sql', '')}"
            
            contents.append(content)
            types.append("qsql")
        
        self._insert_data(contents, types)
    
    def _add_description_data(self, data: List[Dict]):
        """添加描述数据"""
        contents = []
        types = []
        
        for item in data:
            content = f"表名: {item.get('table_name', '')}\n"
            content += f"表描述: {item.get('table_description', '')}\n"
            
            columns = item.get('columns', [])
            if columns:
                content += "字段信息:\n"
                for col in columns:
                    content += f"  - {col.get('name', '')}: {col.get('description', '')} ({col.get('type', '')})\n"
            
            contents.append(content)
            types.append("description")
        
        self._insert_data(contents, types)
    
    def _insert_data(self, contents: List[str], types: List[str]):
        """插入数据"""
        if not contents:
            return
        
        # 生成嵌入 (Vectorization)
        embeddings = self.embedding_function(contents)
        
        # 构建插入数据，每一行是一个字典
        data_to_insert = []
        for i in range(len(contents)):
            data_to_insert.append({
                "content": contents[i],
                "type": types[i],
                "dense_vector": embeddings["dense"][i]
            })
        
        # 插入数据
        result = self.client.insert(
            collection_name=self.collection_name,
            data=data_to_insert
        )
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        核心检索方法
        根据用户问题，在向量空间中寻找最相关的 Schema 和 Examples。
        这就是 Schema Linking 的具体实现。
        """
        self.client.load_collection(collection_name=self.collection_name)
            
        query_embeddings = self.embedding_function([query])
        
        search_results = self.client.search(
            collection_name=self.collection_name,
            data=query_embeddings["dense"],
            anns_field="dense_vector",
            search_params={"metric_type": "IP"},
            limit=top_k,
            output_fields=["content", "type"]
        )
        
        results = []
        for hit in search_results[0]:
            results.append({
                "content": hit["entity"]["content"],
                "type": hit["entity"]["type"],
                "score": hit["distance"]
            })
        
        return results
    
    def cleanup(self):
        """清理资源"""
        try:
            self.client.drop_collection(self.collection_name)
        except:
            pass 