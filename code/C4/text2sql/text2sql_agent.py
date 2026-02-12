import sqlite3
import os
from typing import Dict, Any, List, Tuple
from .knowledge_base import SimpleKnowledgeBase
from .sql_generator import SimpleSQLGenerator


class SimpleText2SQLAgent:
    """
    Text2SQL 代理 (Agent Orchestrator)
    核心作用：这是一个“总指挥”，负责协调知识库 (Knowledge Base) 和 SQL 生成器 (SQL Generator)。
    它把所有子模块连接起来，形成一个完整的 Text-to-SQL 工作流，并处理执行和错误修复。
    """
    
    def __init__(self, milvus_uri: str = "http://localhost:19530", api_key: str = None):
        """初始化代理"""
        # 依赖注入：Agent 依赖 KnowledgeBase 和 SQLGenerator
        self.knowledge_base = SimpleKnowledgeBase(milvus_uri)
        self.sql_generator = SimpleSQLGenerator(api_key)
        self.db_path = None
        self.connection = None
        
        # 配置参数 (这些参数在实际生产中可以优化)
        self.max_retry_count = 3       # 生成的 SQL 最多尝试修复多少次
        self.top_k_retrieval = 5       # 从知识库检索多少条相关信息 (Schema/Examples)
        self.max_result_rows = 100     # SQL 查询结果最多返回多少行，防止数据量过大
    
    def connect_database(self, db_path: str) -> bool:
        """连接SQLite数据库"""
        try:
            self.db_path = db_path
            self.connection = sqlite3.connect(db_path)
            print(f"成功连接到数据库: {db_path}")
            return True
        except Exception as e:
            print(f"数据库连接失败: {str(e)}")
            return False
    
    def load_knowledge_base(self):
        """加载知识库"""
        # 在系统启动时一次性加载所有 Schema 和 Examples 到 Milvus
        self.knowledge_base.load_data()
    
    def query(self, user_question: str) -> Dict[str, Any]:
        """
        核心工作流：执行 Text2SQL 查询
        这是一个包含【检索-生成-执行-修复】的完整 RAG Agent 循环。
        """
        if not self.connection:
            return {
                "success": False,
                "error": "数据库未连接",
                "sql": None,
                "results": None
            }
        
        print(f"\n=== 处理查询: {user_question} ===")
        
        # 1. 检索 (Retrieval / Schema Linking)
        # 根据用户问题，从知识库中检索出相关的表结构、字段描述和 Few-shot SQL 例子。
        # 这一步大大缩小了 LLM 需要关注的上下文范围。
        print("检索知识库...")
        knowledge_results = self.knowledge_base.search(user_question, self.top_k_retrieval)
        print(f"检索到 {len(knowledge_results)} 条相关信息")
        
        # 2. 生成 SQL (Generation)
        # 将用户问题和检索到的知识一起喂给 SQL 生成器 (LLM)，让它生成 SQL 语句。
        print("生成SQL...")
        sql = self.sql_generator.generate_sql(user_question, knowledge_results)
        print(f"生成的SQL: {sql}")
        
        # 3. 执行 SQL 并自修复 (Execution & Self-Correction)
        # 这是 Agent 的精髓所在：尝试执行，如果出错就让 LLM 修正。
        retry_count = 0
        while retry_count < self.max_retry_count:
            print(f"执行SQL (尝试 {retry_count + 1}/{self.max_retry_count})...")
            
            success, result = self._execute_sql(sql)
            
            if success:
                print("SQL执行成功!")
                return {
                    "success": True,
                    "error": None,
                    "sql": sql,
                    "results": result,
                    "retry_count": retry_count
                }
            else:
                # SQL 执行失败，尝试修复
                print(f"SQL执行失败: {result}")
                
                if retry_count < self.max_retry_count - 1: # 如果还有重试机会
                    print("尝试修复SQL...")
                    sql = self.sql_generator.fix_sql(sql, result, knowledge_results) # 再次调用 LLM 修复 SQL
                    print(f"修复后的SQL: {sql}")
                
                retry_count += 1
        
        # 达到最大重试次数，放弃
        return {
            "success": False,
            "error": f"超过最大重试次数 ({self.max_retry_count})",
            "sql": sql,
            "results": None,
            "retry_count": retry_count
        }
    
    def _execute_sql(self, sql: str) -> Tuple[bool, Any]:
        """执行SQL语句"""
        try:
            cursor = self.connection.cursor()
            
            # 安全措施：对 SELECT 语句强制添加 LIMIT，防止返回海量数据导致 OOM 或性能问题。
            if sql.strip().upper().startswith('SELECT') and 'LIMIT' not in sql.upper():
                sql = f"{sql.rstrip(';')} LIMIT {self.max_result_rows}"
            
            cursor.execute(sql)
            
            if sql.strip().upper().startswith('SELECT'):
                # 查询语句
                columns = [desc[0] for desc in cursor.description]
                rows = cursor.fetchall()
                
                results = []
                for row in rows:
                    result_row = {}
                    for i, value in enumerate(row):
                        result_row[columns[i]] = value
                    results.append(result_row)
                
                cursor.close()
                return True, {
                    "columns": columns,
                    "rows": results,
                    "count": len(results)
                }
            else:
                # 非查询语句 (如 INSERT, UPDATE, DELETE)
                # 实际生产中通常只给 SELECT 权限，这里只是演示
                self.connection.commit()
                cursor.close()
                return True, "SQL执行成功"
        
        except Exception as e:
            # 捕获 SQL 执行错误，返回给上层 (Agent) 进行自修复
            return False, str(e)
    
    def add_example(self, question: str, sql: str):
        """
        添加新的 Q->SQL 示例
        这是一个简化的方法，实际生产中可能会有更复杂的管理界面或 ETL 流程。
        """
        # 简化版本：直接保存到文件
        data_dir = os.path.join(os.path.dirname(__file__), "data")
        qsql_path = os.path.join(data_dir, "qsql_examples.json")
        
        try:
            import json
            
            # 读取现有数据
            if os.path.exists(qsql_path):
                with open(qsql_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            else:
                data = []
            
            # 添加新示例
            data.append({
                "question": question,
                "sql": sql,
                "database": "sqlite"
            })
            
            # 保存
            with open(qsql_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            print(f"已添加新示例: {question}")
            
        except Exception as e:
            print(f"添加示例失败: {str(e)}")
    
    def get_table_info(self) -> List[Dict[str, Any]]:
        """
        获取数据库表信息
        这部分代码主要用于调试或初始化知识库时，获取当前的数据库 Schema。
        """
        if not self.connection:
            return []
        
        try:
            cursor = self.connection.cursor()
            
            # 获取所有表名
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            
            table_info = []
            for table in tables:
                table_name = table[0]
                
                # 获取表结构
                cursor.execute(f"PRAGMA table_info({table_name})")
                columns = cursor.fetchall()
                
                table_info.append({
                    "table_name": table_name,
                    "columns": [
                        {
                            "name": col[1],
                            "type": col[2],
                            "nullable": not col[3],
                            "default": col[4],
                            "primary_key": bool(col[5])
                        }
                        for col in columns
                    ]
                })
            
            cursor.close()
            return table_info
            
        except Exception as e:
            print(f"获取表信息失败: {str(e)}")
            return []
    
    def cleanup(self):
        """清理资源"""
        if self.connection:
            self.connection.close()
            self.connection = None
            print("数据库连接已关闭")
        
        # 清理 Milvus 知识库
        self.knowledge_base.cleanup()
        print("知识库已清理") 