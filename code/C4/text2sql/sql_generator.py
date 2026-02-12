import os
from typing import List, Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage


class SimpleSQLGenerator:
    """
    SQL 生成器 (Generation Layer)
    核心作用：扮演一个“精通 SQL 的数据库专家”。
    它接收自然语言问题和知识库检索到的上下文，输出 SQL 语句。
    """
    
    def __init__(self, api_key: str = None):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0,
            google_api_key=api_key or os.getenv("GOOGLE_API_KEY")
        )
    
    def generate_sql(self, user_query: str, knowledge_results: List[Dict[str, Any]]) -> str:
        """
        生成 SQL 语句
        Prompt Engineering 核心：
        1. Context Injection: 将检索到的表结构 (DDL) 和 例子 (Few-shot) 拼接到 Prompt 中。
        2. Role Playing: 设定 "SQL 专家" 角色。
        3. Constraints: 明确要求只返回 SQL，不解释。
        """
        # 构建上下文
        context = self._build_context(knowledge_results)
        
        # 构建提示
        prompt = f"""你是一个SQL专家。请根据以下信息将用户问题转换为SQL查询语句。

数据库信息：
{context}

用户问题：{user_query}

要求：
1. 只返回SQL语句，不要包含任何解释
2. 确保SQL语法正确
3. 使用上下文中提供的表名和字段名
4. 如果需要JOIN，请根据表结构进行合理关联

SQL语句："""

        messages = [HumanMessage(content=prompt)]
        response = self.llm.invoke(messages)
        
        # 清理SQL语句 (去除 markdown 标记)
        sql = response.content.strip()
        if sql.startswith("```sql"):
            sql = sql[6:]
        if sql.startswith("```"):
            sql = sql[3:]
        if sql.endswith("```"):
            sql = sql[:-3]
        
        return sql.strip()
    
    def fix_sql(self, original_sql: str, error_message: str, knowledge_results: List[Dict[str, Any]]) -> str:
        """
        修复 SQL 语句 (Self-Correction)
        这是 Agent 的高级特性。
        当生成的 SQL 执行报错时，不直接抛出异常，而是把【错误信息】和【原始SQL】喂回给 LLM。
        LLM 具有很强的逻辑纠错能力，看到 "Column not found" 就会去检查 Schema 并修正字段名。
        """
        context = self._build_context(knowledge_results)
        
        prompt = f"""请修复以下SQL语句的错误。

数据库信息：
{context}

原始SQL：
{original_sql}

错误信息：
{error_message}

请返回修复后的SQL语句（只返回SQL，不要解释）："""

        messages = [HumanMessage(content=prompt)]
        response = self.llm.invoke(messages)
        
        # 清理SQL语句
        fixed_sql = response.content.strip()
        if fixed_sql.startswith("```sql"):
            fixed_sql = fixed_sql[6:]
        if fixed_sql.startswith("```"):
            fixed_sql = fixed_sql[3:]
        if fixed_sql.endswith("```"):
            fixed_sql = fixed_sql[:-3]
        
        return fixed_sql.strip()
    
    def _build_context(self, knowledge_results: List[Dict[str, Any]]) -> str:
        """构建上下文信息"""
        context = ""
        
        # 按类型分组
        ddl_info = []
        qsql_examples = []
        descriptions = []
        
        for result in knowledge_results:
            if result["type"] == "ddl":
                ddl_info.append(result["content"])
            elif result["type"] == "qsql":
                qsql_examples.append(result["content"])
            elif result["type"] == "description":
                descriptions.append(result["content"])
        
        # 构建上下文
        if ddl_info:
            context += "=== 表结构信息 ===\n"
            context += "\n".join(ddl_info) + "\n\n"
        
        if descriptions:
            context += "=== 表和字段描述 ===\n"
            context += "\n".join(descriptions) + "\n\n"
        
        if qsql_examples:
            context += "=== 查询示例 ===\n"
            context += "\n".join(qsql_examples) + "\n\n"
        
        return context 