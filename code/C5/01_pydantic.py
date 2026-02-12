import os
from typing import List
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

"""
=== Pydantic 结构化输出演示 ===

【核心概念】：
通常 LLM 输出的是一段文本字符串，比如 "张三今年30岁..."。
但在实际开发中，我们往往需要 LLM 输出 **JSON 格式** 的数据，方便程序处理。
Pydantic 是 Python 中最流行的数据验证库，LangChain 利用它来告诉 LLM：“请按照这个格式输出数据”。

【流程解析】：
1. 定义数据结构 (Pydantic Model)：告诉 LLM 我们需要什么字段（姓名、年龄、技能）。
2. 创建解析器 (Parser)：把 Pydantic 模型转换成“格式指令”（提示词的一部分）。
3. 注入提示词：把“格式指令”塞进 Prompt 里，告诉 LLM。
4. 解析结果：LLM 输出 JSON 字符串后，解析器自动把它转换成 Python 对象。
"""

# 加载环境变量
load_dotenv()

# 初始化 LLM (使用 Google Gemini)
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0
)

# ==========================================
# 1. 定义数据结构 (Define Data Structure)
# ==========================================
# 这是一个 Pydantic 模型，它就像是一张“表格”或者“模具”。
# 我们告诉 LLM：你要填这张表，不能乱填。
class PersonInfo(BaseModel):
    name: str = Field(description="人物的姓名")
    age: int = Field(description="人物的年龄，必须是整数")
    skills: List[str] = Field(description="人物擅长的技能列表")

# ==========================================
# 2. 创建解析器 (Create Parser)
# ==========================================
# 这个解析器有两个作用：
# 1. 生成提示词：告诉 LLM "请输出符合 PersonInfo 结构的 JSON"。
# 2. 解析结果：把 LLM 的输出文本转换成 PersonInfo 对象。
parser = PydanticOutputParser(pydantic_object=PersonInfo)

# ==========================================
# 3. 创建提示模板 (Create Prompt)
# ==========================================
# partial_variables={"format_instructions": ...} 
# 这一步非常关键！它把 format_instructions (格式指令) 自动填充到了模板里。
# 这样 LLM 就能看到类似 "The output should be formatted as a JSON instance..." 的要求。
prompt = PromptTemplate(
    template="请根据以下文本提取信息。\n{format_instructions}\n\n待处理文本：\n{text}\n",
    input_variables=["text"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

# ==========================================
# 4. 创建处理链 (Create Chain)
# ==========================================
# 流程：Prompt (含格式指令) -> LLM (生成 JSON 字符串) -> Parser (转成 Python 对象)
chain = prompt | llm | parser

# ==========================================
# 5. 执行调用 (Invoke)
# ==========================================
text = "张三今年30岁，他非常擅长 Python 编程，最近还在学习 Go 语言和 Docker。"
print(f"输入文本: {text}\n")
print("正在调用 LLM 提取信息...")

try:
    result = chain.invoke({"text": text})

    # ==========================================
    # 6. 打印结果 (Print Result)
    # ==========================================
    print("\n--- 解析成功！---")
    print(f"结果类型: {type(result)}") # <class '__main__.PersonInfo'>
    print(f"原始对象: {result}\n")
    
    # 因为结果已经是 PersonInfo 对象，我们可以直接用 .属性名 访问
    print(f"姓名: {result.name}")
    print(f"年龄: {result.age}")
    print(f"技能: {', '.join(result.skills)}")

except Exception as e:
    print(f"解析失败: {e}")

