import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage

"""
=== Function Calling (工具调用) 演示 ===

【核心概念】：
LLM 本身无法联网，也不知道现在的天气、股价等实时信息。
Function Calling (也叫 Tool Calling) 是一种机制：
1. 我们告诉 LLM：“如果你需要查天气，可以使用 `get_weather` 这个工具。”
2. LLM 收到用户问题（“杭州天气如何？”），发现自己不知道，于是决定调用 `get_weather`。
3. LLM **不会直接执行代码**，而是输出一个“请求”：“请帮我调用 get_weather(location='杭州')”。
4. 我们（开发者）捕获这个请求，执行真正的 Python 函数，拿到结果（“24度”）。
5. 我们把结果（“24度”）再喂回给 LLM。
6. LLM 结合结果生成最终回答：“杭州今天24度...”。

这个过程也叫 **ReAct (Reasoning + Acting)** 循环的基础。
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
# 1. 定义工具 (Define Tools)
# ==========================================
# 使用 @tool 装饰器，可以轻松把一个 Python 函数转换成 LLM 能理解的工具。
# 注意：函数的文档注释 (docstring) 非常重要！LLM 会根据它来理解工具的用途和参数。
@tool
def get_weather(location: str):
    """
    查询指定地点的天气信息。
    
    Args:
        location: 城市名称，例如 "杭州", "北京"
    """
    # 这里我们模拟一个 API 调用
    # 在实际项目中，你会在这里调用真实的天气 API
    if "杭州" in location:
        return "24℃，晴朗，微风"
    elif "北京" in location:
        return "18℃，多云"
    else:
        return "未知天气"

tools = [get_weather]

# ==========================================
# 2. 绑定工具 (Bind Tools)
# ==========================================
# 告诉 LLM：“你可以使用这些工具”。
# .bind_tools() 是 LangChain 的标准方法，它会自动把 Python 函数转换成 OpenAI/Gemini 格式的 tool schema。
llm_with_tools = llm.bind_tools(tools)

# ==========================================
# 3. 第一轮交互：LLM 决定调用工具
# ==========================================
query = "杭州和北京今天天气怎么样？"
messages = [HumanMessage(content=query)]
print(f"用户问题: {query}\n")

print("--- 第1步：LLM 思考 ... ---")
ai_msg = llm_with_tools.invoke(messages)

# 检查 LLM 是否想调用工具
if ai_msg.tool_calls:
    print(f"LLM 决定调用 {len(ai_msg.tool_calls)} 个工具：")
    for tool_call in ai_msg.tool_calls:
        print(f"  - 工具名: {tool_call['name']}")
        print(f"  - 参数: {tool_call['args']}")
    
    # 把 LLM 的回复（包含工具调用请求）加入对话历史
    messages.append(ai_msg)

    # ==========================================
    # 4. 执行工具 (Execute Tools)
    # ==========================================
    print("\n--- 第2步：执行工具 ... ---")
    for tool_call in ai_msg.tool_calls:
        # 根据工具名找到对应的函数并执行
        if tool_call["name"] == "get_weather":
            tool_result = get_weather.invoke(tool_call)
            print(f"  - 工具返回结果: {tool_result}")
            
            # 创建一个 ToolMessage，表示这是工具执行的结果
            messages.append(ToolMessage(
                content=str(tool_result),
                tool_call_id=tool_call["id"]
            ))

    # ==========================================
    # 5. 第二轮交互：LLM 生成最终回答
    # ==========================================
    print("\n--- 第3步：LLM 生成最终回答 ... ---")
    final_response = llm_with_tools.invoke(messages)
    print(f"最终回答: {final_response.content}")

else:
    print("LLM 没有调用工具，直接回答了问题：")
    print(ai_msg.content)

