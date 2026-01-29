import os
import google.generativeai as genai
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("错误：未在 .env 文件中找到 GOOGLE_API_KEY")
else:
    genai.configure(api_key=api_key)
    try:
        print("正在查询可用模型...")
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                print(f"- {m.name}")
    except Exception as e:
        print(f"查询失败: {e}")
