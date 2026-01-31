from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

loader = TextLoader("../../data/C2/txt/蜂医.txt", encoding="utf-8")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    # 针对中英文混合文本，定义一个更全面的分隔符列表
    separators=["\n\n", "\n", "。", "，", " ", ""], # 按顺序尝试分割
    chunk_size=1000, 
    chunk_overlap=50 # 增大重叠以便观察
)

chunks = text_splitter.split_documents(docs)

print(f"文本被切分为 {len(chunks)} 个块。\n")

if len(chunks) >= 2:
    print("=== 验证重叠 (Overlap) ===")
    print(f"块 1 的最后 60 个字符: ...{chunks[0].page_content[-60:]}")
    print("-" * 40)
    print(f"块 2 的前 60 个字符: {chunks[1].page_content[:60]}...")
    print("=" * 60)

print("--- 前5个块内容示例 ---")
for i, chunk in enumerate(chunks[:5]):
    print("=" * 60)
    print(f'块 {i+1} (长度: {len(chunk.page_content)}): "{chunk.page_content}"')
