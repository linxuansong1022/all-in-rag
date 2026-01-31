import torch
from visual_bge.visual_bge.modeling import Visualized_BGE

model = Visualized_BGE(model_name_bge="BAAI/bge-base-en-v1.5",
                      model_weight="../../models/bge/Visualized_base_en_v1.5.pth")
model.eval()

with torch.no_grad():
    text_emb = model.encode(text="bluewhale")
    img_emb_1 = model.encode(image="../../data/C3/imgs/datawhale01.png")
    multi_emb_1 = model.encode(image="../../data/C3/imgs/datawhale01.png", text="bluewhale")#把图+文 混合变成向量 从而实现跨模态检索 用户搜文字 找到匹配的图片 用户上传图片 找到相似的文字描述
    img_emb_2 = model.encode(image="../../data/C3/imgs/datawhale02.png")
    multi_emb_2 = model.encode(image="../../data/C3/imgs/datawhale02.png", text="bluewhale")

# 计算相似度,也就是向量积，点积
sim_1 = img_emb_1 @ img_emb_2.T # @表示矩阵乘法
sim_2 = img_emb_1 @ multi_emb_1.T
sim_3 = text_emb @ multi_emb_1.T
sim_4 = multi_emb_1 @ multi_emb_2.T

print("=== 相似度计算结果 ===")
print(f"纯图像 vs 纯图像: {sim_1}")
print(f"图文结合1 vs 纯图像: {sim_2}")
print(f"图文结合1 vs 纯文本: {sim_3}")
print(f"图文结合1 vs 图文结合2: {sim_4}")

# 向量信息分析
print("\n=== 嵌入向量信息 ===")
print(f"多模态向量维度: {multi_emb_1.shape}") #打印出向量的形状torch.Size([1, 768])
print(f"图像向量维度: {img_emb_1.shape}")#torch.Size([1, 768])
print(f"多模态向量示例 (前10个元素): {multi_emb_1[0][:10]}") #tensor([ 0.0360, -0.0032, -0.0377,  0.0240,  0.0140,  0.0340,  0.0148,  0.0292, 0.0060, -0.0145])
print(f"图像向量示例 (前10个元素):   {img_emb_1[0][:10]}")
