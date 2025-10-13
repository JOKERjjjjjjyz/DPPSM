from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import torch
import numpy as np

# 加载 BERT 模型
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModel.from_pretrained("distilbert-base-uncased")

def edit_distance(s1: str, s2: str) -> int:
    m, n = len(s1), len(s2)
    # 初始化 DP 表
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # 初始化边界
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    # 动态规划计算距离
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],    # 删除
                    dp[i][j - 1],    # 插入
                    dp[i - 1][j - 1] # 替换
                )
    return dp[m][n]


# 获取句向量（取 [CLS] token）
def get_embedding(text):
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt")
        outputs = model(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze()
        return cls_embedding.numpy()

# 密码对
pw1 = "password123"
# pw2 = "passphrase321"
# pw2 = "p@ssw0rd123"
pw2 = "123password"

# 计算余弦距离
emb1 = get_embedding(pw1)
emb2 = get_embedding(pw2)
cos_sim = cosine_similarity([emb1], [emb2])[0][0]
cos_dist = 1 - cos_sim

print(pw1)
print(pw2)
print(f"Cosine Distance: {cos_dist:.4f}")
print(f"Edit Distance: {edit_distance(pw1,pw2):.4f}")
