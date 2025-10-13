# from dp_utils import GaussianMomentsAccountant, MomentsAccountant
# accountant = GaussianMomentsAccountant(total_examples=60000, max_moment_order=32)

# for i in range(10000):
#     accountant.accumulate_privacy_spending(sigma=4.0, num_examples=600)
# eps_deltas = accountant.get_privacy_spent(target_deltas=[1e-5,0.05])
# for eps, delta in eps_deltas:
#     print("epsilon:",eps, "delta", delta)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from opacus import PrivacyEngine

# 创建一个简单的数据集
X = torch.randn(1000, 10)  # 1000 个样本，每个样本有 10 个特征
y = (X.sum(dim=1) > 0).long()  # 简单的线性可分标签

dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

# 定义一个简单的模型
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 2)  # 输入 10 个特征，输出 2 个类别

    def forward(self, x):
        return self.fc(x)

model = SimpleModel()

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 添加隐私引擎
privacy_engine = PrivacyEngine()
model, optimizer, dataloader = privacy_engine.make_private(
    module=model,
    optimizer=optimizer,
    data_loader=dataloader,
    noise_multiplier=4.0,
    max_grad_norm=1.0,
)

# 训练模型
epochs = 100
for epoch in range(epochs):
    for batch_idx, (data, target) in enumerate(dataloader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# 打印隐私损失
privacy_spent = privacy_engine.accountant.get_epsilon(delta=1e-5)
print(f"(ε, δ) = ({privacy_spent:.2f}, 1e-5)")
