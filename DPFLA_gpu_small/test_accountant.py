from utils.dp_utils import GaussianMomentsAccountant, MomentsAccountant
accountant = GaussianMomentsAccountant(total_examples=60000, max_moment_order=32)

for i in range(500):
    accountant.accumulate_privacy_spending(sigma=4.0, num_examples=600)
eps_deltas = accountant.get_privacy_spent(target_deltas=[1e-5,0.05])
for eps, delta in eps_deltas:
    print("epsilon:",eps, "delta", delta)



# import torch
# import torch.nn as nn
# import torch.optim as optim
# from opacus import PrivacyEngine

# # 检查 GPU 是否可用
# # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"
# print(f"Using device: {device}")

# # 定义一个简单的一层线性模型
# class MinimalLinearModel(nn.Module):
#     def __init__(self):
#         super(MinimalLinearModel, self).__init__()
#         self.linear = nn.Linear(1, 1)  # 输入一个特征，输出一个值

#     def forward(self, x):
#         return self.linear(x)

# # 参数设置
# BATCH_SIZE = 1024  # 批量大小较大，以加快训练速度
# EPOCHS = 5         # 少量训练轮数
# LEARNING_RATE = 0.01
# DELTA = 1e-5  # 差分隐私参数 delta

# # 生成随机数据并传输到 GPU
# num_samples = 10240000  # 50万样本
# data = torch.randn(num_samples, 1).to(device)  # 每个样本一个特征
# labels = torch.randn(num_samples, 1).to(device)  # 标签为一个数值

# # 数据加载器
# train_loader = torch.utils.data.DataLoader(
#     list(zip(data, labels)), batch_size=BATCH_SIZE, shuffle=True
# )

# # 初始化模型、损失函数和优化器，并将模型移到 GPU
# model = MinimalLinearModel().to(device)
# criterion = nn.MSELoss()  # 使用均方误差损失函数
# optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

# # 初始化 PrivacyEngine 并附加到优化器
# privacy_engine = PrivacyEngine()

# model, optimizer, train_loader = privacy_engine.make_private(
#     module=model,
#     optimizer=optimizer,
#     data_loader=train_loader,
#     noise_multiplier=1.0,  # 较小的噪声倍数
#     max_grad_norm=1.0,  # 梯度裁剪
# )

# # 训练模型
# model.train()
# for epoch in range(EPOCHS):
#     epoch_loss = 0
#     for batch_data, batch_labels in train_loader:
#         batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)

#         optimizer.zero_grad()
#         output = model(batch_data)
#         loss = criterion(output, batch_labels)
#         loss.backward()
#         optimizer.step()
#         epoch_loss += loss.item()
#         epsilon = privacy_engine.get_epsilon(DELTA)
#         print(f"Final ε = {epsilon:.2f} for δ = {DELTA}")

#     print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {epoch_loss / len(train_loader)}")

# # 计算并打印最终的 epsilon
# epsilon = privacy_engine.get_epsilon(DELTA)
# print(f"Final ε = {epsilon:.2f} for δ = {DELTA}")

