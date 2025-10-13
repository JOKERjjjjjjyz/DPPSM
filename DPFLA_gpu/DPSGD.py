# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader
# from opacus import PrivacyEngine
# from opacus.layers import DPLSTM  # 导入 DPLSTM
# import matplotlib.pyplot as plt
# import numpy as np
# import os
# import warnings
# import logging

# # 忽略 FutureWarning
# warnings.filterwarnings("ignore", category=FutureWarning)

# # 配置日志记录
# logging.basicConfig(
#     filename='training.log',
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s'
# )

# # 打印 Opacus 和 PyTorch 版本
# import opacus
# print(f"Opacus version: {opacus.__version__}")
# import torch
# print(f"PyTorch version: {torch.__version__}")

# # 1. 数据准备：合成序列数据集
# class SequenceDataset(Dataset):
#     def __init__(self, num_samples=10000, seq_length=10, input_size=5):
#         super(SequenceDataset, self).__init__()
#         self.num_samples = num_samples
#         self.seq_length = seq_length
#         self.input_size = input_size
#         # 生成随机序列
#         self.X = torch.randn(num_samples, seq_length, input_size)
#         # 目标是序列的和
#         self.y = self.X.sum(dim=1)  # 形状: (num_samples, input_size)

#     def __len__(self):
#         return self.num_samples

#     def __getitem__(self, idx):
#         return self.X[idx], self.y[idx]

# # 2. 模型定义：使用 DPLSTM 的经典LSTM模型
# class LSTMNet(nn.Module):
#     def __init__(self, input_size=5, hidden_size=50, output_size=5, num_layers=2):
#         super(LSTMNet, self).__init__()
#         self.lstm = DPLSTM(input_size, hidden_size, num_layers, batch_first=True)  # 使用 DPLSTM
#         self.fc = nn.Linear(hidden_size, output_size)
    
#     def forward(self, x):
#         # x: (batch, seq_length, input_size)
#         out, _ = self.lstm(x)  # out: (batch, seq_length, hidden_size)
#         out = out[:, -1, :]    # 取最后一个时间步
#         out = self.fc(out)     # out: (batch, output_size)
#         return out

# # 3. 训练函数
# def train(model, dataloader, optimizer, criterion, device):
#     model.train()
#     total_loss = 0.0
#     for X, y in dataloader:
#         X, y = X.to(device), y.to(device)
#         optimizer.zero_grad()
#         outputs = model(X)
#         loss = criterion(outputs, y)
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item() * X.size(0)
#     avg_loss = total_loss / len(dataloader.dataset)
#     return avg_loss

# def train_dp(model, dataloader, optimizer, criterion, device):
#     model.train()
#     total_loss = 0.0
#     for X, y in dataloader:
#         X, y = X.to(device), y.to(device)
#         optimizer.zero_grad()
#         outputs = model(X)
#         loss = criterion(outputs, y)
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item() * X.size(0)
#     avg_loss = total_loss / len(dataloader.dataset)
#     return avg_loss

# # 4. 主对比函数
# def main():
#     # 超参数
#     input_size = 5
#     hidden_size = 50
#     output_size = 5
#     num_layers = 2
#     num_epochs = 20
#     batch_size = 1024
#     learning_rate = 0.05

#     # 隐私参数
#     noise_multiplier = 1.1
#     max_grad_norm = 1.0
#     target_delta = 1e-5  # 设置目标 delta

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # 准备数据集和数据加载器
#     dataset = SequenceDataset()
#     dataloader = DataLoader(
#         dataset,
#         batch_size=batch_size,
#         shuffle=True,
#         num_workers=4,        # 根据您的硬件调整
#         pin_memory=True if torch.cuda.is_available() else False
#     )

#     # 初始化模型和损失函数
#     model_sgd = LSTMNet(input_size, hidden_size, output_size, num_layers).to(device)
#     model_dp = LSTMNet(input_size, hidden_size, output_size, num_layers).to(device)
#     criterion = nn.MSELoss()

#     # 标准SGD优化器
#     optimizer_sgd = optim.SGD(model_sgd.parameters(), lr=learning_rate)

#     # DP-SGD优化器
#     optimizer_dp = optim.SGD(model_dp.parameters(), lr=learning_rate)

#     # 初始化隐私引擎，指定使用 GaussianAccountant
#     privacy_engine = PrivacyEngine(
#         accountant='gdp',       # 指定使用 GaussianAccountant
#         secure_mode=False       # 实验阶段可以关闭安全模式，加快训练速度；生产环境建议开启
#     )
#     model_dp, optimizer_dp, dataloader_dp = privacy_engine.make_private(
#         module=model_dp,
#         optimizer=optimizer_dp,
#         data_loader=dataloader,
#         noise_multiplier=noise_multiplier,
#         max_grad_norm=max_grad_norm,
#     )

#     # 打印当前使用的 accountant 类型
#     print(f"使用的 accountant 类型: {type(privacy_engine.accountant).__name__}")
#     logging.info(f"使用的 accountant 类型: {type(privacy_engine.accountant).__name__}")

#     # 用于存储损失和隐私预算的列表
#     loss_sgd = []
#     loss_dp = []
#     epsilon_list = []  # 存储每个 epoch 的 epsilon

#     # 训练循环
#     for epoch in range(1, num_epochs + 1):
#         avg_loss_sgd = train(model_sgd, dataloader, optimizer_sgd, criterion, device)
#         avg_loss_dp = train_dp(model_dp, dataloader_dp, optimizer_dp, criterion, device)

#         loss_sgd.append(avg_loss_sgd)
#         loss_dp.append(avg_loss_dp)

#         # 计算当前的 epsilon
#         epsilon = privacy_engine.accountant.get_epsilon(delta=target_delta)
#         epsilon_list.append(epsilon)

#         print(f"Epoch [{epoch}/{num_epochs}] - SGD Loss: {avg_loss_sgd:.4f}, DP-SGD Loss: {avg_loss_dp:.4f}, ε: {epsilon:.2f} (δ={target_delta})")
#         logging.info(f"Epoch [{epoch}/{num_epochs}] - SGD Loss: {avg_loss_sgd:.4f}, DP-SGD Loss: {avg_loss_dp:.4f}, ε: {epsilon:.2f} (δ={target_delta})")

#     # 5. 可视化并保存图像
#     epochs_np = np.arange(1, num_epochs + 1)
#     fig, ax1 = plt.subplots(figsize=(10, 6))

#     color = 'tab:blue'
#     ax1.set_xlabel('Epoch')
#     ax1.set_ylabel('平均损失', color=color)
#     ax1.plot(epochs_np, loss_sgd, label='标准 SGD', marker='o', color='tab:blue')
#     ax1.plot(epochs_np, loss_dp, label='DP-SGD', marker='x', color='tab:orange')
#     ax1.tick_params(axis='y', labelcolor=color)
#     ax1.legend(loc='upper left')

#     ax2 = ax1.twinx()  # 实例化第二个坐标轴，共享 x 轴
#     color = 'tab:red'
#     ax2.set_ylabel('ε', color=color)  # 设置 y 轴标签
#     ax2.plot(epochs_np, epsilon_list, label='隐私预算 ε', marker='s', color=color)
#     ax2.tick_params(axis='y', labelcolor=color)
#     ax2.legend(loc='upper right')

#     plt.title('损失收敛对比及隐私预算跟踪：SGD vs DP-SGD')
#     fig.tight_layout()  # 防止标签被遮挡

#     # 确保保存图像的目录存在
#     save_dir = 'plots'
#     os.makedirs(save_dir, exist_ok=True)
#     save_path = os.path.join(save_dir, 'loss_convergence_with_privacy.png')
#     plt.savefig(save_path, dpi=300)  # 增加 dpi 以提高图像清晰度
#     plt.close()  # 关闭图形以释放内存

#     print(f"损失收敛及隐私预算图已保存至 {save_path}")
#     logging.info(f"损失收敛及隐私预算图已保存至 {save_path}")

#     # 最终的隐私预算
#     final_epsilon = privacy_engine.accountant.get_epsilon(delta=target_delta)
#     print(f"训练结束后的总隐私预算：ε = {final_epsilon:.2f}, δ = {target_delta}")
#     logging.info(f"训练结束后的总隐私预算：ε = {final_epsilon:.2f}, δ = {target_delta}")

# if __name__ == "__main__":
#     main()
import torch
import torch.nn as nn
import torch.optim as optim
from opacus import PrivacyEngine
from opacus.layers.dp_rnn import DPLSTM
from torch.utils.data import DataLoader, TensorDataset

# 假设有输入数据x, 标签y和权重w
x = torch.randn(1000, 10)  # 1000个样本，每个样本10维
y = torch.randint(0, 2, (1000,))  # 二分类标签
w = torch.rand(1000)  # 每个样本的权重

dataset = TensorDataset(x, y, w)
dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

# 定义使用DPLSTM的模型
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMModel, self).__init__()
        self.lstm = DPLSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # 假设x的形状为(batch_size, seq_length, input_size)
        out, _ = self.lstm(x.unsqueeze(1))  # 假设序列长度为1
        out = out[:, -1, :]  # 取最后一个时间步的输出
        out = self.fc(out)
        return out

model = LSTMModel(input_size=10, hidden_size=20, num_layers=2, num_classes=2)
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 初始化PrivacyEngine
privacy_engine = PrivacyEngine()
model, optimizer, dataloader = privacy_engine.make_private(
    module=model,
    optimizer=optimizer,
    data_loader=dataloader,
    noise_multiplier=4.0,
    max_grad_norm=1.0,
)

# 训练循环
num_epochs = 100
criterion = nn.CrossEntropyLoss(reduction='none')

for epoch in range(num_epochs):
    epoch_loss = 0.0
    for batch_x, batch_y, batch_w in dataloader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        losses = criterion(outputs, batch_y)
        weighted_losses = losses * batch_w
        loss = weighted_losses.mean()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    # 获取当前隐私损失（epsilon）
    epsilon = privacy_engine.accountant.get_epsilon(delta=1e-5)
    
    # 打印每个epoch的损失和隐私损失
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss / len(dataloader):.4f}, (ε = {epsilon:.2f})")