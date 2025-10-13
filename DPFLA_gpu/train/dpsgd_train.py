# # train/dp_train.py
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import random
# import numpy as np
# from models.dpsgd_lstm_model import DPSGDLSTMModel
# from data.dataset import PasswordDataset, PasswordDatasetExtend
# from utils.preprocessing import Preprocessor, SubstringDictGenerator, DPSubstringDictGenerator, PreprocessorDP
# from opacus import PrivacyEngine
# from torch.utils.data import DataLoader, TensorDataset
# import json
# import os
# from utils.dp_utils import GaussianMomentsAccountant

# class DPTrainer:
#     def __init__(self, config_path, data_path):
#         # 配置设备
#         # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         self.device = torch.device('cuda:6' if torch.cuda.is_available() else 'cpu')
#         print(f"Using device: {self.device}")

#         with open(config_path, 'r') as config_file:
#             self.config = json.load(config_file)
        
#         # 初始化数据集
#         self.password_dataset_extended = PasswordDatasetExtend(txt_path=data_path, config_path=config_path)
#         self.pwd_max_len = max(len(pwd) for pwd in self.password_dataset_extended)

#         # 初始化模型并移动到 GPU
#         self.model = DPSGDLSTMModel(config_path=config_path).to(self.device)
        
#         # 定义损失函数并移动到 GPU
#         self.criterion = nn.CrossEntropyLoss(reduction='none').to(self.device)
        
#         # 定义优化器
#         self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        
#         # 准备训练数据
#         self.pwd_whole_list = [self.password_dataset_extended[i] for i in range(len(self.password_dataset_extended))]
#         self.epoch_size = self.config.get('epoch_size', 10)
#         self.batch_per_epoch = self.config.get('batch_per_epoch', 30000)
#         # self.sample_rate = 0.01  # 采样率
#         self.sample_rate = self.config.get('sample_rate', 0.000035)
#         self.batch_print_size = self.config.get('batch_print_size', 10)
#         self.smoothed_loss = 0
#         self.alpha = 0.9  # 平滑系数
#         self.model_save_path = './models/model/rockyou_new_dpsgd_0.1415.pth'

#         # 初始化 PrivacyEngine
#         self.priv_accountant = GaussianMomentsAccountant(len(self.password_dataset_extended))
#         self.target_deltas = [1e-5, 0.05]
#         self.clip_norm = 1.0
#         self.noise_multiplier = 15.0
#         # self.noise_multiplier = 2.8

#         # 创建 dummy_loader 并将数据移动到 GPU
#         x = torch.randn(1000, 10, device=self.device)  # 1000个样本，每个样本10维，移动到 GPU
#         y = torch.randint(0, 2, (1000,), device=self.device)  # 二分类标签，移动到 GPU
#         w = torch.rand(1000, device=self.device)  # 每个样本的权重，移动到 GPU
#         dataset = TensorDataset(x, y, w)
#         dummy_loader = DataLoader(
#             dataset,
#             batch_size=10,
#             shuffle=True,
#             pin_memory=True if self.device.type == 'cuda' else False
#         )

#         # 初始化 PrivacyEngine
#         self.privacy_engine = PrivacyEngine()
#         self.model, self.optimizer, _ = self.privacy_engine.make_private(
#             module=self.model,
#             optimizer=self.optimizer,
#             data_loader=dummy_loader,  # 仅用于初始化 PrivacyEngine
#             noise_multiplier=self.noise_multiplier,
#             max_grad_norm=self.clip_norm,
#         )

#     def next_batch(self):
#         batch_size = max(1, int(len(self.pwd_whole_list) * self.sample_rate))
#         return random.sample(self.pwd_whole_list, batch_size)

#     def train_step(self, pwd_list):
#         # 预处理批次数据
#         x_batch, y_batch = zip(*pwd_list)

#         # 将批次转换为张量并移动到 GPU
#         x_tensor = torch.tensor(x_batch, dtype=torch.long).to(self.device, non_blocking=True)
#         y_tensor = torch.tensor(y_batch, dtype=torch.long).to(self.device, non_blocking=True)
        
#         # 每条数据的权重设为1并移动到 GPU
#         w_tensor = torch.ones(len(y_batch), dtype=torch.float).to(self.device, non_blocking=True)

#         # 进行一个 batch 的训练步骤
#         self.optimizer.zero_grad()
#         outputs = self.model(x_tensor)  # x_tensor 的形状为 (batch_size, seq_len)
#         loss = self.criterion(outputs, y_tensor)  # y_tensor 现在是字符索引，不需要 argmax
#         weighted_loss = (loss * w_tensor).mean()
#         weighted_loss.backward()
#         self.optimizer.step()

#         # 累积隐私支出
#         self.priv_accountant.accumulate_privacy_spending(
#             sigma=self.noise_multiplier,
#             num_examples=self.sample_rate * len(self.password_dataset_extended)
#         )

#         # 更新平滑损失
#         self.smoothed_loss = self.alpha * self.smoothed_loss + (1 - self.alpha) * weighted_loss.item()

#     def train_epoch(self):
#         for batch_idx in range(self.batch_per_epoch):
#             pwd_list = self.next_batch()
#             self.train_step(pwd_list)
#             if (batch_idx + 1) % self.batch_print_size == 0:
#                 print(f"Batch [{batch_idx + 1}/{self.batch_per_epoch}], Smoothed Loss: {self.smoothed_loss:.4f}")

#     def train(self):
#         for epoch in range(self.epoch_size):
#             print(f"Starting Epoch {epoch + 1}/{self.epoch_size}")
#             self.train_epoch()
#             eps_deltas = self.priv_accountant.get_privacy_spent(target_deltas=self.target_deltas)
#             for eps, delta in eps_deltas:
#                 print(f"Epoch [{epoch+1}/{self.epoch_size}] completed, (ε = {eps:.2f}, δ = {delta})")
#             if (epoch+1) % 5 == 0:
#                 os.makedirs(os.path.dirname(f"./models/model/rockyou_new_dpsgd_0.1415_{epoch}.pth"), exist_ok=True)
#                 torch.save(self.model.state_dict(), f"./models/model/rockyou_new_dpsgd_0.1415_{epoch}.pth")
#                 print(f"Model saved to ./models/model/rockyou_new_dpsgd_0.1415_{epoch}.pth")
#             if epoch == 0:
#                 os.makedirs(os.path.dirname(f"./models/model/rockyou_new_dpsgd_0.1415_{epoch}.pth"), exist_ok=True)
#                 torch.save(self.model.state_dict(), f"./models/model/rockyou_new_dpsgd_0.1415_{epoch}.pth")
#                 print(f"Model saved to ./models/model/rockyou_new_dpsgd_0.1415_{epoch}.pth")
#         os.makedirs(os.path.dirname(self.model_save_path), exist_ok=True)
#         torch.save(self.model.state_dict(), self.model_save_path)
#         print(f"Model saved to {self.model_save_path}")
# train/dp_train.py
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from models.dpsgd_lstm_model import DPSGDLSTMModel
from data.dataset import PasswordDataset, PasswordDatasetExtend
from utils.preprocessing import Preprocessor, SubstringDictGenerator, DPSubstringDictGenerator, PreprocessorDP
from opacus import PrivacyEngine
from torch.utils.data import DataLoader, TensorDataset
import json
import os
from utils.dp_utils import GaussianMomentsAccountant

class DPTrainer:
    def __init__(self, config_path, data_path):
        # 配置设备
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')
        # self.device = 'cpu'
        print(f"Using device: {self.device}")

        with open(config_path, 'r') as config_file:
            self.config = json.load(config_file)
        
        # 初始化数据集
        self.password_dataset_extended = PasswordDatasetExtend(txt_path=data_path, config_path=config_path)
        self.pwd_max_len = max(len(pwd) for pwd in self.password_dataset_extended)

        # 初始化模型并移动到 GPU
        self.model = DPSGDLSTMModel(config_path=config_path).to(self.device)
        
        # 定义损失函数并移动到 GPU
        self.criterion = nn.CrossEntropyLoss(reduction='none').to(self.device)
        
        # 定义优化器
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        
        # 准备训练数据
        self.pwd_whole_list = [self.password_dataset_extended[i] for i in range(len(self.password_dataset_extended))]
        self.epoch_size = self.config.get('epoch_size', 10)
        self.batch_per_epoch = self.config.get('batch_per_epoch', 30000)
        # self.sample_rate = 0.01  # 采样率
        self.sample_rate = self.config.get('sample_rate', 0.000035)
        self.batch_print_size = self.config.get('batch_print_size', 10)
        self.smoothed_loss = 0
        self.alpha = 0.9  # 平滑系数
        self.model_save_path = './models/model/rockyou_new_dpsgd_1.pth'
        self.smoothed_losses = []

        # 初始化 PrivacyEngine
        self.priv_accountant = GaussianMomentsAccountant(len(self.password_dataset_extended))
        self.target_deltas = [1e-5, 0.05]
        self.clip_norm = 2.0
        self.noise_multiplier = 2 * 3.1
        # self.noise_multiplier = 2.8

        # 创建 dummy_loader 并将数据移动到 GPU
        x = torch.randn(1000, 10, device=self.device)  # 1000个样本，每个样本10维，移动到 GPU
        y = torch.randint(0, 2, (1000,), device=self.device)  # 二分类标签，移动到 GPU
        w = torch.rand(1000, device=self.device)  # 每个样本的权重，移动到 GPU
        dataset = TensorDataset(x, y, w)
        dummy_loader = DataLoader(
            dataset,
            batch_size=10,
            shuffle=True,
            pin_memory=True if self.device.type == 'cuda' else False
        )

        # 初始化 PrivacyEngine
        self.privacy_engine = PrivacyEngine()
        self.model, self.optimizer, _ = self.privacy_engine.make_private(
            module=self.model,
            optimizer=self.optimizer,
            data_loader=dummy_loader,  # 仅用于初始化 PrivacyEngine
            noise_multiplier=self.noise_multiplier,
            max_grad_norm=self.clip_norm,
        )

    def next_batch(self):
        batch_size = max(1, int(len(self.pwd_whole_list) * self.sample_rate))
        return random.sample(self.pwd_whole_list, batch_size)

    def train_step(self, pwd_list):
        # 预处理批次数据
        x_batch, y_batch = zip(*pwd_list)

        # 将批次转换为张量并移动到 GPU
        x_tensor = torch.tensor(x_batch, dtype=torch.long).to(self.device, non_blocking=True)
        y_tensor = torch.tensor(y_batch, dtype=torch.long).to(self.device, non_blocking=True)
        
        # 每条数据的权重设为1并移动到 GPU
        w_tensor = torch.ones(len(y_batch), dtype=torch.float).to(self.device, non_blocking=True)

        # 进行一个 batch 的训练步骤
        self.optimizer.zero_grad()
        outputs = self.model(x_tensor)  # x_tensor 的形状为 (batch_size, seq_len)
        loss = self.criterion(outputs, y_tensor)  # y_tensor 现在是字符索引，不需要 argmax
        weighted_loss = (loss * w_tensor).mean()
        weighted_loss.backward()
        self.optimizer.step()

        # 累积隐私支出
        self.priv_accountant.accumulate_privacy_spending(
            sigma=self.noise_multiplier,
            num_examples=self.sample_rate * len(self.password_dataset_extended)
        )

        # 更新平滑损失
        self.smoothed_loss = self.alpha * self.smoothed_loss + (1 - self.alpha) * weighted_loss.item()
        self.smoothed_losses.append(self.smoothed_loss)

    def train_epoch(self,epoch):
        for batch_idx in range(self.batch_per_epoch):
            pwd_list = self.next_batch()
            self.train_step(pwd_list)
            if (batch_idx + 1) % self.batch_print_size == 0:
                print(f"Epoch {epoch+1}, Batch [{batch_idx + 1}/{self.batch_per_epoch}], Smoothed Loss: {self.smoothed_loss:.4f}")

    def train(self):
        for epoch in range(self.epoch_size):
            print(f"Starting Epoch {epoch + 1}/{self.epoch_size}")
            self.train_epoch(epoch)
            eps_deltas = self.priv_accountant.get_privacy_spent(target_deltas=self.target_deltas)
            for eps, delta in eps_deltas:
                print(f"Epoch [{epoch+1}/{self.epoch_size}] completed, (ε = {eps:.2f}, δ = {delta})")
            if (epoch+1) % 5 == 0:
                os.makedirs(os.path.dirname(f"./models/model/rockyou_new_dpsgd_1_{epoch+1}.pth"), exist_ok=True)
                torch.save(self.model.state_dict(), f"./models/model/rockyou_new_dpsgd_1_{epoch+1}.pth")
                print(f"Model saved to ./models/model/rockyou_new_dpsgd_1_{epoch+1}.pth")
            if epoch == 0:
                os.makedirs(os.path.dirname(f"./models/model/rockyou_new_dpsgd_1_{epoch+1}.pth"), exist_ok=True)
                torch.save(self.model.state_dict(), f"./models/model/rockyou_new_dpsgd_1_{epoch+1}.pth")
                print(f"Model saved to ./models/model/rockyou_new_dpsgd_1_{epoch+1}.pth")
        os.makedirs(os.path.dirname(self.model_save_path), exist_ok=True)
        torch.save(self.model.state_dict(), self.model_save_path)
        print(f"Model saved to {self.model_save_path}")
        
        # Save smoothed losses to a file
        with open("./models/model/losses_rock_city_dpsgd_1.json", "w") as f:
            json.dump(self.smoothed_losses, f)
        print("Smoothed losses saved to ./models/model/losses_rock_city_dpsgd_1.json")
