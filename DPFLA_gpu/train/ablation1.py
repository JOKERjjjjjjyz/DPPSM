# train/dpfla_train.py
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from models.dpsgd_lstm_model import DPSGDLSTMModel
from data.dataset import PasswordDataset
from utils.preprocessing import Preprocessor_DPSGD
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
        print(f"Using device: {self.device}")

        with open(config_path, 'r') as config_file:
            self.config = json.load(config_file)
        
        self.password_dataset = PasswordDataset(txt_path=data_path)
        self.pwd_max_len = max(len(pwd) for pwd in self.password_dataset)
        self.preprocessor = Preprocessor_DPSGD(config_path)
        self.model = DPSGDLSTMModel(config_path=config_path).to(self.device)  # 将模型移动到 GPU
        self.criterion = nn.CrossEntropyLoss(reduction='none').to(self.device)  # 将损失函数移动到 GPU
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        self.pwd_whole_list = [self.password_dataset[i] for i in range(len(self.password_dataset))]
        self.epoch_size = self.config.get('epoch_size', 10)
        self.batch_per_epoch = self.config.get('batch_per_epoch', 10000)
        # 使用计算出的 sample_rate
        self.sample_rate = self.config.get('sample_rate', 0.0001)
        self.batch_print_size = self.config.get('batch_print_size', 10)
        self.smoothed_loss = 0
        self.alpha = 0.9  # 平滑系数
        # self.model_save_path = f"{self.config.get('pwd_file', './model/rockyou10w_dp.pth')}"
        self.model_save_path = './models/ablation/rockyou_new_abla1_0.9701.pth'
        self.smoothed_losses = []

        # 创建 dummy_loader 并将数据移动到 GPU
        x = torch.randn(1000, 10, device=self.device)  # 1000个样本，每个样本10维，移动到 GPU
        y = torch.randint(0, 2, (1000,), device=self.device)  # 二分类标签，移动到 GPU
        w = torch.rand(1000, device=self.device)  # 每个样本的权重，移动到 GPU
        dataset = TensorDataset(x, y, w)
        dummy_loader = DataLoader(dataset, batch_size=10, shuffle=True, pin_memory=True if self.device.type == 'cuda' else False)

        # 初始化 PrivacyEngine
        self.privacy_engine = PrivacyEngine()
        self.clip_norm = 1.0
        # self.noise_multiplier_account = 1.15
        self.noise_multiplier_account = 0.72
        self.noise_multiplier = 32 * 2 * self.noise_multiplier_account
        self.model, self.optimizer, _ = self.privacy_engine.make_private(
            module=self.model,
            optimizer=self.optimizer,
            data_loader=dummy_loader,  # 仅用于初始化 PrivacyEngine
            noise_multiplier=self.noise_multiplier,
            max_grad_norm=self.clip_norm,
        )
        self.priv_accountant = GaussianMomentsAccountant(len(self.password_dataset))
        self.target_deltas = [1e-5, 0.05]

    def next_batch(self):
        batch_size = max(1, int(len(self.pwd_whole_list) * self.sample_rate))
        return random.sample(self.pwd_whole_list, batch_size)

    def train_step(self, pwd_list):
        # 预处理批次数据
        x_batches, y_batches, w_batches = self.preprocessor.preprocess(pwd_list)
        
        # 合并批次数据
        x_batch = torch.cat(x_batches, dim=0).to(self.device, non_blocking=True)  # 合并所有 x_batches 并移动到 GPU
        y_batch = torch.cat(y_batches, dim=0).to(self.device, non_blocking=True)  # 合并所有 y_batches 并移动到 GPU
        w_batch = torch.cat(w_batches, dim=0).to(self.device, non_blocking=True)  # 合并所有 w_batches 并移动到 GPU

        # 进行一个 batch 的训练步骤
        self.optimizer.zero_grad()
        outputs = self.model(x_batch)  # x_batch 的形状为 (batch_size, seq_len)
        loss = self.criterion(outputs, y_batch)  # y_batch 现在是字符索引，不需要 argmax
        weighted_loss = (loss * w_batch).mean()
        weighted_loss.backward()
        self.optimizer.step()
        self.priv_accountant.accumulate_privacy_spending(sigma=self.noise_multiplier_account, num_examples=self.sample_rate * len(self.password_dataset))

        # 更新平滑损失
        self.smoothed_loss = self.alpha * self.smoothed_loss + (1 - self.alpha) * weighted_loss.item()
        self.smoothed_losses.append(self.smoothed_loss)

    def train_epoch(self,epoch):
        for batch_idx in range(self.batch_per_epoch):
            pwd_list = self.next_batch()
            self.train_step(pwd_list)
            if (batch_idx + 1) % self.batch_print_size == 0:
                print(f"Epoch {epoch+1}, Batch [{batch_idx + 1}/{self.batch_per_epoch}], Smoothed Loss: {self.smoothed_loss:.4f}")
                # eps_deltas = self.priv_accountant.get_privacy_spent(target_deltas=self.target_deltas)
                # for eps, delta in eps_deltas:
                #     print(f"(ε = {eps:.2f}, δ = {delta})")

    def train(self):
        for epoch in range(self.epoch_size):
            print(f"Starting Epoch {epoch + 1}/{self.epoch_size}")
            self.train_epoch(epoch)
            eps_deltas = self.priv_accountant.get_privacy_spent(target_deltas=self.target_deltas)
            for eps, delta in eps_deltas:
                print(f"Epoch [{epoch+1}/{self.epoch_size}] completed, (ε = {eps:.2f}, δ = {delta})")
            if (epoch+1) % 5 == 0:
                os.makedirs(os.path.dirname(f"./models/ablation/rockyou_new_abla1_0.9701_{epoch+1}.pth"), exist_ok=True)
                torch.save(self.model.state_dict(), f"./models/ablation/rockyou_new_abla1_0.9701_{epoch+1}.pth")
                print(f"Model saved to ./models/ablation/rockyou_new_abla1_0.9701_{epoch+1}.pth")
            if epoch == 0:
                os.makedirs(os.path.dirname(f"./models/ablation/rockyou_new_abla1_0.9701_{epoch+1}.pth"), exist_ok=True)
                torch.save(self.model.state_dict(), f"./models/ablation/rockyou_new_abla1_0.9701_{epoch+1}.pth")
                print(f"Model saved to ./models/ablation/rockyou_new_abla1_0.9701_{epoch+1}.pth")
        os.makedirs(os.path.dirname(self.model_save_path), exist_ok=True)
        torch.save(self.model.state_dict(), self.model_save_path)
        print(f"Model saved to {self.model_save_path}")

        with open("./models/ablation/losses_rock_city_abla1_0.9701.json", "w") as f:
            json.dump(self.smoothed_losses, f)
        print("Smoothed losses saved to ./models/ablation/losses_rock_city_abla1_0.9701.json")
