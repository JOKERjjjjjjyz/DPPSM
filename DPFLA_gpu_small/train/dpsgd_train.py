# train/dp_train.py (已添加 DP 开关和 tqdm)
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from models.dpsgd_lstm_model import DPSGDLSTMModel
from data.dataset import PasswordDataset, PasswordDatasetExtend
from utils.preprocessing import Preprocessor
from opacus import PrivacyEngine
from torch.utils.data import DataLoader, TensorDataset
import json
import os
from utils.dp_utils import GaussianMomentsAccountant
from tqdm import tqdm

class DPTrainer:
    def __init__(self, config_path, data_path):
        # 配置设备
        self.device = torch.device('cuda:5' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        with open(config_path, 'r') as config_file:
            self.config = json.load(config_file)
        
        # 初始化数据集
        self.password_dataset_extended = PasswordDatasetExtend(txt_path=data_path, config_path=config_path)
        self.pwd_max_len = max(len(pwd) for pwd in self.password_dataset_extended)

        # 初始化模型、损失函数和优化器
        self.model = DPSGDLSTMModel(config_path=config_path).to(self.device)
        self.criterion = nn.CrossEntropyLoss(reduction='none').to(self.device)
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        
        # 准备训练数据
        self.pwd_whole_list = [self.password_dataset_extended[i] for i in range(len(self.password_dataset_extended))]
        self.epoch_size = self.config.get('epoch_size', 10)
        self.batch_per_epoch = self.config.get('batch_per_epoch', 3000)
        self.sample_rate = self.config.get('sample_rate', 0.00035)
        self.smoothed_loss = 0.0
        self.alpha = 0.98
        self.model_save_path = self.config.get('model_file', './models/model/default_dpsgd_model.pth')

        # ## --- DP 开关逻辑 --- ##
        if self.config.get("dp_enabled", False):
            print("INFO: Differential Privacy is ENABLED. Attaching PrivacyEngine.")
            
            # 初始化 PrivacyEngine 所需参数
            self.priv_accountant = GaussianMomentsAccountant(len(self.password_dataset_extended))
            self.target_deltas = [1e-5, 0.05]
            self.clip_norm = 1.0
            self.noise_multiplier = self.config.get("noise_multiplier", 70)

            # 创建 dummy_loader 用于初始化
            x = torch.randn(1000, 10, device=self.device)
            y = torch.randint(0, 2, (1000,), device=self.device)
            dataset = TensorDataset(x, y)
            dummy_loader = DataLoader(dataset, batch_size=10)

            # 附加 PrivacyEngine，"改装"模型和优化器
            self.privacy_engine = PrivacyEngine()
            self.model, self.optimizer, _ = self.privacy_engine.make_private(
                module=self.model,
                optimizer=self.optimizer,
                data_loader=dummy_loader,
                noise_multiplier=self.noise_multiplier,
                max_grad_norm=self.clip_norm,
            )
        else:
            print("INFO: Differential Privacy is DISABLED. Training in non-private mode.")
            # 如果不启用DP，则不进行任何操作，并将 accountant 设为 None
            self.priv_accountant = None

    def next_batch(self):
        batch_size = max(1, int(len(self.pwd_whole_list) * self.sample_rate))
        return random.sample(self.pwd_whole_list, batch_size)

    def train_step(self, pwd_list):
        x_batch, y_batch = zip(*pwd_list)

        x_tensor = torch.tensor(x_batch, dtype=torch.long).to(self.device, non_blocking=True)
        y_tensor = torch.tensor(y_batch, dtype=torch.long).to(self.device, non_blocking=True)
        w_tensor = torch.ones(len(y_batch), dtype=torch.float).to(self.device, non_blocking=True)

        self.optimizer.zero_grad()
        outputs = self.model(x_tensor)
        loss = self.criterion(outputs, y_tensor)
        weighted_loss = (loss * w_tensor).mean()
        weighted_loss.backward()
        self.optimizer.step()

        # 仅在启用DP时累积隐私支出
        if self.priv_accountant:
            self.priv_accountant.accumulate_privacy_spending(
                sigma=self.noise_multiplier,
                num_examples=self.sample_rate * len(self.password_dataset_extended)
            )

        # 更新平滑损失
        loss_item = weighted_loss.item()
        if self.smoothed_loss == 0.0:
            self.smoothed_loss = loss_item
        else:
            self.smoothed_loss = self.alpha * self.smoothed_loss + (1 - self.alpha) * loss_item

    def train_epoch(self, epoch: int):
        # 使用 tqdm 显示进度条
        pbar = tqdm(range(self.batch_per_epoch), desc=f"Epoch {epoch+1}/{self.epoch_size}", unit="batch")
        for _ in pbar:
            pwd_list = self.next_batch()
            self.train_step(pwd_list)
            pbar.set_postfix(smoothed_loss=f'{self.smoothed_loss:.4f}')

    def train(self):
        for epoch in range(self.epoch_size):
            self.train_epoch(epoch)
            
            # 仅在启用DP时计算并打印 epsilon
            if self.priv_accountant:
                eps_deltas = self.priv_accountant.get_privacy_spent(target_deltas=self.target_deltas)
                for eps, delta in eps_deltas:
                    print(f"Epoch [{epoch+1}/{self.epoch_size}] completed, (ε = {eps:.2f}, δ = {delta})")
            else:
                print(f"Epoch [{epoch+1}/{self.epoch_size}] completed in non-private mode.")

            # 保存模型检查点
            if (epoch+1) % 5 == 0:
                # 动态生成保存路径，避免硬编码
                base_name = os.path.splitext(os.path.basename(self.model_save_path))[0]
                save_path = f"./models/checkpoints/{base_name}_epoch_{epoch+1}.pth"
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(self.model.state_dict(), save_path)
                print(f"Model checkpoint saved to {save_path}")

        # 保存最终模型
        os.makedirs(os.path.dirname(self.model_save_path), exist_ok=True)
        torch.save(self.model.state_dict(), self.model_save_path)
        print(f"Final model saved to {self.model_save_path}")