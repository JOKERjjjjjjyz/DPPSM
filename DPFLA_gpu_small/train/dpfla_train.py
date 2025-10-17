# train/dpfla_train.py (Final version with tqdm only, TensorBoard removed)
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from models.dpsgd_lstm_model import DPSGDLSTMModel
from data.dataset import PasswordDataset
from utils.preprocessing import Preprocessor, SubstringDictGenerator, DPSubstringDictGenerator, PreprocessorDP
from opacus import PrivacyEngine
from torch.utils.data import DataLoader, TensorDataset
import json
import os
from utils.dp_utils import GaussianMomentsAccountant
from tqdm import tqdm

class DPTrainer:
    def __init__(self, config_path, data_path):
        # 配置设备
        self.device = torch.device('cuda:6' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        with open(config_path, 'r') as config_file:
            self.config = json.load(config_file)
        
        self.password_dataset = PasswordDataset(txt_path=data_path)
        self.substring_generator = SubstringDictGenerator(self.password_dataset, config_path)
        self.substring_dict = self.substring_generator.generate()
        self.DPsubstring_generator = DPSubstringDictGenerator(substring_dict=self.substring_dict, config_path=config_path)
        self.DPsubstring_dict = self.DPsubstring_generator.generate_dp()
        self.pwd_max_len = max(len(pwd) for pwd in self.password_dataset)
        self.preprocessor = PreprocessorDP(config_path, self.DPsubstring_dict, self.pwd_max_len)
        self.model = DPSGDLSTMModel(config_path=config_path).to(self.device)
        self.criterion = nn.CrossEntropyLoss(reduction='none').to(self.device)
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        self.pwd_whole_list = [self.password_dataset[i] for i in range(len(self.password_dataset))]
        self.epoch_size = self.config.get('epoch_size', 10)
        self.batch_per_epoch = self.config.get('batch_per_epoch', 1000)
        self.sample_rate = self.config.get('sample_rate', 0.001)
        self.smoothed_loss = 0.0
        self.alpha = 0.98
        self.model_save_path = self.config.get('model_file', './models/model/default_model.pth')
        print(f"INFO: Model final save path set from config: {self.model_save_path}")

        x = torch.randn(1000, 10, device=self.device)
        y = torch.randint(0, 2, (1000,), device=self.device)
        w = torch.rand(1000, device=self.device)
        dataset = TensorDataset(x, y, w)
        dummy_loader = DataLoader(dataset, batch_size=10, shuffle=True, pin_memory=True if self.device.type == 'cuda' else False)
        if self.config.get("dp_enabled", False): # 默认为False（不加噪）
            print("INFO: Differential Privacy is ENABLED. Attaching PrivacyEngine.")
            self.privacy_engine = PrivacyEngine()
            self.clip_norm = 1.0
            self.noise_multiplier_account = self.config.get("noise_multiplier_account", 3.0)
            self.noise_multiplier = 2 * self.clip_norm * self.noise_multiplier_account / (len(self.password_dataset) * self.sample_rate)
            self.model, self.optimizer, _ = self.privacy_engine.make_private(
                module=self.model,
                optimizer=self.optimizer,
                data_loader=dummy_loader,
                noise_multiplier=self.noise_multiplier,
                max_grad_norm=self.clip_norm,
            )
            self.priv_accountant = GaussianMomentsAccountant(len(self.password_dataset))
            self.target_deltas = [1e-5, 0.05]
        else:
            print("INFO: Differential Privacy is DISABLED. Training in non-private mode.")
            # 如果禁用DP，则 priv_accountant 设为 None，避免后续代码出错
            self.priv_accountant = None
            
    def next_batch(self):
        batch_size = max(1, int(len(self.pwd_whole_list) * self.sample_rate))
        return random.sample(self.pwd_whole_list, batch_size)

    def train_step(self, pwd_list):
        x_batches, y_batches, w_batches = self.preprocessor.preprocess(pwd_list)
        
        x_batch = torch.cat(x_batches, dim=0).to(self.device, non_blocking=True)
        y_batch = torch.cat(y_batches, dim=0).to(self.device, non_blocking=True)
        w_batch = torch.cat(w_batches, dim=0).to(self.device, non_blocking=True)

        self.optimizer.zero_grad()
        outputs = self.model(x_batch)
        loss = self.criterion(outputs, y_batch)
        weighted_loss = (loss * w_batch).mean()
        weighted_loss.backward()
        self.optimizer.step()
        self.priv_accountant.accumulate_privacy_spending(sigma=self.noise_multiplier_account, num_examples=self.sample_rate * len(self.password_dataset))

        loss_item = weighted_loss.item()
        if self.smoothed_loss == 0.0:
            self.smoothed_loss = loss_item
        else:
            self.smoothed_loss = self.alpha * self.smoothed_loss + (1 - self.alpha) * loss_item

    def train_epoch(self, epoch):
        pbar = tqdm(range(self.batch_per_epoch), desc=f"Epoch {epoch+1}/{self.epoch_size}", unit="batch")
        for batch_idx in pbar:
            pwd_list = self.next_batch()
            self.train_step(pwd_list)
            pbar.set_postfix(smoothed_loss=f'{self.smoothed_loss:.4f}')

    def train(self):
        for epoch in range(self.epoch_size):
            self.train_epoch(epoch)
            eps_deltas = self.priv_accountant.get_privacy_spent(target_deltas=self.target_deltas)
            for eps, delta in eps_deltas:
                print(f"Epoch [{epoch+1}/{self.epoch_size}] completed. (ε = {eps:.2f}, δ = {delta})")
            if (epoch+1) % 5 == 0:
                save_path = f"./models/model/rockyou320w_dpfla_0.1178_{epoch}.pth"
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(self.model.state_dict(), save_path)
                print(f"Model saved to {save_path}")
        
        # 4. 移除了关闭 writer 的调用
        # self.writer.close()
        
        os.makedirs(os.path.dirname(self.model_save_path), exist_ok=True)
        torch.save(self.model.state_dict(), self.model_save_path)
        print(f"Model saved to {self.model_save_path}")