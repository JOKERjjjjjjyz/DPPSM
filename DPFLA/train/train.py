# train/train.py
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from models.lstm_model import LSTMModel
from data.dataset import PasswordDataset
from utils.preprocessing import Preprocessor
import json
import os

class Trainer:
    def __init__(self, config_path, data_path):
        with open(config_path, 'r') as config_file:
            self.config = json.load(config_file)
        
        self.password_dataset = PasswordDataset(txt_path=data_path)
        self.preprocessor = Preprocessor(config_path=config_path)
        self.model = LSTMModel(config_path=config_path)
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.1)
        self.pwd_whole_list = [self.password_dataset[i] for i in range(len(self.password_dataset))]
        self.epoch_size = self.config.get('epoch_size', 10)
        self.batch_per_epoch = self.config.get('batch_per_epoch', 100)
        self.sample_rate = 0.01
        self.batch_print_size = self.config.get('batch_print_size', 10)
        self.smoothed_loss = 0
        self.alpha = 0.9  # 平滑系数
        self.model_save_path = f"{self.config.get('model_file', './models/model/rockyou10w.pth')}"

    def next_batch(self):
        batch_size = max(1, int(len(self.pwd_whole_list) * self.sample_rate))
        return random.sample(self.pwd_whole_list, batch_size)

    # def train_step(self, pwd_list):
    #     # 预处理批次数据
    #     x_batches, y_batches, w_batches = self.preprocessor.preprocess(pwd_list)
    #     for x, y, w in zip(x_batches, y_batches, w_batches):
    #         # 现在 x 和 y 是索引序列，而不是 one-hot 编码
    #         self.optimizer.zero_grad()
    #         outputs = self.model(x)  # x 的形状为 (batch_size, seq_len)
    #         loss = self.criterion(outputs, y)  # y 现在是字符索引，不需要 argmax
    #         weighted_loss = (loss * w).mean()
    #         weighted_loss.backward()
    #         self.optimizer.step()

    #         # 更新平滑损失
    #         self.smoothed_loss = self.alpha * self.smoothed_loss + (1 - self.alpha) * weighted_loss.item()
    def train_step(self, pwd_list):
        # 预处理批次数据
        x_batches, y_batches, w_batches = self.preprocessor.preprocess(pwd_list)
        # input(y_batches)
        # 合并批次数据
        x_batch = torch.cat(x_batches, dim=0)  # 合并所有 x_batches
        y_batch = torch.cat(y_batches, dim=0)  # 合并所有 y_batches
        w_batch = torch.cat(w_batches, dim=0)  # 合并所有 w_batches
        # input(f"{y_batch}")
        # 进行一个 batch 的训练步骤
        self.optimizer.zero_grad()
        outputs = self.model(x_batch)  # x_batch 的形状为 (batch_size, seq_len)
        loss = self.criterion(outputs, y_batch)  # y_batch 现在是字符索引，不需要 argmax
        weighted_loss = (loss * w_batch).mean()
        weighted_loss.backward()
        self.optimizer.step()

        # 更新平滑损失
        self.smoothed_loss = self.alpha * self.smoothed_loss + (1 - self.alpha) * weighted_loss.item()


    def train_epoch(self):
        for batch_idx in range(self.batch_per_epoch):
            pwd_list = self.next_batch()
            self.train_step(pwd_list)
            if (batch_idx + 1) % self.batch_print_size == 0:
                print(f"Batch [{batch_idx + 1}/{self.batch_per_epoch}], Smoothed Loss: {self.smoothed_loss:.4f}")

    def train(self):
        for epoch in range(self.epoch_size):
            self.train_epoch()
            print(f"Epoch [{epoch+1}/{self.epoch_size}] completed")
        os.makedirs(os.path.dirname(self.model_save_path), exist_ok=True)
        torch.save(self.model.state_dict(), self.model_save_path)
        print(f"Model saved to {self.model_save_path}")