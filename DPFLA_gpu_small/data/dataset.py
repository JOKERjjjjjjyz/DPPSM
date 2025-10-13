# data/dataset.py
import torch
from torch.utils.data import Dataset

class PasswordDataset(Dataset):
    def __init__(self, txt_path):
        # 读取TXT文件，每行一个密码
        with open(txt_path, 'r') as file:
            self.passwords = [line.strip() for line in file]
        # 记录密码中最长的长度
        self.max_length = max(len(password) for password in self.passwords)

    def __len__(self):
        return len(self.passwords)

    def __getitem__(self, idx):
        password = self.passwords[idx]
        return password
    
import torch
import string
import json
import os

class PasswordDatasetExtend:
    def __init__(self, txt_path, config_path):
        # 读取配置文件
        with open(config_path, 'r') as config_file:
            config = json.load(config_file)
        
        self.lstm_seq_len = config['lstm_seq_len']
        self.start_token = '\t'  # 定义起始符号
        self.end_token = '\n'    # 定义结束符号
        self.valid_chars = string.ascii_letters + string.digits + string.punctuation + ' ' + self.start_token + self.end_token
        self.char_to_index = {char: idx for idx, char in enumerate(self.valid_chars)}
        
        # 读取TXT文件，每行一个密码
        with open(txt_path, 'r') as file:
            self.passwords = [line.strip() for line in file]
        
        # 生成扩展后的密码序列
        self.passwords_extended = []
        for password in self.passwords:
            extended_sequences = self._extend_password(password)
            self.passwords_extended.extend(extended_sequences)

    def _extend_password(self, password):
        """
        扩展每个密码，生成所有可能的前缀，并对其进行截断或填充。
        """
        # 添加起始和结束符号
        password = self.start_token + password + self.end_token
        extended_sequences = []
        
        # 切分 s1, s1s2, ..., s1s2...sn
        for i in range(1, len(password)):
            x_seq = password[:i]
            y_seq = password[i]
            
            # 处理 x_seq 的长度
            if len(x_seq) > self.lstm_seq_len:
                x_seq = x_seq[-self.lstm_seq_len:]  # 截断，保留最后部分
            else:
                x_seq = (self.start_token * (self.lstm_seq_len - len(x_seq))) + x_seq  # 补全，前端补 start
            
            # 将字符转换为索引
            x_indices = [self.char_to_index[c] for c in x_seq]
            y_index = self.char_to_index[y_seq]
            
            # 存储 x 和 y 的索引序列
            extended_sequences.append((x_indices, y_index))
        
        return extended_sequences

    def __len__(self):
        return len(self.passwords_extended)

    def __getitem__(self, idx):
        return self.passwords_extended[idx]