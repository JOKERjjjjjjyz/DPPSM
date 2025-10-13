# utils/preprocessing.py
import torch
import string
import numpy as np
import json
import os
import pickle
from utils.dp_utils import AboveThreshold

class Preprocessor:
    def __init__(self, config_path):
        with open(config_path, 'r') as config_file:
            self.config = json.load(config_file)
        self.lstm_seq_len = self.config['lstm_seq_len']
        self.start_token = '\t'  # 定义起始符号
        self.end_token = '\n'    # 定义结束符号
        self.valid_chars = string.ascii_letters + string.digits + string.punctuation + ' ' + self.start_token + self.end_token
        self.char_to_index = {char: idx for idx, char in enumerate(self.valid_chars)}
        self.vocab_size = len(self.valid_chars)
        self.index_to_char = {idx: char for char, idx in self.char_to_index.items()}

    def preprocess(self, pwd_list):
        x_batches = []
        y_batches = []
        w_batches = []

        for pwd in pwd_list:
            # 添加起始和结束符号
            pwd = self.start_token + pwd + self.end_token

            # 如果包含无效字符，则跳过此密码
            if any(c not in self.valid_chars for c in pwd):
                continue

            # 切分 x, y, w
            x_splits = []
            y_splits = []
            w_splits = []
            for i in range(1, len(pwd)):
                x_seq = pwd[:i]
                y_seq = pwd[i]
                # weight = i
                weight = 1

                # 处理 x_seq 的长度
                if len(x_seq) > self.lstm_seq_len:
                    x_seq = x_seq[-self.lstm_seq_len:]  # 截断，保留最后部分
                else:
                    x_seq = (self.start_token * (self.lstm_seq_len - len(x_seq))) + x_seq  # 补全，前端补 start

                # 使用字符索引而不是 one-hot 编码
                x_indices = [self.char_to_index[c] for c in x_seq]
                y_index = self.char_to_index[y_seq]

                x_tensor = torch.tensor(x_indices, dtype=torch.long)
                y_tensor = torch.tensor(y_index, dtype=torch.long)

                x_splits.append(x_tensor)
                y_splits.append(y_tensor)
                w_splits.append(weight)
                
            x_batches.append(torch.stack(x_splits))
            y_batches.append(torch.stack(y_splits))
            w_batches.append(torch.tensor(w_splits, dtype=torch.float))

        return x_batches, y_batches, w_batches

class Preprocessor_Ablation1:
    def __init__(self, config_path, substring_dict, pwd_max_len):
        with open(config_path, 'r') as config_file:
            self.config = json.load(config_file)
        self.lstm_seq_len = self.config['lstm_seq_len']
        self.start_token = '\t'  # 定义起始符号
        self.end_token = '\n'    # 定义结束符号
        self.valid_chars = string.ascii_letters + string.digits + string.punctuation + ' ' + self.start_token + self.end_token
        self.char_to_index = {char: idx for idx, char in enumerate(self.valid_chars)}
        self.index_to_char = {idx: char for char, idx in self.char_to_index.items()}
        self.vocab_size = len(self.valid_chars)
        self.alpha = 1.5
        self.substring_dict = substring_dict
        self.threshold = self.config.get('threshold', 0)
        self.pwd_max_len = pwd_max_len
        self.epsilon_Abovethreshold = self.config.get('epsilon_Abovethreshold', 0.1)
        self.threshold_DP = self.threshold + np.random.laplace(0, 2 * (self.pwd_max_len + 1) / self.epsilon_Abovethreshold, size=1).item()

    def preprocess(self, pwd_list):
        x_batches = []
        y_batches = []
        w_batches = []

        for pwd in pwd_list:
            # 添加起始和结束符号
            pwd = self.start_token + pwd + self.end_token

            # 如果包含无效字符，则跳过此密码
            if any(c not in self.valid_chars for c in pwd):
                continue

            # 切分 x, y, w
            x_splits = []
            y_splits = []
            w_splits = []
            for i in range(1, len(pwd)):
                x_seq = pwd[:i]
                y_seq = pwd[i]
                weight = 1
                combined_seq = x_seq + y_seq

                # 处理 x_seq 的长度，并计算权重
                if len(x_seq) > self.lstm_seq_len:
                    x_seq = x_seq[-self.lstm_seq_len:]  # 截断，保留最后部分
                    if self.substring_dict[combined_seq] < self.threshold_DP:
                        continue
                else:
                    if self.substring_dict[combined_seq] < self.threshold_DP:
                        continue
                    x_seq = (self.start_token * (self.lstm_seq_len - len(x_seq))) + x_seq  # 补全，前端补 start

                # 使用字符索引而不是 one-hot 编码
                x_indices = [self.char_to_index[c] for c in x_seq]
                y_index = self.char_to_index[y_seq]

                x_tensor = torch.tensor(x_indices, dtype=torch.long)
                y_tensor = torch.tensor(y_index, dtype=torch.long)

                x_splits.append(x_tensor)
                y_splits.append(y_tensor)
                w_splits.append(weight)

            if w_splits:                
                x_batches.append(torch.stack(x_splits))
                y_batches.append(torch.stack(y_splits))
                w_batches.append(torch.tensor(w_splits, dtype=torch.float))

        return x_batches, y_batches, w_batches

class Preprocessor_Ablation2:
    def __init__(self, config_path, pwd_max_len):
        with open(config_path, 'r') as config_file:
            self.config = json.load(config_file)
        self.lstm_seq_len = self.config['lstm_seq_len']
        self.start_token = '\t'  # 定义起始符号
        self.end_token = '\n'    # 定义结束符号
        self.valid_chars = string.ascii_letters + string.digits + string.punctuation + ' ' + self.start_token + self.end_token
        self.char_to_index = {char: idx for idx, char in enumerate(self.valid_chars)}
        self.index_to_char = {idx: char for char, idx in self.char_to_index.items()}
        self.vocab_size = len(self.valid_chars)
        self.alpha = 1.5
        
    def _compute_weight(self, length):
        """
        计算权重，使用公式 f(x) = x^alpha。
        """
        return length ** self.alpha

    def preprocess(self, pwd_list):
        x_batches = []
        y_batches = []
        w_batches = []

        for pwd in pwd_list:
            # 添加起始和结束符号
            pwd = self.start_token + pwd + self.end_token

            # 如果包含无效字符，则跳过此密码
            if any(c not in self.valid_chars for c in pwd):
                continue

            # 切分 x, y, w
            x_splits = []
            y_splits = []
            w_splits = []
            for i in range(1, len(pwd)):
                x_seq = pwd[:i]
                y_seq = pwd[i]
                combined_seq = x_seq + y_seq

                # 处理 x_seq 的长度，并计算权重
                if len(x_seq) > self.lstm_seq_len:
                    x_seq = x_seq[-self.lstm_seq_len:]  # 截断，保留最后部分
                    weight = self._compute_weight(self.lstm_seq_len)
                else:
                    weight = self._compute_weight(len(x_seq))
                    x_seq = (self.start_token * (self.lstm_seq_len - len(x_seq))) + x_seq  # 补全，前端补 start

                # 使用字符索引而不是 one-hot 编码
                x_indices = [self.char_to_index[c] for c in x_seq]
                y_index = self.char_to_index[y_seq]

                x_tensor = torch.tensor(x_indices, dtype=torch.long)
                y_tensor = torch.tensor(y_index, dtype=torch.long)

                x_splits.append(x_tensor)
                y_splits.append(y_tensor)
                w_splits.append(weight)

            # 对 w_splits 进行归一化处理
            if w_splits:
                w_splits = np.array(w_splits, dtype=np.float32)
                w_splits /= w_splits.sum() if w_splits.sum() != 0 else 1
                
                x_batches.append(torch.stack(x_splits))
                y_batches.append(torch.stack(y_splits))
                w_batches.append(torch.tensor(w_splits, dtype=torch.float))

        return x_batches, y_batches, w_batches

class Preprocessor_DPSGD:
    def __init__(self, config_path):
        with open(config_path, 'r') as config_file:
            self.config = json.load(config_file)
        self.lstm_seq_len = self.config['lstm_seq_len']
        self.start_token = '\t'  # 定义起始符号
        self.end_token = '\n'    # 定义结束符号
        self.valid_chars = string.ascii_letters + string.digits + string.punctuation + ' ' + self.start_token + self.end_token
        self.char_to_index = {char: idx for idx, char in enumerate(self.valid_chars)}
        self.index_to_char = {idx: char for char, idx in self.char_to_index.items()}
        self.vocab_size = len(self.valid_chars)

    def preprocess(self, pwd_list):
        x_batches = []
        y_batches = []
        w_batches = []

        for pwd in pwd_list:
            # 添加起始和结束符号
            pwd = self.start_token + pwd + self.end_token

            # 如果包含无效字符，则跳过此密码
            if any(c not in self.valid_chars for c in pwd):
                continue

            # 切分 x, y, w
            x_splits = []
            y_splits = []
            w_splits = []
            for i in range(1, len(pwd)):
                x_seq = pwd[:i]
                y_seq = pwd[i]
                weight = 1

                # 处理 x_seq 的长度，并计算权重
                if len(x_seq) > self.lstm_seq_len:
                    x_seq = x_seq[-self.lstm_seq_len:]  # 截断，保留最后部分
                else:
                    x_seq = (self.start_token * (self.lstm_seq_len - len(x_seq))) + x_seq  # 补全，前端补 start

                # 使用字符索引而不是 one-hot 编码
                x_indices = [self.char_to_index[c] for c in x_seq]
                y_index = self.char_to_index[y_seq]

                x_tensor = torch.tensor(x_indices, dtype=torch.long)
                y_tensor = torch.tensor(y_index, dtype=torch.long)

                x_splits.append(x_tensor)
                y_splits.append(y_tensor)
                w_splits.append(weight)

            # 对 w_splits 进行归一化处理
            if w_splits:                
                x_batches.append(torch.stack(x_splits))
                y_batches.append(torch.stack(y_splits))
                w_batches.append(torch.tensor(w_splits, dtype=torch.float))

        return x_batches, y_batches, w_batches

class PreprocessorDP:
    def __init__(self, config_path, substring_dict, pwd_max_len):
        with open(config_path, 'r') as config_file:
            self.config = json.load(config_file)
        self.lstm_seq_len = self.config['lstm_seq_len']
        self.start_token = '\t'  # 定义起始符号
        self.end_token = '\n'    # 定义结束符号
        self.valid_chars = string.ascii_letters + string.digits + string.punctuation + ' ' + self.start_token + self.end_token
        self.char_to_index = {char: idx for idx, char in enumerate(self.valid_chars)}
        self.index_to_char = {idx: char for char, idx in self.char_to_index.items()}
        self.vocab_size = len(self.valid_chars)
        self.alpha = 1.5
        self.substring_dict = substring_dict
        self.threshold = self.config.get('threshold', 0)
        self.pwd_max_len = pwd_max_len
        self.epsilon_Abovethreshold = self.config.get('epsilon_Abovethreshold', 0.1)
        self.threshold_DP = self.threshold + np.random.laplace(0, 2 * (self.pwd_max_len + 1) / self.epsilon_Abovethreshold, size=1).item()
        
    def _compute_weight(self, length):
        """
        计算权重，使用公式 f(x) = x^alpha。
        """
        return length ** self.alpha

    def preprocess(self, pwd_list):
        x_batches = []
        y_batches = []
        w_batches = []

        for pwd in pwd_list:
            # 添加起始和结束符号
            pwd = self.start_token + pwd + self.end_token

            # 如果包含无效字符，则跳过此密码
            if any(c not in self.valid_chars for c in pwd):
                continue

            # 切分 x, y, w
            x_splits = []
            y_splits = []
            w_splits = []
            for i in range(1, len(pwd)):
                x_seq = pwd[:i]
                y_seq = pwd[i]
                combined_seq = x_seq + y_seq

                # 处理 x_seq 的长度，并计算权重
                if len(x_seq) > self.lstm_seq_len:
                    x_seq = x_seq[-self.lstm_seq_len:]  # 截断，保留最后部分
                    if self.substring_dict[combined_seq] < self.threshold_DP:
                        continue
                    weight = self._compute_weight(self.lstm_seq_len)
                else:
                    if self.substring_dict[combined_seq] < self.threshold_DP:
                        continue
                    weight = self._compute_weight(len(x_seq))
                    x_seq = (self.start_token * (self.lstm_seq_len - len(x_seq))) + x_seq  # 补全，前端补 start

                # 使用字符索引而不是 one-hot 编码
                x_indices = [self.char_to_index[c] for c in x_seq]
                y_index = self.char_to_index[y_seq]

                x_tensor = torch.tensor(x_indices, dtype=torch.long)
                y_tensor = torch.tensor(y_index, dtype=torch.long)

                x_splits.append(x_tensor)
                y_splits.append(y_tensor)
                w_splits.append(weight)

            # 对 w_splits 进行归一化处理
            if w_splits:
                w_splits = np.array(w_splits, dtype=np.float32)
                w_splits /= w_splits.sum() if w_splits.sum() != 0 else 1
                
                x_batches.append(torch.stack(x_splits))
                y_batches.append(torch.stack(y_splits))
                w_batches.append(torch.tensor(w_splits, dtype=torch.float))

        return x_batches, y_batches, w_batches
    
class SubstringDictGenerator:
    def __init__(self, password_dataset, config_path):
        with open(config_path, 'r') as config_file:
            self.config = json.load(config_file)
        self.password_dataset = password_dataset
        self.lstm_seq_len = self.config['lstm_seq_len']
        self.start_token = '\t'
        self.end_token = '\n'
        self.substring_dict = {}
        self.pwd_file = self.config.get('pwd_file', 'passwords.txt')
        self.substring_dict_file = './substring_dict/' + os.path.basename(self.pwd_file).replace('.txt', '') + '_substring_dict.pkl'

    def generate(self):
        # 尝试从文件中读取已存在的 substring_dict
        if os.path.exists(self.substring_dict_file):
            with open(self.substring_dict_file, 'rb') as f:
                self.substring_dict = pickle.load(f)
            print(f"Loaded substring_dict from {self.substring_dict_file}")
        else:
            # 生成 substring_dict
            for pwd in self.password_dataset:
                # 添加起始和结束符号
                pwd = self.start_token + pwd + self.end_token

                # 切分子串并更新频数
                for i in range(1, len(pwd) + 1):
                    substring = pwd[:i]
                    if len(substring) > self.lstm_seq_len:
                        substring = substring[-self.lstm_seq_len:]  # 截断，保留最后部分
                    if substring in self.substring_dict:
                        self.substring_dict[substring] += 1
                    else:
                        self.substring_dict[substring] = 1

            # 保存 substring_dict 到文件
            with open(self.substring_dict_file, 'wb') as f:
                pickle.dump(self.substring_dict, f)
            print(f"Saved substring_dict to {self.substring_dict_file}")

        return self.substring_dict
    
class DPSubstringDictGenerator:
    def __init__(self, substring_dict, config_path):
        with open(config_path, 'r') as config_file:
            self.config = json.load(config_file)
        self.substring_dict = substring_dict
        self.epsilon_Abovethreshold = self.config.get('epsilon_Abovethreshold', 0.1)
        self.pwd_max_len = self.config.get('pwd_max_len', 40)
        self.pwd_file = self.config.get('pwd_file', 'passwords.txt')
        self.DPsubstring_dict_file = './substring_dict/' + os.path.basename(self.pwd_file).replace('.txt', '') + '_DPsubstring_dict.pkl'

    def generate_dp(self):
        b = 2 * (self.pwd_max_len + 1) / self.epsilon_Abovethreshold
        laplace_noise = np.random.laplace(0, 4 * b, size=len(self.substring_dict))

        # 对 substring_dict 的频数添加噪声
        for idx, key in enumerate(self.substring_dict):
            self.substring_dict[key] += laplace_noise[idx]

        # 保存 DP substring_dict 到文件
        # with open(self.DPsubstring_dict_file, 'wb') as f:
        #     pickle.dump(self.substring_dict, f)
        # print(f"Saved DP substring_dict to {self.DPsubstring_dict_file}")

        return self.substring_dict