# utils/guessing.py
import torch
import numpy as np
import torch.nn.functional as F
import csv

class Guesser:
    def __init__(self, model, preprocessor):
        self.device = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)  # 将模型迁移到指定设备
        self.preprocessor = preprocessor
          # 设置设备

    def guess_one_substring(self, substring, label):
        # 将 substring 转换为索引序列，并移动到设备上
        x_indices = [self.preprocessor.char_to_index[c] for c in substring]
        x = torch.tensor(x_indices, dtype=torch.long).unsqueeze(0).to(self.device)

        # 获取模型输出的概率向量
        with torch.no_grad():
            output = self.model(x)
            probabilities = F.softmax(output, dim=-1)

        # 获取 label 对应的概率
        label_idx = self.preprocessor.char_to_index[label]
        return probabilities[0, label_idx].item()


    def guess_one_pwd(self, pwd):
        # 添加起始和结束符号
        pwd = self.preprocessor.start_token + pwd + self.preprocessor.end_token

        # 生成所有子串和对应标签
        substrings = []
        labels = []
        for i in range(1, len(pwd)):
            x_seq = pwd[:i]
            y_label = pwd[i]

            # 处理 x_seq 的长度
            if len(x_seq) > self.preprocessor.lstm_seq_len:
                x_seq = x_seq[-self.preprocessor.lstm_seq_len:]
            else:
                x_seq = (self.preprocessor.start_token * (self.preprocessor.lstm_seq_len - len(x_seq))) + x_seq

            substrings.append(x_seq)
            labels.append(y_label)

        # 转换为索引并批量处理
        x_indices = [[self.preprocessor.char_to_index[c] for c in substring] for substring in substrings]
        x_tensor = torch.tensor(x_indices, dtype=torch.long).to(self.device)
        y_indices = torch.tensor([self.preprocessor.char_to_index[label] for label in labels], dtype=torch.long).to(self.device)

        with torch.no_grad():
            outputs = self.model(x_tensor)
            probabilities = F.softmax(outputs, dim=-1)
            selected_probs = probabilities.gather(1, y_indices.unsqueeze(1)).squeeze(1)

        total_prob = selected_probs.prod().item()
        return total_prob
    
    def guess_one_pwd_batch(self, passwords):
        # Batch process multiple passwords
        all_substrings = []
        all_labels = []
        for pwd in passwords:
            pwd = self.preprocessor.start_token + pwd + self.preprocessor.end_token
            for i in range(1, len(pwd)):
                x_seq = pwd[:i]
                y_label = pwd[i]
                if len(x_seq) > self.preprocessor.lstm_seq_len:
                    x_seq = x_seq[-self.preprocessor.lstm_seq_len:]
                else:
                    x_seq = (self.preprocessor.start_token * (self.preprocessor.lstm_seq_len - len(x_seq))) + x_seq
                all_substrings.append(x_seq)
                all_labels.append(y_label)
        
        # Convert to tensors
        x_indices = [[self.preprocessor.char_to_index[c] for c in substring] for substring in all_substrings]
        x_tensor = torch.tensor(x_indices, dtype=torch.long).to(self.device)
        y_indices = torch.tensor([self.preprocessor.char_to_index[label] for label in all_labels], dtype=torch.long).to(self.device)
        
        # Model inference
        with torch.no_grad():
            outputs = self.model(x_tensor)
            probabilities = F.softmax(outputs, dim=-1)
            selected_probs = probabilities.gather(1, y_indices.unsqueeze(1)).squeeze(1)
        
        # Reshape to (num_passwords, num_substrings)
        num_passwords = len(passwords)
        num_substrings = len(all_substrings) // num_passwords
        selected_probs = selected_probs.view(num_passwords, num_substrings)
        
        # Calculate total probabilities
        total_probs = selected_probs.prod(dim=1).cpu().numpy()
        log_probs = -np.log2(total_probs)
        
        return total_probs, log_probs

    
import bisect
import os
import pickle
import torch
import torch.nn.functional as F
import random
import numpy as np

class MonteCarloEstimator:
    def __init__(self, model, preprocessor, n_samples, estimator_file, config={}):
        self.device = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')
        self.config = config
        self.model = model.to(self.device)
        self.preprocessor = preprocessor
        self.n_samples = n_samples
        self.estimator_file = estimator_file
        self.guesser = Guesser(self.model, self.preprocessor)
        self.estimators = []
        self.A_tensors = []
        self.C_tensors = []
        print(f"Using device: {self.device}")

        if os.path.exists(self.estimator_file):
            with open(self.estimator_file, 'rb') as f:
                self.estimators = pickle.load(f)
                print(f"Loaded precomputed estimator from {self.estimator_file}")
        else:
            print(f"No precomputed estimator found. It will be generated and saved as {self.estimator_file}")
            self.sample_probabilities()
            self._save_estimator()

        # Preload A and C as tensors on GPU
        self.A_tensors = torch.stack([torch.tensor(estimator['A'], device=self.device) for estimator in self.estimators], dim=0)
        self.C_tensors = torch.stack([torch.tensor(estimator['C'], device=self.device) for estimator in self.estimators], dim=0)



    def _get_model_name(self):
        # 获取模型名称（去掉 .pth 后缀）
        if hasattr(self.model, 'model_save_path'):
            model_name = os.path.basename(self.model.model_save_path).replace('.pth', '')
        else:
            model_name = "model_40"
        return model_name

    def sample_probabilities(self):
        # 生成五套 A 和 C
        for _ in range(self.config.get('estimation_number', 5)):
            sampled_passwords, log_probabilities = self.generate()
            probabilities = [np.exp(-log_prob) for log_prob in log_probabilities]

            # 根据概率对样本进行排序（升序）
            sorted_probs = sorted(probabilities)
            A = sorted_probs

            # 计算数组 C 的值（逆向累积和）
            C = [0.0] * self.n_samples  # 初始化 C 数组
            cumulative_rank = 0.0
            for i in reversed(range(self.n_samples)):
                cumulative_rank += 1 / (self.n_samples * A[i])
                C[i] = cumulative_rank

            self.estimators.append({'A': A, 'C': C})

    def _save_estimator(self):
        # 保存五套 A 和 C 到文件
        with open(self.estimator_file, 'wb') as f:
            pickle.dump(self.estimators, f)

    def generate(self, end_token='\n', end_threshold=0.05, max_decay_steps=10, max_len=None):
        if max_len is None:
            max_len = self.config.get('max_len', 40)
        # 使用 weighted random walk 生成密码
        start_seq = [self.preprocessor.char_to_index['\t']] * self.preprocessor.lstm_seq_len  # 起始符号序列
        sequences = [start_seq[:] for _ in range(self.n_samples)]  # 初始化 n_samples 个起始序列
        complete_sequences = [[] for _ in range(self.n_samples)]
        finished = torch.zeros(self.n_samples, dtype=torch.bool)
        decay_counter = torch.zeros(self.n_samples, dtype=torch.int)
        log_probabilities = torch.zeros(self.n_samples, dtype=torch.float)

        generated_count = 0
        device = next(self.model.parameters()).device  # 获取模型设备

        while generated_count < self.n_samples:
            # 确保所有序列的长度一致
            for i in range(len(sequences)):
                if len(sequences[i]) < self.preprocessor.lstm_seq_len:
                    padding = [self.preprocessor.char_to_index['	']] * (self.preprocessor.lstm_seq_len - len(sequences[i]))
                    sequences[i] = padding + sequences[i]
            inputs = torch.tensor([seq[-self.preprocessor.lstm_seq_len:] for seq in sequences], dtype=torch.long).to(device)
            with torch.no_grad():
                logits = self.model(inputs)  # 获取模型输出的 logits
            probabilities = F.softmax(logits, dim=-1)  # 计算概率分布

            for i in range(self.n_samples):
                if finished[i]:
                    continue

                # 增加 `end` 的概率以便更快停止
                if decay_counter[i] >= max_decay_steps:
                    logits[i][self.preprocessor.char_to_index[end_token]] += decay_counter[i] * 0.1
                    probabilities[i] = F.softmax(logits[i], dim=-1)  # 更新概率分布

            cdfs = torch.cumsum(probabilities, dim=-1).cpu().numpy()  # 计算累积分布函数
            random_values = np.random.uniform(0, 1, size=self.n_samples)

            for i in range(self.n_samples):
                if finished[i]:
                    continue

                cdf = cdfs[i]
                random_value = random_values[i]
                idx = bisect.bisect_right(cdf, random_value)
                next_char = idx

                if next_char == self.preprocessor.char_to_index[end_token] and probabilities[i][idx] >= end_threshold:
                    finished[i] = True
                    generated_count += 1
                else:
                    sequences[i].append(next_char)
                    complete_sequences[i].append(next_char)
                    log_probabilities[i] -= np.log(probabilities[i][idx].item())
                    decay_counter[i] += 1

                # 如果达到最大长度，删除最后 20 个字符继续生长
                if len(sequences[i]) > max_len:
                    sequences[i] = sequences[i][-max_len + 20:]

        # 将索引序列转换回字符
        generated_passwords = [''.join([self.preprocessor.index_to_char[idx] for idx in seq]) for seq in complete_sequences]
        return generated_passwords, log_probabilities.cpu().numpy()

    def compute_strength(self, pwd):
        # 使用第一套 A 和 C 来计算强度
        estimator = self.estimators[0]
        prob = self._compute_probability(pwd)
        # 使用 bisect 找到第一个 A[j] > prob 的索引 j
        j = bisect.bisect_right(estimator['A'], prob, lo=0, hi=len(estimator['A']))
        if j >= self.n_samples:
            return 1  # 所有 A[i] <= prob，强度为 1
        return estimator['C'][j]

    def compute_average_strength(self, pwd):
        # 使用 estimation_number 套 A 和 C 计算密码强度，返回平均值
        strengths = []
        prob = self._compute_probability(pwd)
        for estimator in self.estimators:
            # 使用 bisect 找到第一个 A[j] > prob 的索引 j
            j = bisect.bisect_right(estimator['A'], prob, lo=0, hi=len(estimator['A']))
            if j >= self.n_samples:
                strengths.append(1)  # 所有 A[i] <= prob，强度为 1
            else:
                strengths.append(estimator['C'][j])
        return sum(strengths) / len(strengths), prob

    def _compute_probability(self, pwd):
        # 使用 Guesser 类来计算给定密码的概率
        return self.guesser.guess_one_pwd(pwd)

    def compute_average_strength_batch(self, passwords):
        # Step 1: Generate all substrings and labels for the entire batch
        all_substrings = []
        all_labels = []
        substrings_per_password = []
        for pwd in passwords:
            pwd = self.preprocessor.start_token + pwd + self.preprocessor.end_token
            substrings = []
            labels = []
            for i in range(1, len(pwd)):
                x_seq = pwd[:i]
                y_label = pwd[i]
                if len(x_seq) > self.preprocessor.lstm_seq_len:
                    x_seq = x_seq[-self.preprocessor.lstm_seq_len:]
                else:
                    x_seq = (self.preprocessor.start_token * (self.preprocessor.lstm_seq_len - len(x_seq))) + x_seq
                substrings.append(x_seq)
                labels.append(y_label)
            substrings_per_password.append(len(substrings))
            all_substrings.extend(substrings)
            all_labels.extend(labels)

        # Convert all substrings and labels to tensors
        x_indices = [[self.preprocessor.char_to_index[c] for c in substring] for substring in all_substrings]
        x_tensor = torch.tensor(x_indices, dtype=torch.long).to(self.device)
        y_indices = torch.tensor([self.preprocessor.char_to_index[label] for label in all_labels], dtype=torch.long).to(self.device)

        # Step 2: Batch model inference
        with torch.no_grad():
            outputs = self.model(x_tensor)
            probabilities = F.softmax(outputs, dim=-1)
            selected_probs = probabilities.gather(1, y_indices.unsqueeze(1)).squeeze(1)  # shape: (total_substrings,)

        # Step 3: Calculate log probabilities per password using PyTorch operations
        selected_probs_cpu = selected_probs.cpu().numpy()
        log_probs = np.zeros(len(passwords))
        start = 0
        for i, count in enumerate(substrings_per_password):
            probs = selected_probs_cpu[start:start+count]
            log_probs[i] = np.sum(np.log(probs))
            start += count
        total_probs = np.exp(log_probs)

        # Step 4: Batch searchsorted for strength calculation using preloaded A and C tensors
        probs_tensor = torch.tensor(total_probs, device=self.device).unsqueeze(0).expand(self.A_tensors.size(0), -1)  # shape: (num_estimators, batch_size)
        indices = torch.searchsorted(self.A_tensors, probs_tensor)  # shape: (num_estimators, batch_size)
        indices = torch.clamp(indices, max=self.n_samples - 1)

        selected_C = torch.gather(self.C_tensors, 1, indices)  # shape: (num_estimators, batch_size)

        # Calculate average strengths
        avg_strengths = selected_C.mean(dim=0).cpu().numpy()  # shape: (batch_size,)

        # Calculate log probabilities
        log_probs_final = -np.log2(total_probs)  # shape: (batch_size,)

        return avg_strengths.tolist(), log_probs_final.tolist()

from tqdm import tqdm  # 导入 tqdm 库

def test_montecarlo(montecarlo_estimator, test_file, result_file):
    with open(test_file, 'r') as f:
        passwords = [line.strip() for line in f]

    results = []
    batch_size = 1024  # 设置合理的批大小，避免显存超限

    # 使用 tqdm 展示进度条，设置总迭代次数为总批次数
    total_batches = (len(passwords) + batch_size - 1) // batch_size  # 向上取整
    with tqdm(total=total_batches, desc="Processing Passwords", unit="batch") as pbar:
        for i in range(0, len(passwords), batch_size):
            batch = passwords[i:i + batch_size]
            avg_strengths, log_probs = montecarlo_estimator.compute_average_strength_batch(batch)

            for pwd, avg_strength, log_prob in zip(batch, avg_strengths, log_probs):
                results.append((pwd, log_prob, avg_strength))

            # 更新进度条
            pbar.update(1)

    # 保存结果到 CSV 文件
    with open(result_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Password', 'Log_Prob', 'Strength'])
        writer.writerows(results)
    print(f"Results saved to {result_file}")

