# utils/guessing.py
import torch
import numpy as np
import torch.nn.functional as F
import csv

class Guesser:
    def __init__(self, model, preprocessor):
        self.model = model
        self.preprocessor = preprocessor

    def guess_one_substring(self, substring, label):
        # 将 substring 转换为索引序列
        x_indices = [self.preprocessor.char_to_index[c] for c in substring]
        x = torch.tensor(x_indices, dtype=torch.long).unsqueeze(0)  # 添加 batch 维度

        # 获取模型输出的概率向量
        with torch.no_grad():
            output = self.model(x)
            probabilities = F.softmax(output, dim=-1)  # 使用 softmax 获得概率分布

        # 获取 label 对应的概率
        label_idx = self.preprocessor.char_to_index[label]
        # print(f"{substring}->{label}:{probabilities[0, label_idx].item()}")
        return probabilities[0, label_idx].item()

    def guess_one_pwd(self, pwd):
        # 添加起始和结束符号
        pwd = self.preprocessor.start_token + pwd + self.preprocessor.end_token
        total_prob = 1.0

        # 对 pwd 进行切分，计算每个子串的概率
        for i in range(1, len(pwd)):
            x_seq = pwd[:i]
            y_label = pwd[i]

            # 处理 x_seq 的长度
            if len(x_seq) > self.preprocessor.lstm_seq_len:
                x_seq = x_seq[-self.preprocessor.lstm_seq_len:]  # 截断，保留最后部分
            else:
                x_seq = (self.preprocessor.start_token * (self.preprocessor.lstm_seq_len - len(x_seq))) + x_seq  # 补全，前端补 start

            # 获取概率并更新 total_prob
            prob = self.guess_one_substring(x_seq, y_label)
            total_prob *= prob

        return total_prob
    
import bisect
import os
import pickle
import torch
import torch.nn.functional as F
import random
import numpy as np

class MonteCarloEstimator:
    def __init__(self, model, preprocessor, n_samples, estimator_file, config={}):
        self.config = config
        self.model = model
        self.preprocessor = preprocessor
        self.n_samples = n_samples
        self.estimator_file = estimator_file
        self.guesser = Guesser(self.model, self.preprocessor)
        self.estimators = []  # 用于存储五套 A 和 C

        # 尝试从文件加载已有的五套 A 和 C
        model_name = self._get_model_name()
        # self.estimator_file = f"{model_name}_estimator.pkl"
        if os.path.exists(self.estimator_file):
            with open(self.estimator_file, 'rb') as f:
                self.estimators = pickle.load(f)
                print(f"Loaded precomputed estimator from {self.estimator_file}")
        else:
            print(f"No precomputed estimator found. It will be generated and saved as {self.estimator_file}")
            self.sample_probabilities()
            self._save_estimator()

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
    
def test_montecarlo(montecarlo_estimator, test_file, result_file):
    with open(test_file, 'r') as f:
        passwords = [line.strip() for line in f]

    results = []
    for pwd in passwords:
        avg_strength, prob = montecarlo_estimator.compute_average_strength(pwd)
        log_prob = -np.log2(prob)
        results.append((pwd, log_prob, avg_strength))

    # 保存结果到 CSV 文件
    output_file = result_file
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Password', 'Log_Prob', 'Strength'])
        writer.writerows(results)
    print(f"Results saved to {output_file}")