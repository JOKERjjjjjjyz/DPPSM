# from data.dataset import PasswordDataset
# from utils.preprocessing import Preprocessor, SubstringDictGenerator, PreprocessorDP,DPSubstringDictGenerator
# from train.dpsgd_train import DPTrainer
# from utils.guessing import Guesser, MonteCarloEstimator, test_montecarlo

# # 配置文件路径
# # data_path = "data/trainset/rockyou_new.txt"
# data_path = "data/test/rockyou10w.txt"
# config_path = "config/rockyou_dpsgd_train.json"

# # 创建并运行训练器
# trainer = DPTrainer(config_path=config_path, data_path=data_path)
# trainer.train()
# preprocessor = Preprocessor(config_path=config_path)
# # 假设有一个模型和预处理器对象
# model = trainer.model  # 应替换为实际模型
# preprocessor = preprocessor  # 应替换为实际预处理器

# n_samples = 1000
# montecarlo_estimator = MonteCarloEstimator(model=model, preprocessor=preprocessor, n_samples=n_samples, estimator_file='rockyou10w_dpsgd_1.28_estimator.pkl')

# # 示例密码
# test_password = "abcdefgh"

# # 计算密码的强度
# strength = montecarlo_estimator.compute_strength(test_password)
# print(f"Strength of password '{test_password}': {strength}")

# # 计算平均强度
# avg_strength, prob = montecarlo_estimator.compute_average_strength(test_password)
# print(f"Average strength of password '{test_password}': {avg_strength}, Probability: {prob}")

# test_password = 'abcdefghhhhhhhhhh'
# strength = montecarlo_estimator.compute_strength(test_password)
# print(f"Strength of password '{test_password}': {strength}")

# test_file = "./data/test/test.txt"  # 应替换为实际测试文件
# result_file = "./results/rockyou10w_results_dpsgd_1.28.csv"
# test_montecarlo(montecarlo_estimator, test_file)

from data.dataset import PasswordDataset
from utils.preprocessing import Preprocessor, SubstringDictGenerator, PreprocessorDP,DPSubstringDictGenerator
from train.dpsgd_train import DPTrainer
from utils.guessing import Guesser, MonteCarloEstimator, test_montecarlo
import json
from models.dpsgd_lstm_model import DPSGDLSTMModel
import torch


data_path = "data/trainset/rockyou_new.txt"
config_path = "config/rockyou_dpsgd_train.json"

# 创建并运行训练器
trainer = DPTrainer(config_path=config_path, data_path=data_path)
trainer.train()