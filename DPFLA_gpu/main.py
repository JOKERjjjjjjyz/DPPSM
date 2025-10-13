from utils.guessing import MonteCarloEstimator, test_montecarlo
from data.dataset import PasswordDataset
from utils.preprocessing import Preprocessor
from train.train import Trainer
from utils.guessing import Guesser
from models.lstm_model import LSTMModel
import torch
import json

data_path = "data/test/password_test3.txt"
config_path = "config/config.json"
with open(config_path, 'r') as config_file:
    config = json.load(config_file)
# 创建并运行训练器
# trainer = Trainer(config_path=config_path, data_path=data_path)
# trainer.train()

model = LSTMModel(config_path=config_path)
model.load_state_dict(torch.load(f"{config.get('pwd_file', './model/rockyou10w_40.pth')}"))
model.eval()
preprocessor = Preprocessor(config_path=config_path)
# 假设有一个模型和预处理器对象
model = model  # 应替换为实际模型
preprocessor = preprocessor  # 应替换为实际预处理器

n_samples = 1000
montecarlo_estimator = MonteCarloEstimator(model, preprocessor, n_samples)

# 示例密码
test_password = "abcdefgh"

# 计算密码的强度
strength = montecarlo_estimator.compute_strength(test_password)
print(f"Strength of password '{test_password}': {strength}")

# 计算平均强度
avg_strength, prob = montecarlo_estimator.compute_average_strength(test_password)
print(f"Average strength of password '{test_password}': {avg_strength}, Probability: {prob}")

test_password = 'abcdefghhhhhhhhhh'
strength = montecarlo_estimator.compute_strength(test_password)
print(f"Strength of password '{test_password}': {strength}")

test_file = "./data/test/test.txt"  # 应替换为实际测试文件
test_montecarlo(montecarlo_estimator, test_file)
