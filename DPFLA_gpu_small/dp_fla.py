from data.dataset import PasswordDataset
from utils.preprocessing import Preprocessor, SubstringDictGenerator, PreprocessorDP,DPSubstringDictGenerator
from train.dpfla_train import DPTrainer
from utils.guessing_gpu import Guesser, MonteCarloEstimator, test_montecarlo
import json
from models.dpsgd_lstm_model import DPSGDLSTMModel
import torch
# 配置文件路径
# data_path = "data/trainset/rockyou_new.txt"
# data_path = "data/trainset/rockyou320w.txt"
# config_path = "config/rockyou_dpfla_train.json"

# # 创建并运行训练器
# trainer = DPTrainer(config_path=config_path, data_path=data_path)
# trainer.train()

data_path = "data/trainset/rockyou320w.txt"
config_path = "config/rockyou_dpfla_train.json"
with open(config_path, 'r') as config_file:
    config = json.load(config_file)

model = DPSGDLSTMModel(config_path=config_path)
checkpoint = torch.load("./models/model/rockyou320w_dpfla_0.1178.pth")

# 处理 state_dict 的键名
new_state_dict = {}
for k, v in checkpoint.items():
    new_key = k.replace("_module.", "") if "_module." in k else k
    new_state_dict[new_key] = v

# 加载修正后的 state_dict
model.load_state_dict(new_state_dict)
model.eval()

preprocessor = Preprocessor(config_path=config_path)
# 假设有一个模型和预处理器对象
# model = trainer.model  # 应替换为实际模型
preprocessor = preprocessor  # 应替换为实际预处理器

n_samples = 10000
montecarlo_estimator = MonteCarloEstimator(model=model, preprocessor=preprocessor, n_samples=n_samples, estimator_file='rockyou320w_dpfla_0.1178_estimator.pkl')

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

test_file = "./data/test/cityday_less.txt"  # 应替换为实际测试文件
result_file = "./results/rockyou320w_cityday_results_dpfla_0.1178.csv"
test_montecarlo(montecarlo_estimator=montecarlo_estimator, test_file=test_file, result_file=result_file)