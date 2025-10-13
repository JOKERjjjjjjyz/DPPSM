from data.dataset import PasswordDataset
from utils.preprocessing import Preprocessor, SubstringDictGenerator, PreprocessorDP,DPSubstringDictGenerator
from train.ablation2 import DPTrainer
from utils.guessing import Guesser, MonteCarloEstimator, test_montecarlo
import json
from models.dpsgd_lstm_model import DPSGDLSTMModel
import torch
# 配置文件路径
# data_path = "data/trainset/rockyou_new.txt"
data_path = "data/trainset/rockyou_new.txt"
config_path = "config/rockyou_abla2_train.json"

# 创建并运行训练器
trainer = DPTrainer(config_path=config_path, data_path=data_path)
trainer.train()