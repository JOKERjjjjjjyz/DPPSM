import json
import torch
import os
from typing import Dict, Any

# 假设所有自定义模块都位于正确的路径下
from data.dataset import PasswordDataset
from utils.preprocessing import Preprocessor
from train.dpfla_train import DPTrainer
from utils.guessing_gpu import MonteCarloEstimator, test_montecarlo
from models.dpsgd_lstm_model import DPSGDLSTMModel

# ==============================================================================
# 函数接口 1: 训练模型
# ==============================================================================
def train_model(config_path: str, data_path: str):
    """
    初始化 DPTrainer 并开始训练流程。
    模型和检查点将根据配置文件中的设置自动保存。
    """
    print("--- Starting Model Training ---")
    try:
        trainer = DPTrainer(config_path=config_path, data_path=data_path)
        trainer.train()
        print("--- Training Finished ---")
    except Exception as e:
        print(f"An error occurred during training: {e}")

# ==============================================================================
# 函数接口 2: 加载模型
# ==============================================================================
def load_model(config_path: str, checkpoint_path: str) -> DPSGDLSTMModel:
    """
    从配置文件和检查点(.pth)文件加载一个预训练的DPSGD-LSTM模型。

    Args:
        config_path (str): 模型的JSON配置文件路径。
        checkpoint_path (str): 模型的.pth检查点文件路径。

    Returns:
        DPSGDLSTMModel: 加载并设置为评估模式的模型对象。
    """
    print(f"--- Loading Pre-trained Model from {checkpoint_path} ---")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    # 1. 初始化模型结构
    model = DPSGDLSTMModel(config_path=config_path)
    
    # 2. 加载检查点
    # 在加载到GPU或CPU前，先指定map_location，确保设备兼容性
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 3. 处理 state_dict 的键名 (移除分布式训练时可能添加的 '_module.' 前缀)
    new_state_dict = {}
    # 有时checkpoint可能是一个字典，state_dict在 'model_state_dict' 键下
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    for k, v in state_dict.items():
        new_key = k.replace("_module.", "") if "_module." in k else k
        new_state_dict[new_key] = v

    # 4. 加载修正后的 state_dict 并设置为评估模式
    model.load_state_dict(new_state_dict)
    model.to(device) # 将模型移动到正确的设备
    model.eval()
    
    print(f"Model loaded successfully on device: {device}")
    return model

# ==============================================================================
# 辅助函数: 准备推理组件
# ==============================================================================
def setup_inference(model: DPSGDLSTMModel, config_path: str, n_samples: int, estimator_cache_file: str) -> MonteCarloEstimator:
    """
    创建并返回推理所需的核心组件 MonteCarloEstimator。
    这一步开销较大，因为它会生成大量样本。
    """
    print("--- Setting up Inference Components ---")
    preprocessor = Preprocessor(config_path=config_path)
    
    print(f"Initializing MonteCarloEstimator with {n_samples} samples...")
    # 这个过程如果缓存文件不存在，会比较耗时
    estimator = MonteCarloEstimator(
        model=model, 
        preprocessor=preprocessor, 
        n_samples=n_samples, 
        estimator_file=estimator_cache_file
    )
    print("Inference components ready.")
    return estimator

# ==============================================================================
# 函数接口 3: 推理 (批处理 & 单个)
# ==============================================================================
def inference_for_batch(estimator: MonteCarloEstimator, test_file: str, result_file: str):
    """
    【批处理】使用准备好的评估器对整个测试文件进行推理。
    """
    print(f"--- Starting Batch Inference on {test_file} ---")
    test_montecarlo(
        montecarlo_estimator=estimator, 
        test_file=test_file, 
        result_file=result_file
    )
    print(f"Batch inference complete. Results saved to {result_file}")

def inference_for_single_password(estimator: MonteCarloEstimator, password: str) -> Dict[str, Any]:
    """
    【单个】使用准备好的评估器对单个密码进行推理。
    """
    print(f"--- Starting Single Password Inference for '{password}' ---")
    strength = estimator.compute_strength(password)
    avg_strength, prob = estimator.compute_average_strength(password)
    
    return {
        "password": password,
        "strength": strength,
        "average_strength": avg_strength,
        "probability": prob
    }

# ==============================================================================
# 调用示例 (Demos)
# ==============================================================================
if __name__ == '__main__':

    # --- DEMO 1: 训练一个新模型 ---
    def run_training_demo():
        """演示如何调用 train_model 函数"""
        print("\n*** RUNNING TRAINING DEMO ***")
        DATA_PATH = "data/trainset/rockyou320w.txt"  # <-- 修改这里
        CONFIG_PATH = "config/rockyou_dpfla_train.json"  # <-- 修改这里
        train_model(config_path=CONFIG_PATH, data_path=DATA_PATH)

    # --- DEMO 2: 对密码文件进行批处理推理 ---
    def run_batch_inference_demo():
        """演示加载模型 -> 设置推理组件 -> 进行批处理推理的完整流程"""
        print("\n*** RUNNING BATCH INFERENCE DEMO ***")
        CONFIG_PATH = "config/rockyou_dpfla_train.json"  # <-- 修改这里
        CHECKPOINT_PATH = "./models/model/rockyou320w_dpfla_0.1178.pth"  # <-- 修改这里
        ESTIMATOR_CACHE = "./estimators/rockyou320w_dpfla_0.1178_estimator.pkl" # <-- 修改这里
        TEST_FILE = "./data/test/cityday_less.txt"  # <-- 修改这里
        RESULT_FILE = "./results/rockyou320w_cityday_results_dpfla_0.1178.csv" # <-- 修改这里
        
        try:
            # 步骤 1: 加载模型
            model = load_model(config_path=CONFIG_PATH, checkpoint_path=CHECKPOINT_PATH)
            
            # 步骤 2: 准备推理组件 (这一步开销大)
            estimator = setup_inference(
                model=model, 
                config_path=CONFIG_PATH, 
                n_samples=10000, 
                estimator_cache_file=ESTIMATOR_CACHE
            )
            
            # 步骤 3: 执行批处理推理
            inference_for_batch(estimator=estimator, test_file=TEST_FILE, result_file=RESULT_FILE)
            
        except Exception as e:
            print(f"An error occurred during batch inference demo: {e}")

    # --- DEMO 3: 对单个密码进行推理 ---
    def run_single_inference_demo():
        """演示加载模型 -> 设置推理组件 -> 对单个密码进行推理"""
        print("\n*** RUNNING SINGLE PASSWORD INFERENCE DEMO ***")
        CONFIG_PATH = "config/rockyou_dpfla_train.json" # <-- 修改这里
        CHECKPOINT_PATH = "./models/model/rockyou320w_dpfla_0.1178.pth" # <-- 修改这里
        ESTIMATOR_CACHE = "./estimators/rockyou320w_dpfla_0.1178_estimator.pkl" # <-- 修改这里
        PASSWORD_TO_TEST = "abcdefgh" # <-- 修改这里
        
        try:
            # 步骤 1 & 2 同上
            model = load_model(config_path=CONFIG_PATH, checkpoint_path=CHECKPOINT_PATH)
            estimator = setup_inference(
                model=model, 
                config_path=CONFIG_PATH, 
                n_samples=10000, 
                estimator_cache_file=ESTIMATOR_CACHE
            )

            # 步骤 3: 执行单个密码推理
            results = inference_for_single_password(estimator=estimator, password=PASSWORD_TO_TEST)
            
            print("\n--- Inference Result ---")
            for key, value in results.items():
                if isinstance(value, float):
                    print(f"{key.capitalize()}: {value:.4f}")
                else:
                    print(f"{key.capitalize()}: {value}")

        except Exception as e:
            print(f"An error occurred during single inference demo: {e}")

    # --- 指令 ---
    # 根据您的需要，取消下面一行的注释来运行对应的演示。
    
    # run_training_demo()
    # run_batch_inference_demo()
    run_single_inference_demo()