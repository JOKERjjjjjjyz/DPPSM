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
# 函数接口 1: 训练模型 (保持不变)
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
# 函数接口 2: 加载模型 (保持不变)
# ==============================================================================
def load_model(config_path: str) -> DPSGDLSTMModel:
    """
    【修改】从配置文件中 'model_file' 字段指定的路径加载一个预训练的DPSGD-LSTM模型。
    现在不再需要手动传入 checkpoint_path。
    """
    print(f"--- Loading Pre-trained Model based on '{config_path}' ---")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # 1. 从 config 文件读取 model_file 路径
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    model_path = config.get('model_file')
    if not model_path or not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at path specified by 'model_file' in config: {model_path}")
    
    print(f"INFO: Found model path in config: '{model_path}'")

    # 2. 加载模型
    model = DPSGDLSTMModel(config_path=config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device)

    state_dict = checkpoint.get('model_state_dict', checkpoint)
    new_state_dict = {k.replace("_module.", ""): v for k, v in state_dict.items()}

    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    
    print(f"Model loaded successfully on device: {device}")
    return model

# ==============================================================================
# 辅助函数: 准备推理组件 (保持不变)
# ==============================================================================
def setup_inference(model: DPSGDLSTMModel, config_path: str, n_samples: int, estimator_cache_file: str) -> MonteCarloEstimator:
    """
    创建并返回推理所需的核心组件 MonteCarloEstimator。
    """
    print("--- Setting up Inference Components ---")
    preprocessor = Preprocessor(config_path=config_path)
    
    print(f"Initializing MonteCarloEstimator with {n_samples} samples...")
    estimator = MonteCarloEstimator(
        model=model, 
        preprocessor=preprocessor, 
        n_samples=n_samples, 
        estimator_file=estimator_cache_file
    )
    print("Inference components ready.")
    return estimator

# ==============================================================================
# 函数接口 3: 推理 (批处理 & 单个) (保持不变)
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

    # --- DEMO 1: 训练一个新模型 (逻辑正确, 保持不变) ---
    def run_training_demo():
        print("\n*** RUNNING TRAINING DEMO ***")
        CONFIG_PATH = "config/rockyou_dpfla_train.json"
        try:
            with open(CONFIG_PATH, 'r') as f:
                config = json.load(f)
            DATA_PATH = config.get('pwd_file')
            if not DATA_PATH:
                raise ValueError("'pwd_file' key not found in the config file.")
            print(f"Found training data path in config: '{DATA_PATH}'")
            train_model(config_path=CONFIG_PATH, data_path=DATA_PATH)
        except Exception as e:
            print(f"❌ An error occurred during training demo: {e}")

    # --- DEMO 2: 对密码文件进行批处理推理 (*** 已修改 ***) ---
    def run_batch_inference_demo():
        print("\n*** RUNNING BATCH INFERENCE DEMO ***")
        CONFIG_PATH = "config/rockyou_dpfla_train.json"
        try:
            # 1. 从config加载所有必需的路径
            with open(CONFIG_PATH, 'r') as f:
                config = json.load(f)
            
            ESTIMATOR_CACHE = config.get('estimator_cache_file')
            TEST_FILE = config.get('test_file')
            RESULT_FILE = config.get('result_file')
            N_SAMPLES = config.get('n_samples', 10000) # 从config读取或使用默认值

            if not all([TEST_FILE, RESULT_FILE]):
                raise ValueError("'test_file' and 'result_file' must be defined in the config.")
            
            # 2. 调用修改后的 load_model 函数，它会自动读取 model_file
            model = load_model(config_path=CONFIG_PATH)
            
            # 3. 正常设置和运行推理
            estimator = setup_inference(
                model=model, 
                config_path=CONFIG_PATH, 
                n_samples=N_SAMPLES, 
                estimator_cache_file=ESTIMATOR_CACHE
            )
            inference_for_batch(estimator=estimator, test_file=TEST_FILE, result_file=RESULT_FILE)
        except Exception as e:
            print(f"❌ An error occurred during batch inference demo: {e}")

    # --- DEMO 3: 对单个密码进行推理 (*** 已修改 ***) ---
    def run_single_inference_demo():
        print("\n*** RUNNING SINGLE PASSWORD INFERENCE DEMO ***")
        CONFIG_PATH = "config/rockyou_dpfla_train.json"
        PASSWORD_TO_TEST = "abcdefgh"
        
        try:
            # 1. 从config加载所需路径
            with open(CONFIG_PATH, 'r') as f:
                config = json.load(f)

            ESTIMATOR_CACHE = config.get('estimator_cache_file')
            N_SAMPLES = config.get('n_samples', 10000)

            # 2. 调用修改后的 load_model
            model = load_model(config_path=CONFIG_PATH)

            # 3. 正常设置和运行推理
            estimator = setup_inference(
                model=model, 
                config_path=CONFIG_PATH, 
                n_samples=N_SAMPLES, 
                estimator_cache_file=ESTIMATOR_CACHE
            )
            results = inference_for_single_password(estimator=estimator, password=PASSWORD_TO_TEST)
            
            print("\n--- Inference Result ---")
            for key, value in results.items():
                if isinstance(value, float):
                    print(f"{key.capitalize()}: {value:.4f}")
                else:
                    print(f"{key.capitalize()}: {value}")
        except Exception as e:
            print(f"❌ An error occurred during single inference demo: {e}")

    # --- 指令 ---
    # 根据您的需要，取消下面一行的注释来运行对应的演示。
    
    # run_training_demo()
    run_batch_inference_demo()
    # run_single_inference_demo()