import time
import pandas as pd
import os
from typing import Tuple

# 内部模块导入 (确保这些模块与此脚本在同一环境下)
# 您需要确保 model.py 和 ngram_dp.py 文件是可用的
from model import PosEstimator
import ngram_dp

# ==============================================================================
# 核心函数接口
# ==============================================================================

def train_model(training_data_path: str, model_save_path: str, n: int, epsilon: float):
    """
    加载训练数据，训练一个新的 n-gram 模型，并将其保存到指定路径。
    """
    print(f"--- Starting Model Training ---")
    print(f"Loading training data from: {training_data_path}")
    try:
        with open(training_data_path, 'rt', encoding='utf-8', errors='ignore') as f:
            train_data = [w.strip('\r\n') for w in f]
    except FileNotFoundError:
        print(f"Error: Training file not found at {training_data_path}")
        return

    print(f"Training a new {n}-gram model with epsilon={epsilon}...")
    start_time = time.time()
    
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    
    ngram_dp.NGramModel(words=train_data, n=n, epsilon=epsilon, shelfname=model_save_path)
    
    print(f"Training complete. Model saved to: {model_save_path}")
    print(f"Total training time: {time.time() - start_time:.2f} seconds.")


def load_model(model_path: str, n: int) -> ngram_dp.NGramModel:
    """
    从指定路径加载一个已经训练好的 n-gram 模型。
    """
    print(f"--- Loading Pre-trained Model ---")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")
        
    print(f"Loading {n}-gram model from: {model_path}")
    model = ngram_dp.NGramModel(words=None, n=n, shelfname=model_path)
    print("Model loaded successfully.")
    return model


def inference_for_batch(model: ngram_dp.NGramModel, test_data_path: str, sample_size: int) -> pd.DataFrame:
    """
    【批处理】使用加载好的模型对整个测试集进行推理，返回DataFrame。
    """
    print(f"--- Starting Batch Inference ---")
    try:
        with open(test_data_path, 'rt', encoding='utf-8', errors='ignore') as f:
            test_data = [w.strip('\r\n') for w in f]
    except FileNotFoundError:
        print(f"Error: Test file not found at {test_data_path}")
        return pd.DataFrame()

    print(f"Generating {sample_size} samples to create estimator (one-time setup)...")
    sample = list(model.sample(sample_size))
    estimator = PosEstimator(sample)
    print("Estimator created.")

    print(f"Evaluating {len(test_data)} passwords...")
    start_eval_time = time.time()
    
    results = []
    for password in test_data:
        log_prob = model.logprob(password)
        estimation = estimator.position(log_prob)
        results.append({
            'name': password,
            'log_prob': log_prob,
            'guess_number': estimation
        })
    
    print(f"Evaluation finished in {time.time() - start_eval_time:.2f} seconds.")
    
    df = pd.DataFrame(results)
    df_sorted = df.sort_values(by='guess_number', ascending=True)
    
    return df_sorted


def inference_for_single_password(
    model: ngram_dp.NGramModel, 
    password: str, 
    sample_size: int = 10000
) -> Tuple[float, float]:
    """
    【单个】对单个密码进行推理，返回其对数概率和猜测数。
    注意：此函数每次调用都会承担创建评估器的开销，可能较慢。

    Args:
        model (ngram_dp.NGramModel): 已加载的模型实例。
        password (str): 需要评估的单个密码。
        sample_size (int): 为 PosEstimator 生成的样本数量。

    Returns:
        Tuple[float, float]: 一个包含 (log_prob, guess_number) 的元组。
    """
    print(f"--- Starting Single Password Inference for '{password}' ---")
    
    # 1. 创建评估器 (这是主要的性能开销)
    print(f"Generating {sample_size} samples to create estimator...")
    start_time = time.time()
    sample = list(model.sample(sample_size))
    estimator = PosEstimator(sample)
    print(f"Estimator created in {time.time() - start_time:.2f} seconds.")
    
    # 2. 计算 log_prob
    log_prob = model.logprob(password)
    
    # 3. 计算 guess_number
    guess_number = estimator.position(log_prob)
    
    return log_prob, guess_number


# ==============================================================================
# 调用示例 (Demos)
# ==============================================================================
if __name__ == "__main__":

    # --- DEMO 1: 训练一个新模型 ---
    def run_training_demo():
        """演示如何调用 train_model 函数"""
        print("\n*** RUNNING TRAINING DEMO ***")
        TRAIN_FILE = "path/to/your/training_passwords.txt" # <-- 修改这里
        MODEL_SAVE_PATH = "./models/4gram_dp_model_eps_2.0.db" # <-- 修改这里
        N_GRAM = 4
        EPSILON = 2.0
        train_model(
            training_data_path=TRAIN_FILE,
            model_save_path=MODEL_SAVE_PATH,
            n=N_GRAM,
            epsilon=EPSILON
        )

    # --- DEMO 2: 对密码文件进行批处理推理 ---
    def run_batch_inference_demo():
        """演示如何加载模型并对整个文件进行推理"""
        print("\n*** RUNNING BATCH INFERENCE DEMO ***")
        MODEL_PATH = "./models/4gram_dp_model_eps_2.0.db" # <-- 修改这里
        TEST_FILE = "path/to/your/test_passwords.txt" # <-- 修改这里
        OUTPUT_CSV_PATH = "./results/inference_results.csv" # <-- 修改这里
        N_GRAM = 4
        SAMPLE_SIZE = 10000

        try:
            my_model = load_model(model_path=MODEL_PATH, n=N_GRAM)
            results_df = inference_for_batch(
                model=my_model,
                test_data_path=TEST_FILE,
                sample_size=SAMPLE_SIZE
            )
            if not results_df.empty:
                os.makedirs(os.path.dirname(OUTPUT_CSV_PATH), exist_ok=True)
                results_df.to_csv(OUTPUT_CSV_PATH, index=False)
                print(f"Batch inference results saved to {OUTPUT_CSV_PATH}")

        except Exception as e:
            print(f"An error occurred during the batch inference demo: {e}")

    # --- DEMO 3: 对单个密码进行完整推理 ---
    def run_single_inference_demo():
        """演示如何加载模型并对单个密码进行推理，获取概率和猜测数"""
        print("\n*** RUNNING SINGLE PASSWORD INFERENCE DEMO ***")
        MODEL_PATH = "./models/4gram_dp_model_eps_2.0.db" # <-- 修改这里
        N_GRAM = 4
        PASSWORD_TO_CHECK = "Tr0ub4dor&3" # <-- 修改这里

        try:
            my_model = load_model(model_path=MODEL_PATH, n=N_GRAM)
            
            # 调用新函数
            log_prob, guess_num = inference_for_single_password(
                model=my_model,
                password=PASSWORD_TO_CHECK,
                sample_size=10000 # 可以按需调整样本大小
            )
            
            print("\n--- Inference Result ---")
            print(f"Password: '{PASSWORD_TO_CHECK}'")
            print(f"Log Probability: {log_prob:.4f}")
            print(f"Estimated Guess Number: {guess_num:.2f}")

        except Exception as e:
            print(f"An error occurred during the single inference demo: {e}")


    # --- 指令 ---
    # 根据您的需要，取消下面一行的注释来运行对应的演示。
    
    # run_training_demo()
    # run_batch_inference_demo()
    run_single_inference_demo()