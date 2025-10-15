# backoff_inference.py (Corrected Version)
import pandas as pd
import os
import time
from typing import Tuple

# Assume all custom modules are in the correct path
import backoff
import model # Assuming model.py contains PosEstimator or it is a method of the model object

# ==============================================================================
# ## Function 1: Train Model
# ==============================================================================
def train_model(training_data_path: str, model_save_path: str, threshold: int):
    """
    Loads training data, trains a new BackoffModel, and saves it to the specified path.
    """
    print("--- 🚀 Starting Model Training ---")
    try:
        with open(training_data_path, 'rt', encoding='utf-8', errors='ignore') as f:
            train_data = [w.strip('\r\n') for w in f]
    except FileNotFoundError:
        print(f"❌ Error: Training file not found at {training_data_path}")
        return

    print(f"Training a new BackoffModel with threshold={threshold}...")
    start_time = time.time()

    # 确保保存路径的目录存在
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    
    # 【修正第1处】: 在创建模型时通过 shelfname 参数传入保存路径
    model = backoff.BackoffModel(train_data, threshold=threshold, shelfname=model_save_path)
    
    try:
        # 【修正第2处】: 调用 save_to_shelf() 时不再传入任何参数
        if hasattr(model, 'save_to_shelf'):
             model.save_to_shelf()
        print(f"✅ Training complete. Model saved to: {model_save_path}")
    except Exception as e:
        print(f"⚠️ Warning: Could not save the model. Error: {e}")

    print(f"Total training time: {time.time() - start_time:.2f} seconds.")

# ==============================================================================
# ## Function 2: Load Model (No changes needed)
# ==============================================================================
def load_model(model_path: str, threshold: int) -> backoff.BackoffModel:
    """
    Loads a pre-trained BackoffModel from the specified path.
    """
    print(f"--- 📥 Loading Pre-trained Model from {model_path} ---")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    
    model = backoff.BackoffModel.get_from_shelf(model_path, threshold=threshold)
    print("✅ Model loaded successfully.")
    return model

# ==============================================================================
# ## Function 3: Inference (Batch & Single) (No changes needed)
# ==============================================================================
def inference_for_batch(model: backoff.BackoffModel, test_data_path: str, sample_size: int) -> pd.DataFrame:
    """
    **[Batch]** Runs inference on an entire test file using the loaded model.
    """
    print(f"--- 📊 Starting Batch Inference on {test_data_path} ---")
    try:
        with open(test_data_path, 'rt', encoding='utf-8', errors='ignore') as f:
            test_data = [w.strip('\r\n') for w in f]
    except FileNotFoundError:
        print(f"❌ Error: Test file not found at {test_data_path}")
        return pd.DataFrame()

    print(f"Generating {sample_size} samples to create estimator (one-time setup)...")
    sample = list(model.sample(sample_size))
    estimator = model.PosEstimator(sample)
    print("Estimator created.")

    print(f"Evaluating {len(test_data)} passwords...")
    start_eval_time = time.time()
    
    results = []
    processed_passwords = set()
    for password in test_data:
        if password in processed_passwords:
            continue
        log_prob = model.logprob(password)
        estimation = estimator.position(log_prob)
        results.append({
            'name': password,
            'log_prob': log_prob,
            'guess_number': estimation
        })
        processed_passwords.add(password)
    
    print(f"Evaluation finished in {time.time() - start_eval_time:.2f} seconds.")
    
    df = pd.DataFrame(results)
    df_sorted = df.sort_values(by='guess_number', ascending=True)
    
    return df_sorted

def inference_for_single_password(
    model: backoff.BackoffModel, 
    password: str, 
    sample_size: int = 10000
) -> Tuple[float, float]:
    """
    **[Single]** Runs inference on a single password, returning its log probability and guess number.
    """
    print(f"--- 🎯 Starting Single Password Inference for '{password}' ---")
    
    print(f"Generating {sample_size} samples to create estimator...")
    start_time = time.time()
    sample = list(model.sample(sample_size))
    estimator = model.PosEstimator(sample)
    print(f"Estimator created in {time.time() - start_time:.2f} seconds.")
    
    log_prob = model.logprob(password)
    guess_number = estimator.position(log_prob)
    
    return log_prob, guess_number

# ==============================================================================
# ## Demos (No changes needed)
# ==============================================================================
if __name__ == '__main__':

    def run_training_demo():
        """Demonstrates how to call the train_model function."""
        print("\n*** ▶️ RUNNING TRAINING DEMO ***")
        TRAIN_FILE = "dataset/csdn/csdnn_new.txt"
        MODEL_SAVE_PATH = "./models/backoff_model.db"
        THRESHOLD = 10
        train_model(
            training_data_path=TRAIN_FILE,
            model_save_path=MODEL_SAVE_PATH,
            threshold=THRESHOLD
        )

    def run_batch_inference_demo():
        """Demonstrates the full flow: load -> batch inference."""
        print("\n*** ▶️ RUNNING BATCH INFERENCE DEMO ***")
        MODEL_PATH = "./models/backoff_model.db"
        TEST_FILE = "path/to/your/test_passwords.txt"
        OUTPUT_CSV_PATH = "./results/backoff_inference_results.csv"
        THRESHOLD = 10
        SAMPLE_SIZE = 10000
        
        try:
            model = load_model(model_path=MODEL_PATH, threshold=THRESHOLD)
            results_df = inference_for_batch(
                model=model,
                test_data_path=TEST_FILE,
                sample_size=SAMPLE_SIZE
            )
            if not results_df.empty:
                os.makedirs(os.path.dirname(OUTPUT_CSV_PATH), exist_ok=True)
                results_df.to_csv(OUTPUT_CSV_PATH, index=False)
                print(f"✅ Batch inference results saved to {OUTPUT_CSV_PATH}")

        except Exception as e:
            print(f"❌ An error occurred during batch inference demo: {e}")

    def run_single_inference_demo():
        """Demonstrates the full flow: load -> single password inference."""
        print("\n*** ▶️ RUNNING SINGLE PASSWORD INFERENCE DEMO ***")
        MODEL_PATH = "./models/backoff_model.db"
        PASSWORD_TO_TEST = "123456"
        THRESHOLD = 10
        
        try:
            model = load_model(model_path=MODEL_PATH, threshold=THRESHOLD)
            log_prob, guess_num = inference_for_single_password(
                model=model,
                password=PASSWORD_TO_TEST,
                sample_size=10000
            )
            
            print("\n--- 📝 Inference Result ---")
            print(f"Password: '{PASSWORD_TO_TEST}'")
            print(f"Log Probability: {log_prob:.4f}")
            print(f"Estimated Guess Number: {guess_num:.2f}")

        except Exception as e:
            print(f"❌ An error occurred during single inference demo: {e}")
    
    # --- Instructions ---
    # Uncomment the function you want to run.
    
    run_training_demo()
    # run_batch_inference_demo()
    # run_single_inference_demo()