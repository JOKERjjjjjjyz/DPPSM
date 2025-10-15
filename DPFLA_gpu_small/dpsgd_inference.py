import json
import torch
import os
from typing import Dict, Any

# Assume all custom modules are in the correct path
from data.dataset import PasswordDataset
from utils.preprocessing import Preprocessor
from train.dpsgd_train import DPTrainer
from utils.guessing import Guesser, MonteCarloEstimator, test_montecarlo
from models.dpsgd_lstm_model import DPSGDLSTMModel

# ==============================================================================
# ## Function 1: Train Model
# ==============================================================================
def train_model(config_path: str, data_path: str):
    """
    Initializes the DPTrainer and starts the training process.
    The model checkpoint is saved automatically based on settings in the config file.
    """
    print("--- 🚀 Starting Model Training ---")
    try:
        trainer = DPTrainer(config_path=config_path, data_path=data_path)
        trainer.train()
        print("--- ✅ Training Finished ---")
    except Exception as e:
        print(f"❌ An error occurred during training: {e}")

# ==============================================================================
# ## Function 2: Load Model
# ==============================================================================
def load_model(config_path: str, checkpoint_path: str) -> DPSGDLSTMModel:
    """
    Loads a pre-trained DPSGD-LSTM model from a config and checkpoint (.pth) file.

    Args:
        config_path (str): The model's JSON configuration file path.
        checkpoint_path (str): The model's .pth checkpoint file path.

    Returns:
        DPSGDLSTMModel: The loaded model object, set to evaluation mode.
    """
    print(f"--- 📥 Loading Pre-trained Model from {checkpoint_path} ---")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    # 1. Initialize model structure
    model = DPSGDLSTMModel(config_path=config_path)
    
    # 2. Load checkpoint, ensuring device compatibility
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 3. Correct state_dict keys (removes '_module.' prefix from distributed training)
    new_state_dict = {}
    state_dict = checkpoint.get('model_state_dict', checkpoint) # Handle checkpoints that are dicts
    for k, v in state_dict.items():
        new_key = k.replace("_module.", "") if "_module." in k else k
        new_state_dict[new_key] = v

    # 4. Load the corrected state_dict and set to evaluation mode
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    
    print(f"✅ Model loaded successfully on device: {device}")
    return model

# ==============================================================================
# ## Helper Function: Setup Inference
# ==============================================================================
def setup_inference(model: DPSGDLSTMModel, config_path: str, n_samples: int, estimator_cache_file: str) -> MonteCarloEstimator:
    """
    Creates the core components needed for inference, primarily the MonteCarloEstimator.
    This step is resource-intensive as it involves generating many samples.
    """
    print("--- ⚙️ Setting up Inference Components ---")
    preprocessor = Preprocessor(config_path=config_path)
    
    print(f"Initializing MonteCarloEstimator with {n_samples} samples...")
    # This process can be slow if the cache file does not exist
    estimator = MonteCarloEstimator(
        model=model, 
        preprocessor=preprocessor, 
        n_samples=n_samples, 
        estimator_file=estimator_cache_file
    )
    print("✅ Inference components ready.")
    return estimator

# ==============================================================================
# ## Function 3: Inference (Batch & Single)
# ==============================================================================
def inference_for_batch(estimator: MonteCarloEstimator, test_file: str, result_file: str):
    """
    **[Batch]** Runs inference on an entire test file using the prepared estimator.
    """
    print(f"--- 📊 Starting Batch Inference on {test_file} ---")
    # Ensure the results directory exists
    os.makedirs(os.path.dirname(result_file), exist_ok=True)
    test_montecarlo(
        montecarlo_estimator=estimator, 
        test_file=test_file, 
        result_file=result_file
    )
    print(f"✅ Batch inference complete. Results saved to {result_file}")

def inference_for_single_password(estimator: MonteCarloEstimator, password: str) -> Dict[str, Any]:
    """
    **[Single]** Runs inference on a single password using the prepared estimator.
    """
    print(f"--- 🎯 Starting Single Password Inference for '{password}' ---")
    strength = estimator.compute_strength(password)
    avg_strength, prob = estimator.compute_average_strength(password)
    
    return {
        "password": password,
        "strength": strength,
        "average_strength": avg_strength,
        "probability": prob
    }

# ==============================================================================
# ## Demos
# ==============================================================================
if __name__ == '__main__':

    # --- DEMO 1: Train a New Model ---
    def run_training_demo():
        """Demonstrates how to call the train_model function."""
        print("\n*** ▶️ RUNNING TRAINING DEMO ***")
        DATA_PATH = "data/trainset/rockyou320w.txt"
        CONFIG_PATH = "config/rockyou_dpsgd_train.json"
        train_model(config_path=CONFIG_PATH, data_path=DATA_PATH)

    # --- DEMO 2: Run Batch Inference ---
    def run_batch_inference_demo():
        """Demonstrates the full flow: load -> setup -> batch inference."""
        print("\n*** ▶️ RUNNING BATCH INFERENCE DEMO ***")
        CONFIG_PATH = "config/rockyou_dpsgd_train.json"
        CHECKPOINT_PATH = "./models/model/rockyou320w_dpsgd_0.1236.pth"
        ESTIMATOR_CACHE = "./estimators/rockyou320w_dpsgd_0.1236_estimator.pkl"
        TEST_FILE = "./data/test/cityday_less.txt"
        RESULT_FILE = "./results/rockyou320w_cityday_results_dpsgd_0.1236.csv"
        
        try:
            # Step 1: Load the model
            model = load_model(config_path=CONFIG_PATH, checkpoint_path=CHECKPOINT_PATH)
            
            # Step 2: Prepare inference components (this is the slow part)
            estimator = setup_inference(
                model=model, 
                config_path=CONFIG_PATH, 
                n_samples=10000, 
                estimator_cache_file=ESTIMATOR_CACHE
            )
            
            # Step 3: Run batch inference
            inference_for_batch(estimator=estimator, test_file=TEST_FILE, result_file=RESULT_FILE)
            
        except Exception as e:
            print(f"❌ An error occurred during batch inference demo: {e}")

    # --- DEMO 3: Run Single Password Inference ---
    def run_single_inference_demo():
        """Demonstrates the full flow: load -> setup -> single password inference."""
        print("\n*** ▶️ RUNNING SINGLE PASSWORD INFERENCE DEMO ***")
        CONFIG_PATH = "config/rockyou_dpsgd_train.json"
        CHECKPOINT_PATH = "./models/model/rockyou320w_dpsgd_0.1236.pth"
        ESTIMATOR_CACHE = "./estimators/rockyou320w_dpsgd_0.1236_estimator.pkl"
        PASSWORD_TO_TEST = "password123"
        
        try:
            # Steps 1 & 2 are the same as the batch demo
            model = load_model(config_path=CONFIG_PATH, checkpoint_path=CHECKPOINT_PATH)
            estimator = setup_inference(
                model=model, 
                config_path=CONFIG_PATH, 
                n_samples=10000, 
                estimator_cache_file=ESTIMATOR_CACHE
            )

            # Step 3: Run inference on the single password
            results = inference_for_single_password(estimator=estimator, password=PASSWORD_TO_TEST)
            
            print("\n--- 📝 Inference Result ---")
            for key, value in results.items():
                if isinstance(value, float):
                    print(f"{key.capitalize():<20}: {value:.4f}")
                else:
                    print(f"{key.capitalize():<20}: {value}")

        except Exception as e:
            print(f"❌ An error occurred during single inference demo: {e}")

    # --- Instructions ---
    # Uncomment the function you want to run.
    
    # run_training_demo()
    # run_batch_inference_demo()
    run_single_inference_demo()