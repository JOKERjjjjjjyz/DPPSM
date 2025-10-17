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
# ## Function 1: Train Model (Unchanged)
# ==============================================================================
def train_model(config_path: str, data_path: str):
    """
    Initializes the DPTrainer and starts the training process.
    """
    print("--- 🚀 Starting Model Training ---")
    try:
        trainer = DPTrainer(config_path=config_path, data_path=data_path)
        trainer.train()
        print("--- ✅ Training Finished ---")
    except Exception as e:
        print(f"❌ An error occurred during training: {e}")

# ==============================================================================
# ## Function 2: Load Model (Modified)
# ==============================================================================
def load_model(config_path: str) -> DPSGDLSTMModel:
    """
    ## MODIFIED ##
    Loads a pre-trained model using only the config file.
    The path to the model checkpoint is now read from the 'model_file' key within the config.
    """
    print(f"--- 📥 Loading Pre-trained Model using '{config_path}' ---")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # 1. Read the specific model's checkpoint path from its config file
    with open(config_path, 'r') as f:
        config = json.load(f)
    checkpoint_path = config.get('model_file')
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Key 'model_file' not found in config or path is invalid: {checkpoint_path}")
    
    print(f"INFO: Found model checkpoint path in config: '{checkpoint_path}'")

    # 2. Load model structure and state_dict
    model = DPSGDLSTMModel(config_path=config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    new_state_dict = {k.replace("_module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    
    print(f"✅ Model loaded successfully on device: {device}")
    return model

# ==============================================================================
# ## Helper Function: Setup Inference (Unchanged)
# ==============================================================================
def setup_inference(model: DPSGDLSTMModel, config_path: str, n_samples: int, estimator_cache_file: str) -> MonteCarloEstimator:
    """
    Creates the core components needed for inference, primarily the MonteCarloEstimator.
    """
    print("--- ⚙️ Setting up Inference Components ---")
    preprocessor = Preprocessor(config_path=config_path)
    print(f"Initializing MonteCarloEstimator with {n_samples} samples...")
    estimator = MonteCarloEstimator(
        model=model, 
        preprocessor=preprocessor, 
        n_samples=n_samples, 
        estimator_file=estimator_cache_file
    )
    print("✅ Inference components ready.")
    return estimator

# ==============================================================================
# ## Function 3: Inference (Batch & Single) (Unchanged)
# ==============================================================================
def inference_for_batch(estimator: MonteCarloEstimator, test_file: str, result_file: str):
    """
    **[Batch]** Runs inference on an entire test file.
    """
    print(f"--- 📊 Starting Batch Inference on {test_file} ---")
    os.makedirs(os.path.dirname(result_file), exist_ok=True)
    test_montecarlo(
        montecarlo_estimator=estimator, 
        test_file=test_file, 
        result_file=result_file
    )
    print(f"✅ Batch inference complete. Results saved to {result_file}")

def inference_for_single_password(estimator: MonteCarloEstimator, password: str) -> Dict[str, Any]:
    """
    **[Single]** Runs inference on a single password.
    """
    print(f"--- 🎯 Starting Single Password Inference for '{password}' ---")
    strength = estimator.compute_strength(password)
    avg_strength, prob = estimator.compute_average_strength(password)
    return { "password": password, "strength": strength, "average_strength": avg_strength, "probability": prob }

# ==============================================================================
# ## Demos (Modified)
# ==============================================================================
if __name__ == '__main__':
    CONFIG_PATH = "config/rockyou_dpsgd_train.json"

    # --- DEMO 1: Train a New Model ---
    def run_training_demo():
        """
        ## MODIFIED ##
        Reads the training data path ('pwd_file') from the config file.
        """
        print("\n*** ▶️ RUNNING TRAINING DEMO ***")
        try:
            with open(CONFIG_PATH, 'r') as f:
                config = json.load(f)
            data_path = config.get('pwd_file')
            if not data_path:
                raise ValueError("'pwd_file' key not found in the config file.")
            print(f"INFO: Found training data path in config: '{data_path}'")
            train_model(config_path=CONFIG_PATH, data_path=data_path)
        except Exception as e:
            print(f"❌ An error occurred during training demo: {e}")

    # --- DEMO 2: Run Batch Inference ---
    def run_batch_inference_demo():
        """
        ## MODIFIED ##
        Reads all required paths (test_file, result_file, etc.) from the config file.
        """
        print("\n*** ▶️ RUNNING BATCH INFERENCE DEMO ***")
        try:
            # Step 1: Load all paths from the config file
            with open(CONFIG_PATH, 'r') as f:
                config = json.load(f)
            estimator_cache = config.get('estimator_cache_file')
            test_file = config.get('test_file')
            result_file = config.get('result_file')
            n_samples = config.get('n_samples', 10000)

            # Step 2: Load the model (the function now finds the path automatically)
            model = load_model(config_path=CONFIG_PATH)
            
            # Step 3: Prepare inference components
            estimator = setup_inference(
                model=model, 
                config_path=CONFIG_PATH, 
                n_samples=n_samples, 
                estimator_cache_file=estimator_cache
            )
            
            # Step 4: Run batch inference
            inference_for_batch(estimator=estimator, test_file=test_file, result_file=result_file)
        except Exception as e:
            print(f"❌ An error occurred during batch inference demo: {e}")

    # --- DEMO 3: Run Single Password Inference ---
    def run_single_inference_demo():
        """
        ## MODIFIED ##
        Reads the estimator cache path from the config file.
        """
        print("\n*** ▶️ RUNNING SINGLE PASSWORD INFERENCE DEMO ***")
        PASSWORD_TO_TEST = "password123"
        try:
            # Step 1: Load required paths from config
            with open(CONFIG_PATH, 'r') as f:
                config = json.load(f)
            estimator_cache = config.get('estimator_cache_file')
            n_samples = config.get('n_samples', 10000)

            # Step 2: Load the model
            model = load_model(config_path=CONFIG_PATH)

            # Step 3: Prepare inference components
            estimator = setup_inference(
                model=model, 
                config_path=CONFIG_PATH, 
                n_samples=n_samples, 
                estimator_cache_file=estimator_cache
            )
            
            # Step 4: Run inference
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
    run_batch_inference_demo()
    # run_single_inference_demo()