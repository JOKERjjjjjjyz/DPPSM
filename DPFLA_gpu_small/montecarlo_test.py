import json
import torch
import argparse
from utils.preprocessing import Preprocessor
from utils.guessing_gpu import MonteCarloEstimator, test_montecarlo
from models.dpsgd_lstm_model import DPSGDLSTMModel

def main(model_type, model_suffix):
    data_path = "data/trainset/rockyou_new.txt"
    config_path = f"config/rockyou_{model_type}_train.json"
    # config_path = f"config/rockyou_dpfla_train.json"
    
    # Load model

    model = DPSGDLSTMModel(config_path=config_path)

    checkpoint_path = f"./models/model/rockyou_new_{model_type}_{model_suffix}.pth"
    checkpoint = torch.load(checkpoint_path)

    # Process state_dict
    new_state_dict = {}
    for k, v in checkpoint.items():
        new_key = k.replace("_module.", "") if "_module." in k else k
        new_state_dict[new_key] = v

    # Load state_dict into model
    model.load_state_dict(new_state_dict)
    model.eval()

    # Load preprocessor
    preprocessor = Preprocessor(config_path=config_path)

    # Monte Carlo Estimator
    n_samples = 10000
    estimator_file = f'rockyou_new_{model_type}_{model_suffix}_estimator.pkl'
    montecarlo_estimator = MonteCarloEstimator(model=model, preprocessor=preprocessor, n_samples=n_samples, estimator_file=estimator_file)

    # Example passwords
    test_passwords = ["abcdefgh", "abcdefghhhhhhhhhh"]
    for test_password in test_passwords:
        strength = montecarlo_estimator.compute_strength(test_password)
        print(f"Strength of password '{test_password}': {strength}")
        avg_strength, prob = montecarlo_estimator.compute_average_strength(test_password)
        print(f"Average strength of password '{test_password}': {avg_strength}, Probability: {prob}")

    # Monte Carlo test on a file
    # test_file = "./data/test/cityday_less.txt"  # Replace with actual test file
    test_file = "./data/test/rockyou2024_less.txt"
    result_file = f"./results/rock2019_rock2024_results_{model_type}_{model_suffix}.csv"
    # result_file = f"./results/rock2019_city_results_{model_type}_{model_suffix}.csv"
    test_montecarlo(montecarlo_estimator=montecarlo_estimator, test_file=test_file, result_file=result_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Monte Carlo test with given model type and suffix.")
    parser.add_argument("model_type", type=str, help="Type of the model (e.g., 'dpfla' or 'dpsgd')")
    parser.add_argument("model_suffix", type=str, help="Suffix for the model and result files (e.g., '0.1178')")
    args = parser.parse_args()
    main(args.model_type, args.model_suffix)