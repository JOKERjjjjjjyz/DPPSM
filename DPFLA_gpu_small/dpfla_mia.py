# mia_eval_dpfla_vs_dpfla.py (已更新)

import json
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import os
import torch

# --- 导入您项目中的模型和工具代码 ---
from models.dpsgd_lstm_model import DPSGDLSTMModel
from utils.preprocessing import Preprocessor
from utils.guessing_gpu import MonteCarloEstimator

# ==============================================================================
# 辅助函数 (进行适应性修改)
# ==============================================================================
def load_model(config_path: str) -> DPSGDLSTMModel:
    """
    ## 修改 ##
    从指定的配置文件中读取 'model_file' 路径并加载模型。
    """
    print(f"--- Loading model using config: '{config_path}' ---")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # 1. 从 config 文件读取 model_file 路径
    with open(config_path, 'r') as f:
        config = json.load(f)
    model_path = config.get('model_file')
    if not model_path or not os.path.exists(model_path):
        raise FileNotFoundError(f"Key 'model_file' not found in '{config_path}' or path is invalid: {model_path}")
    
    # 2. 加载模型
    model = DPSGDLSTMModel(config_path=config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    new_state_dict = {k.replace("_module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    print(f"Model from '{model_path}' loaded successfully on device: {device}")
    return model

def setup_inference(model: DPSGDLSTMModel, config_path: str) -> MonteCarloEstimator:
    """
    ## 修改 ##
    从指定的配置文件中读取 estimator 和 n_samples 设置来创建 Estimator。
    """
    print(f"--- Setting up inference estimator using config: '{config_path}' ---")
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    estimator_cache_file = config.get('estimator_cache_file')
    n_samples = config.get('n_samples', 10000) # 从config读取或使用默认值

    preprocessor = Preprocessor(config_path=config_path)
    estimator = MonteCarloEstimator(
        model=model,
        preprocessor=preprocessor,
        n_samples=n_samples,
        estimator_file=estimator_cache_file
    )
    print("Estimator ready.")
    return estimator

# ==============================================================================
# 核心MIA函数 (保持不变)
# ==============================================================================
def get_mia_scores(target_infer_fn, ref_infer_fn, member_file, non_member_file) -> pd.DataFrame:
    # ... 此函数无需修改，保持原样 ...
    results = []
    print("\nProcessing members...")
    with open(member_file, 'r', encoding='utf-8', errors='ignore') as f:
        for pwd in tqdm(f, desc="Members"):
            pwd = pwd.strip();
            if not pwd: continue
            try:
                mia_score = ref_infer_fn(pwd) - target_infer_fn(pwd)
                results.append({"mia_score": mia_score, "true_label": 1})
            except Exception: continue

    print("Processing non-members...")
    with open(non_member_file, 'r', encoding='utf-8', errors='ignore') as f:
        for pwd in tqdm(f, desc="Non-Members"):
            pwd = pwd.strip();
            if not pwd: continue
            try:
                mia_score = ref_infer_fn(pwd) - target_infer_fn(pwd)
                results.append({"mia_score": mia_score, "true_label": 0})
            except Exception: continue
            
    return pd.DataFrame(results)

def analyze_and_plot_roc_auc(df_scores: pd.DataFrame, model_name: str, output_path: str = None):
    # ... 此函数无需修改，保持原样 ...
    y_true = df_scores['true_label']
    y_scores = df_scores['mia_score']
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    print(f"\n--- Analysis for {model_name} ---")
    print(f"AUC Score: {roc_auc:.4f}")
    plt.figure(figsize=(8, 6)); plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--'); plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title(f'MIA ROC Curve for {model_name} Model'); plt.legend(loc="lower right"); plt.grid(True)
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight')
        print(f"✅ ROC AUC plot saved to: {output_path}")
    else: plt.show()
    plt.close()


# ==============================================================================
# 主程序入口 (## 已修改 ##)
# ==============================================================================
if __name__ == '__main__':
    MIA_CONFIG_FILE = "config_mia.json"
    try:
        with open(MIA_CONFIG_FILE, 'r') as f: config = json.load(f)
    except FileNotFoundError:
        print(f"❌ Error: MIA Config file '{MIA_CONFIG_FILE}' not found."); exit()

    # --- 选择要攻击的目标模型 (从 config_mia.json 的键中选择一个) ---
    TARGET_MODEL_KEY = "dpfla_epsilon_0.2" # 您可以在这里切换要测试的模型
    
    # --- 固定参考模型 (只有一个) ---
    # 获取参考模型的配置字典 (我们假设只有一个参考模型)
    ref_model_name = list(config['reference_model'].keys())[0]
    cfg_ref_model = config['reference_model'][ref_model_name]
    ref_model_config_path = cfg_ref_model['config_path']

    # --- 获取选定的目标模型的配置 ---
    print(f"--- Running MIA against Target Model: '{TARGET_MODEL_KEY}' ---")
    try:
        cfg_target_model = config['target_models'][TARGET_MODEL_KEY]
        target_model_config_path = cfg_target_model['config_path']

        # 1. 加载并设置【参考模型】(只加载一次)
        print("\n--- Loading Reference Model ---")
        ref_model = load_model(config_path=ref_model_config_path)
        ref_estimator = setup_inference(model=ref_model, config_path=ref_model_config_path)

        # 2. 加载并设置【目标模型】
        print("\n--- Loading Target Model ---")
        target_model = load_model(config_path=target_model_config_path)
        target_estimator = setup_inference(model=target_model, config_path=target_model_config_path)
        
        # 3. 创建推理接口 (不变)
        def target_infer(password):
            _, prob = target_estimator.compute_average_strength(password)
            return prob
        
        def ref_infer(password):
            _, prob = ref_estimator.compute_average_strength(password)
            return prob

        # 4. 获取MIA分数 (不变)
        mia_scores_df = get_mia_scores(
            target_infer_fn=target_infer,
            ref_infer_fn=ref_infer,
            member_file=config['data']['ground_truth_members'],
            non_member_file=config['data']['ground_truth_non_members']
        )

        # 5. 数据清洗 (不变)
        mia_scores_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        mia_scores_df.dropna(subset=['mia_score'], inplace=True)
        
        # 6. 保存和绘图
        cfg_output = config['output_config']
        plot_path = cfg_output['plot_path_template'].format(model_name=TARGET_MODEL_KEY)
        scores_csv_path = cfg_output['scores_csv_path_template'].format(model_name=TARGET_MODEL_KEY)
        
        os.makedirs(os.path.dirname(scores_csv_path), exist_ok=True)
        mia_scores_df.to_csv(scores_csv_path, index=False)
        print(f"\n✅ Cleaned MIA scores saved to: {scores_csv_path}")

        plot_title = f"{TARGET_MODEL_KEY} vs {ref_model_name}"
        analyze_and_plot_roc_auc(df_scores=mia_scores_df, model_name=plot_title, output_path=plot_path)

    except (FileNotFoundError, KeyError, Exception) as e:
        print(f"\n❌ An error occurred during the process: {e}")