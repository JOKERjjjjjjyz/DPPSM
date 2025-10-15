# mia_eval_ngram.py
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# --- 导入之前重构好的模型加载模块 ---
from ngram_dp_inference import load_model as load_ngram_model
from backoff_inference import load_model as load_backoff_model

def get_mia_scores(target_model, ref_model, member_file, non_member_file) -> pd.DataFrame:
    """
    计算MIA分数并与真实标签配对，返回一个DataFrame。
    """
    results = []
    # 定义统一的推理接口
    target_infer = target_model.logprob
    ref_infer = ref_model.logprob
    
    # 处理成员文件 (label=1)
    print("Processing members...")
    with open(member_file, 'r', encoding='utf-8', errors='ignore') as f:
        for pwd in tqdm(f):
            pwd = pwd.strip()
            if not pwd: continue
            mia_score = target_infer(pwd) - ref_infer(pwd)
            results.append({"mia_score": mia_score, "true_label": 1})

    # 处理非成员文件 (label=0)
    print("Processing non-members...")
    with open(non_member_file, 'r', encoding='utf-8', errors='ignore') as f:
        for pwd in tqdm(f):
            pwd = pwd.strip()
            if not pwd: continue
            mia_score = target_infer(pwd) - ref_infer(pwd)
            results.append({"mia_score": mia_score, "true_label": 0})
            
    return pd.DataFrame(results)

def analyze_and_plot_roc_auc(df_scores: pd.DataFrame, model_name: str):
    """
    根据MIA分数和真实标签计算AUC并绘制ROC曲线。
    """
    y_true = df_scores['true_label']
    y_scores = df_scores['mia_score']

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    print(f"\n--- Analysis for {model_name} ---")
    print(f"AUC Score: {roc_auc:.4f}")

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title(f'MIA ROC Curve for {model_name} Model')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    CONFIG_FILE = "config_mia.json"
    with open(CONFIG_FILE, 'r') as f:
        config = json.load(f)

    # --- Demo Code ---
    print("--- Running MIA Demo for DP n-gram Model ---")

    # 1. 加载模型
    cfg_target = config['target_models']['ngram_dp']
    cfg_ref = config['reference_model']['backoff']
    
    target_model = load_ngram_model(model_path=cfg_target['model_path'], n=cfg_target['n'])
    ref_model = load_backoff_model(model_path=cfg_ref['model_path'], threshold=cfg_ref['threshold'])

    # 2. 获取MIA分数
    mia_scores_df = get_mia_scores(
        target_model=target_model,
        ref_model=ref_model,
        member_file=config['data']['ground_truth_members'],
        non_member_file=config['data']['ground_truth_non_members']
    )

    print("\nSample of generated scores:")
    print(mia_scores_df.head())
    
    # 3. 分析并绘图
    analyze_and_plot_roc_auc(mia_scores_df, "DP n-gram")