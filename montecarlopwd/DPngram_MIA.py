# DPngram_MIA.py (最终修正版)
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import os
import dbm

from ngram_dp_inference import load_model as load_ngram_model
from backoff_inference import load_model as load_backoff_model

def get_mia_scores(target_model, ref_model, member_file, non_member_file) -> pd.DataFrame:
    results = []
    target_infer = target_model.logprob
    ref_infer = ref_model.logprob
    
    print("Processing members...")
    with open(member_file, 'r', encoding='utf-8', errors='ignore') as f:
        for pwd in tqdm(f, desc="Members"):
            pwd = pwd.strip()
            if not pwd: continue
            try:
                mia_score = ref_infer(pwd) - target_infer(pwd)
                results.append({"mia_score": mia_score, "true_label": 1})
            except Exception: continue

    print("Processing non-members...")
    with open(non_member_file, 'r', encoding='utf-8', errors='ignore') as f:
        for pwd in tqdm(f, desc="Non-Members"):
            pwd = pwd.strip()
            if not pwd: continue
            try:
                mia_score = ref_infer(pwd) - target_infer(pwd)
                results.append({"mia_score": mia_score, "true_label": 0})
            except Exception: continue
            
    return pd.DataFrame(results)

def analyze_and_plot_roc_auc(df_scores: pd.DataFrame, model_name: str, output_path: str = None):
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

if __name__ == '__main__':
    CONFIG_FILE = "dpngram_mia.json"
    try:
        with open(CONFIG_FILE, 'r') as f: config = json.load(f)
    except FileNotFoundError:
        print(f"❌ Error: Config file '{CONFIG_FILE}' not found."); exit()

    MODEL_KEY = "ngram_dp"
    MODEL_NAME_PRETTY = "DP n-gram"
    
    print(f"--- Running MIA Demo for {MODEL_NAME_PRETTY} Model ---")

    try:
        cfg_target = config['target_models'][MODEL_KEY]
        cfg_ref = config['reference_model']['backoff']
        
        target_model = load_ngram_model(model_path=cfg_target['model_path'], n=cfg_target['n'], epsilon=cfg_target['epsilon'])
        ref_model = load_backoff_model(model_path=cfg_ref['model_path'], threshold=cfg_ref['threshold'])

        mia_scores_df = get_mia_scores(
            target_model=target_model,
            ref_model=ref_model,
            member_file=config['data']['ground_truth_members'],
            non_member_file=config['data']['ground_truth_non_members']
        )

        print("\nSample of generated scores (before cleaning):")
        print(mia_scores_df.head())
        
        # ==============================================================================
        # 【最终修正】: 智能处理边界值，并增加最终检查
        # ==============================================================================
        scores = mia_scores_df['mia_score'].to_numpy(dtype=np.float64)
        
        # 1. 处理 NaN (inf - inf)，视为极强的非成员信号 -> 赋予最低分
        nan_mask = np.isnan(scores)
        if np.any(nan_mask):
            scores[nan_mask] = np.finfo(np.float64).min
        
        # 2. 处理 -inf (有限数 - inf)，也视为极强的非成员信号 -> 赋予最低分
        neginf_mask = scores == -np.inf
        if np.any(neginf_mask):
            scores[neginf_mask] = np.finfo(np.float64).min

        # 3. 处理 inf (inf - 有限数)，视为极强的成员信号 -> 赋予最高分
        posinf_mask = scores == np.inf
        if np.any(posinf_mask):
            scores[posinf_mask] = np.finfo(np.float64).max
        
        # 将处理好的分数写回DataFrame
        mia_scores_df['mia_score'] = scores
        
        num_cleaned = np.sum(nan_mask) + np.sum(neginf_mask) + np.sum(posinf_mask)
        print(f"\n[INFO] Data Cleaning: Replaced {num_cleaned} non-finite scores (inf, -inf, NaN) with appropriate boundary values.")

        # 【新增】: 最终检查，确保所有非有限值都已被处理
        if not np.isfinite(mia_scores_df['mia_score']).all():
            # 如果处理后仍然存在 inf, -inf 或 NaN，则主动报错
            raise ValueError("FATAL: Non-finite values still exist after cleaning. Aborting.")
        else:
            print("[INFO] Final check passed: All scores are finite and ready for analysis.")
        # ==============================================================================
        # ==============================================================================
        
        cfg_output = config['output_config']
        plot_path = cfg_output['plot_path_template'].format(model_name=MODEL_KEY)
        scores_csv_path = cfg_output['scores_csv_path_template'].format(model_name=MODEL_KEY)

        os.makedirs(os.path.dirname(scores_csv_path), exist_ok=True)
        mia_scores_df.to_csv(scores_csv_path, index=False)
        print(f"\n✅ Cleaned MIA scores saved to: {scores_csv_path}")

        analyze_and_plot_roc_auc(
            df_scores=mia_scores_df, 
            model_name=MODEL_NAME_PRETTY,
            output_path=plot_path
        )
    except (FileNotFoundError, KeyError, Exception) as e:
        print(f"\n❌ An error occurred during the process: {e}")



# DPngram_MIA.py (使用n-gram作为参考模型)
# import json
# import pandas as pd
# import numpy as np
# from tqdm import tqdm
# import matplotlib.pyplot as plt
# from sklearn.metrics import roc_curve, auc
# import os
# import dbm

# # 【核心修正 1】: 不再需要 backoff 的加载器
# from ngram_dp_inference import load_model as load_ngram_model

# def get_mia_scores(target_model, ref_model, member_file, non_member_file) -> pd.DataFrame:
#     # 此函数完全不需要修改，因为它的设计是通用的
#     results = []
#     target_infer = target_model.logprob
#     ref_infer = ref_model.logprob
    
#     print("Processing members...")
#     with open(member_file, 'r', encoding='utf-8', errors='ignore') as f:
#         for pwd in tqdm(f, desc="Members"):
#             pwd = pwd.strip();
#             if not pwd: continue
#             try:
#                 mia_score = ref_infer(pwd) - target_infer(pwd)
#                 results.append({"mia_score": mia_score, "true_label": 1})
#             except Exception: continue

#     print("Processing non-members...")
#     with open(non_member_file, 'r', encoding='utf-8', errors='ignore') as f:
#         for pwd in tqdm(f, desc="Non-Members"):
#             pwd = pwd.strip();
#             if not pwd: continue
#             try:
#                 mia_score = ref_infer(pwd) - target_infer(pwd)
#                 results.append({"mia_score": mia_score, "true_label": 0})
#             except Exception: continue
            
#     return pd.DataFrame(results)

# def analyze_and_plot_roc_auc(df_scores: pd.DataFrame, model_name: str, output_path: str = None):
#     # 此函数也完全不需要修改
#     y_true = df_scores['true_label']
#     y_scores = df_scores['mia_score']
#     fpr, tpr, _ = roc_curve(y_true, y_scores)
#     roc_auc = auc(fpr, tpr)
#     print(f"\n--- Analysis for {model_name} ---")
#     print(f"AUC Score: {roc_auc:.4f}")
#     plt.figure(figsize=(8, 6)); plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
#     plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--'); plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
#     plt.title(f'MIA ROC Curve for {model_name} Model'); plt.legend(loc="lower right"); plt.grid(True)
#     if output_path:
#         os.makedirs(os.path.dirname(output_path), exist_ok=True)
#         plt.savefig(output_path, bbox_inches='tight')
#         print(f"✅ ROC AUC plot saved to: {output_path}")
#     else: plt.show()
#     plt.close()

# if __name__ == '__main__':
#     CONFIG_FILE = "dpngram_mia.json"
#     try:
#         with open(CONFIG_FILE, 'r') as f: config = json.load(f)
#     except FileNotFoundError:
#         print(f"❌ Error: Config file '{CONFIG_FILE}' not found."); exit()

#     MODEL_KEY = "ngram_dp"
#     MODEL_NAME_PRETTY = "DP n-gram"
    
#     print(f"--- Running MIA Demo for {MODEL_NAME_PRETTY} Model ---")

#     try:
#         cfg_target = config['target_models'][MODEL_KEY]
        
#         # 【核心修正 2】: 从配置中读取新的参考模型 "ngram_ref"
#         cfg_ref = config['reference_model']['ngram_ref']
        
#         # 加载目标模型
#         print("\n--- Loading Target Model ---")
#         target_model = load_ngram_model(model_path=cfg_target['model_path'], n=cfg_target['n'], epsilon=cfg_target['epsilon'])
        
#         # 【核心修正 3】: 使用同一个加载器来加载参考模型
#         print("\n--- Loading Reference Model ---")
#         ref_model = load_ngram_model(model_path=cfg_ref['model_path'], n=cfg_ref['n'], epsilon=cfg_ref['epsilon'])

#         mia_scores_df = get_mia_scores(
#             target_model=target_model,
#             ref_model=ref_model,
#             member_file=config['data']['ground_truth_members'],
#             non_member_file=config['data']['ground_truth_non_members']
#         )

#         print("\nSample of generated scores (before cleaning):")
#         print(mia_scores_df.head())
        
#         # 数据清洗逻辑保持不变
#         scores = mia_scores_df['mia_score'].to_numpy(dtype=np.float64)
#         nan_mask = np.isnan(scores); scores[nan_mask] = np.finfo(np.float64).min
#         neginf_mask = scores == -np.inf; scores[neginf_mask] = np.finfo(np.float64).min
#         posinf_mask = scores == np.inf; scores[posinf_mask] = np.finfo(np.float64).max
#         mia_scores_df['mia_score'] = scores
#         num_cleaned = np.sum(nan_mask) + np.sum(neginf_mask) + np.sum(posinf_mask)
#         print(f"\n[INFO] Data Cleaning: Replaced {num_cleaned} non-finite scores.")
#         if not np.isfinite(mia_scores_df['mia_score']).all():
#             raise ValueError("FATAL: Non-finite values still exist after cleaning.")
#         else:
#             print("[INFO] Final check passed: All scores are finite.")
        
#         # 保存和绘图逻辑保持不变
#         cfg_output = config['output_config']
#         plot_path = cfg_output['plot_path_template'].format(model_name=MODEL_KEY)
#         scores_csv_path = cfg_output['scores_csv_path_template'].format(model_name=MODEL_KEY)
#         os.makedirs(os.path.dirname(scores_csv_path), exist_ok=True)
#         mia_scores_df.to_csv(scores_csv_path, index=False)
#         print(f"\n✅ Cleaned MIA scores saved to: {scores_csv_path}")
#         analyze_and_plot_roc_auc(df_scores=mia_scores_df, model_name=MODEL_NAME_PRETTY, output_path=plot_path)

#     except (FileNotFoundError, KeyError, Exception) as e:
#         print(f"\n❌ An error occurred during the process: {e}")