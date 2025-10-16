import argparse
import os
import random
from tqdm import tqdm

def create_mia_data_sets(input_path: str, output_train_path: str, output_eval_members_path: str, output_eval_non_members_path: str, train_ratio: float, seed: int):
    """
    Reads a de-duplicated password file and splits it into three sets:
    1. A large set for training the target model.
    2. A balanced set of true members for evaluation.
    3. A balanced set of true non-members for evaluation.

    Args:
        input_path (str): Path to the large, de-duplicated source password file.
        output_train_path (str): Path to save the main training set (e.g., the 80% part).
        output_eval_members_path (str): Path to save the sampled true member file for evaluation.
        output_eval_non_members_path (str): Path to save the true non-member file for evaluation.
        train_ratio (float): The proportion of the data to be used as the training pool.
        seed (int): A random seed for ensuring reproducibility.
    """
    print(f"--- Creating All Datasets for MIA from '{input_path}' ---")
    print(f"Configuration: Train/Holdout Ratio = {train_ratio*100:.0f}/{100-train_ratio*100:.0f}, Seed = {seed}")

    try:
        # --- 1. 读取所有密码 ---
        print("Step 1: Reading all passwords into memory...")
        with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
            passwords = [line.strip() for line in f if line.strip()]
        
        if not passwords:
            print("❌ Error: Input file is empty.")
            return

        # --- 2. 打乱数据 ---
        print(f"Step 2: Shuffling {len(passwords)} passwords...")
        random.seed(seed)
        random.shuffle(passwords)

        # --- 3. 分割为训练池和留出集(非成员) ---
        split_index = int(len(passwords) * train_ratio)
        
        training_pool = passwords[:split_index]
        holdout_set_non_members = passwords[split_index:] # 这就是评估用的非成员集
        
        n_training = len(training_pool)
        n_non_members = len(holdout_set_non_members)

        print(f"Step 3: Split complete.")
        print(f"  - Main Training Pool size: {n_training}")
        print(f"  - Holdout Set (for eval non-members): {n_non_members}")

        if n_training < n_non_members:
            print(f"❌ Error: Training pool ({n_training}) is smaller than the holdout set ({n_non_members}). Cannot sample enough members.")
            return
            
        # --- 4. 从训练池中采样与非成员集数量相同的成员，用于评估 ---
        print(f"Step 4: Sampling {n_non_members} passwords from the training pool for the evaluation member set...")
        eval_member_subset = random.sample(training_pool, n_non_members)

        # --- 5. 写入所有三个输出文件 ---
        print("Step 5: Writing all three output files...")
        
        # 确保输出目录存在
        for path in [output_train_path, output_eval_members_path, output_eval_non_members_path]:
            os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # 文件1: 主训练集 (80%)
        with open(output_train_path, 'w', encoding='utf-8') as f:
            for pwd in tqdm(training_pool, desc="Writing Train Set"):
                f.write(pwd + '\n')
                
        # 文件2: 评估用的成员集
        with open(output_eval_members_path, 'w', encoding='utf-8') as f:
            for pwd in tqdm(eval_member_subset, desc="Writing Eval Members"):
                f.write(pwd + '\n')
                
        # 文件3: 评估用的非成员集 (20%)
        with open(output_eval_non_members_path, 'w', encoding='utf-8') as f:
            for pwd in tqdm(holdout_set_non_members, desc="Writing Eval Non-Members"):
                f.write(pwd + '\n')

        # --- 6. 最终报告 ---
        print("\n--- Process Report ---")
        print(f"Original distinct passwords: {len(passwords)}")
        print("-" * 40)
        print("Generated three files:")
        print(f"1. ✅ Main Training Set: '{output_train_path}' ({len(training_pool)} passwords)")
        print(f"2. ✅ Evaluation Members: '{output_eval_members_path}' ({len(eval_member_subset)} passwords)")
        print(f"3. ✅ Evaluation Non-Members: '{output_eval_non_members_path}' ({len(holdout_set_non_members)} passwords)")
        print("\nWorkflow complete: Use File 1 to train your model, then use Files 2 and 3 for MIA evaluation.")

    except FileNotFoundError:
        print(f"❌ Error: Input file not found at '{input_path}'")
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Split a distinct password file into a training set and balanced evaluation sets for MIA.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("input_file", help="Path to the large, de-duplicated source password file.")
    parser.add_argument("output_train", help="Path to save the main training set (e.g., the 80% part).")
    parser.add_argument("output_eval_members", help="Path to save the sampled member file for evaluation.")
    parser.add_argument("output_eval_non_members", help="Path to save the non-member file for evaluation.")
    
    parser.add_argument("--ratio", type=float, default=0.8,
                        help="The proportion of data to be used for the training pool (e.g., 0.8 for an 80/20 split).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for shuffling to ensure reproducible results.")
    
    args = parser.parse_args()

    if not 0.0 < args.ratio < 1.0:
        print("Error: --ratio must be between 0.0 and 1.0 (exclusive).")
    else:
        create_mia_data_sets(
            input_path=args.input_file,
            output_train_path=args.output_train,
            output_eval_members_path=args.output_eval_members,
            output_eval_non_members_path=args.output_eval_non_members,
            train_ratio=args.ratio,
            seed=args.seed
        )