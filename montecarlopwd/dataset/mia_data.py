import argparse
import os
import random
from tqdm import tqdm

def create_mia_data_sets_with_duplicates(
    raw_input_path: str, 
    distinct_input_path: str, 
    output_train_path: str, 
    output_eval_members_path: str, 
    output_eval_non_members_path: str, 
    holdout_ratio: float, 
    seed: int
):
    """
    Creates three datasets for MIA training and evaluation.
    - The training set contains duplicates from the raw file, minus the holdout set.
    - The evaluation sets are balanced and sampled from the distinct file.

    Args:
        raw_input_path (str): Path to the original password file, with duplicates.
        distinct_input_path (str): Path to the de-duplicated version of the password file.
        output_train_path (str): Path to save the main training set.
        output_eval_members_path (str): Path to save the sampled true member file for evaluation.
        output_eval_non_members_path (str): Path to save the true non-member file for evaluation.
        holdout_ratio (float): The proportion of distinct data to be used as the holdout/non-member set.
        seed (int): A random seed for reproducibility.
    """
    print(f"--- Creating Datasets with a new logic ---")
    print(f"Configuration: Holdout Ratio = {holdout_ratio*100:.0f}%, Seed = {seed}")

    try:
        # --- 1. 从去重文件中定义非成员集 ---
        print(f"Step 1: Defining holdout set from distinct file '{distinct_input_path}'...")
        with open(distinct_input_path, 'r', encoding='utf-8', errors='ignore') as f:
            distinct_passwords = [line.strip() for line in f if line.strip()]

        if not distinct_passwords:
            print("❌ Error: Distinct input file is empty.")
            return

        random.seed(seed)
        random.shuffle(distinct_passwords)

        split_index = int(len(distinct_passwords) * (1.0 - holdout_ratio))
        
        distinct_training_pool = distinct_passwords[:split_index]
        holdout_set_non_members = distinct_passwords[split_index:] # 这就是最终的非成员集

        print(f"  - Distinct data split into {len(distinct_training_pool)} potential members and {len(holdout_set_non_members)} non-members.")

        # --- 2. 创建包含重复项的训练集 ---
        print(f"\nStep 2: Creating the final training set by subtracting the holdout set from the raw file '{raw_input_path}'...")
        
        # 将非成员集放入Set中以便快速查找
        holdout_set_for_lookup = set(holdout_set_non_members)
        final_training_set = []

        with open(raw_input_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in tqdm(f, desc="Filtering raw file"):
                password = line.strip()
                if password and password not in holdout_set_for_lookup:
                    final_training_set.append(password)
        
        print(f"  - The new training set contains {len(final_training_set)} passwords (with duplicates).")

        # --- 3. 从不重复的训练池中采样，创建均衡的评估成员集 ---
        n_eval_samples = len(holdout_set_non_members)
        if len(distinct_training_pool) < n_eval_samples:
            print(f"❌ Error: Not enough unique passwords in the training pool to create a balanced evaluation set.")
            return
            
        print(f"\nStep 3: Sampling {n_eval_samples} passwords from the distinct training pool for the evaluation member set...")
        eval_member_subset = random.sample(distinct_training_pool, n_eval_samples)

        # --- 4. 写入所有三个输出文件 ---
        print("\nStep 4: Writing all three output files...")
        
        # 确保输出目录存在
        for path in [output_train_path, output_eval_members_path, output_eval_non_members_path]:
             os.makedirs(os.path.dirname(path), exist_ok=True)

        # 文件1: 主训练集 (带重复, 且不含非成员)
        with open(output_train_path, 'w', encoding='utf-8') as f:
            for pwd in tqdm(final_training_set, desc="Writing Train Set"):
                f.write(pwd + '\n')
                
        # 文件2: 评估用的成员集 (不重复, 数量与非成员集相同)
        with open(output_eval_members_path, 'w', encoding='utf-8') as f:
            for pwd in tqdm(eval_member_subset, desc="Writing Eval Members"):
                f.write(pwd + '\n')
                
        # 文件3: 评估用的非成员集 (不重复)
        with open(output_eval_non_members_path, 'w', encoding='utf-8') as f:
            for pwd in tqdm(holdout_set_non_members, desc="Writing Eval Non-Members"):
                f.write(pwd + '\n')

        # --- 5. 最终报告 ---
        print("\n--- Process Report ---")
        print(f"Original distinct passwords: {len(distinct_passwords)}")
        print("-" * 40)
        print("Generated three files:")
        print(f"1. ✅ Main Training Set (with duplicates): '{output_train_path}' ({len(final_training_set)} passwords)")
        print(f"2. ✅ Evaluation Members (distinct):      '{output_eval_members_path}' ({len(eval_member_subset)} passwords)")
        print(f"3. ✅ Evaluation Non-Members (distinct):  '{output_eval_non_members_path}' ({len(holdout_set_non_members)} passwords)")
        print("\nWorkflow complete. Use File 1 to train, Files 2 and 3 to evaluate.")

    except FileNotFoundError as e:
        print(f"❌ Error: Input file not found - {e.filename}")
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Split password data into a training set (with duplicates) and balanced evaluation sets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("raw_input", help="Path to the original password file, WITH duplicates.")
    parser.add_argument("distinct_input", help="Path to the DE-DUPLICATED version of the password file.")
    parser.add_argument("output_train", help="Path to save the main training set.")
    parser.add_argument("output_eval_members", help="Path to save the sampled member file for evaluation.")
    parser.add_argument("output_eval_non_members", help="Path to save the non-member file for evaluation.")
    
    parser.add_argument("--ratio", type=float, default=0.2,
                        help="The proportion of DISTINCT data to be used as the holdout/non-member set (e.g., 0.2 for an 80/20 split).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for shuffling to ensure reproducible results.")
    
    args = parser.parse_args()

    if not 0.0 < args.ratio < 1.0:
        print("Error: --ratio must be between 0.0 and 1.0 (exclusive).")
    else:
        create_mia_data_sets_with_duplicates(
            raw_input_path=args.raw_input,
            distinct_input_path=args.distinct_input,
            output_train_path=args.output_train,
            output_eval_members_path=args.output_eval_members,
            output_eval_non_members_path=args.output_eval_non_members,
            holdout_ratio=args.ratio,
            seed=args.seed
        )