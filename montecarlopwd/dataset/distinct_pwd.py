import argparse
import os
from tqdm import tqdm

def process_and_deduplicate(input_path: str):
    """
    读取密码文件，检测并移除重复项，然后将唯一密码写入新文件。

    Args:
        input_path (str): 原始密码文件的路径。
    """
    print(f"--- Processing file: '{input_path}' ---")

    try:
        # --- 第1步: 读取文件并收集密码 ---
        # 我们同时使用一个列表来保持原始顺序（如果需要），和一个集合来跟踪唯一项
        all_passwords = []
        unique_passwords = set()

        print("Step 1: Reading passwords from the input file...")
        with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in tqdm(f, desc="Reading"):
                password = line.strip()
                if password: # 忽略空行
                    all_passwords.append(password)

        if not all_passwords:
            print("File is empty or contains only empty lines. No action taken.")
            return

        # --- 第2步: 检测重复并创建去重列表 ---
        print("Step 2: Identifying unique passwords...")
        distinct_list = []
        for pwd in tqdm(all_passwords, desc="Deduplicating"):
            if pwd not in unique_passwords:
                unique_passwords.add(pwd)
                distinct_list.append(pwd)
        
        # --- 第3步: 打印重复标志和统计信息 ---
        print("\n--- Processing Report ---")
        
        initial_count = len(all_passwords)
        distinct_count = len(distinct_list)
        
        # 打印标志 (flag)
        if initial_count > distinct_count:
            print("✅ [FLAG] Duplicates were found and removed.")
        else:
            print("✅ [FLAG] No duplicates found in the file.")
            
        print(f"Initial password count: {initial_count}")
        print(f"Distinct password count: {distinct_count}")
        print(f"Number of duplicates removed: {initial_count - distinct_count}")

        # --- 第4步: 写入新文件 ---
        # 生成新文件名，例如 'passwords.txt' -> 'passwords_distinct.txt'
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_distinct{ext}"
        
        print(f"\nStep 3: Writing distinct passwords to '{output_path}'...")
        with open(output_path, 'w', encoding='utf-8') as f:
            for pwd in tqdm(distinct_list, desc="Writing"):
                f.write(pwd + '\n')
        
        print("\n✅ Process complete.")

    except FileNotFoundError:
        print(f"❌ Error: Input file not found at '{input_path}'")
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="A script to find and remove duplicate passwords from a text file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("input_file", help="Path to the raw password file you want to process.")
    
    args = parser.parse_args()
    
    process_and_deduplicate(input_path=args.input_file)