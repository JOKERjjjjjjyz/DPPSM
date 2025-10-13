import sys
import random
import os

def reservoir_sample(input_file, k, output_file):
    """
    从 input_file 中随机抽取 k 行，采用水库抽样算法，
    将抽样结果写入 output_file。
    """
    reservoir = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i < k:
                reservoir.append(line)
            else:
                # 生成一个 [0, i] 的随机整数
                r = random.randint(0, i)
                if r < k:
                    reservoir[r] = line

    with open(output_file, 'w', encoding='utf-8') as out_f:
        out_f.writelines(reservoir)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python sample_lines.py <input_file.txt> [sample_size]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    sample_size = int(sys.argv[2]) if len(sys.argv) > 2 else 1000000
    
    # 构造输出文件名，例如 input.txt -> input_sample.txt
    base_name, ext = os.path.splitext(os.path.basename(input_file))
    output_file = f"{base_name}_sample.txt"
    
    reservoir_sample(input_file, sample_size, output_file)
    print(f"抽样完成：共抽取 {sample_size} 行，结果保存在 {output_file}")
