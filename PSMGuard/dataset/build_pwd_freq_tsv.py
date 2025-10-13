#!/usr/bin/env python
"""
将纯密码列表（.txt）转换为密码及其出现频率的 TSV 文件

Usage:
    python build_pwd_freq_tsv.py --input passwords.txt --output pwd_freq.tsv [--sort]

功能：
  1. 从 input.txt 中读取每行一个密码的列表。
  2. 统计每个密码出现的次数（frequency）。
  3. 将密码和对应的 frequency 写入 output.tsv，每行格式：
       password<tab>frequency
  4. 默认按照原始顺序输出，也可通过 --sort 参数按降序频率排序。

依赖：
  Python 3.6+
"""

import argparse
import sys
from collections import Counter
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="从 .txt 密码列表生成 password\tfrequency 的 TSV 文件"
    )
    parser.add_argument(
        "--input", required=True,
        help="原始密码 .txt 文件路径，每行一个密码"
    )
    parser.add_argument(
        "--output", required=True,
        help="输出的 .tsv 文件路径，格式为 password<TAB>frequency"
    )
    parser.add_argument(
        "--sort", action="store_true",
        help="是否按 frequency 降序排序（默认保留原始顺序）"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"Error: 输入文件 {input_path} 不存在", file=sys.stderr)
        sys.exit(1)

    # 读取所有密码到列表
    with open(input_path, 'r', encoding='utf-8') as f:
        passwords = [line.strip() for line in f if line.strip()]

    # 统计 frequency
    counter = Counter(passwords)

    # 如果没有排序，保持首次出现顺序
    if args.sort:
        # 按 frequency 降序排序，频率相同按密码顺序
        items = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    else:
        # 保留输入中首次出现的顺序
        seen = set()
        ordered = []
        for pwd in passwords:
            if pwd not in seen:
                seen.add(pwd)
                ordered.append(pwd)
        items = [(pwd, counter[pwd]) for pwd in ordered]

    # 写入 TSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as fw:
        for pwd, freq in items:
            fw.write(f"{pwd}\t{freq}\n")

    print(f"已生成 {output_path}，共 {len(items)} 条记录。")


if __name__ == "__main__":
    main()
