import matplotlib.pyplot as plt
from collections import Counter
import sys
import os

def count_ascii_frequency(filename, ascii_count=97):
    """
    读取指定文本文件，统计其中只属于 97 个 ASCII 字符（ASCII码 32～128）的出现频率。
    """
    # 定义允许统计的字符列表
    allowed_chars = [chr(i) for i in range(32, 32 + ascii_count)]
    
    # 读取文件内容
    with open(filename, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # 只统计允许字符出现的次数
    counter = Counter(ch for ch in text if ch in allowed_chars)
    
    # 确保所有允许的字符都有记录（若未出现则记为0）
    freq = {ch: counter.get(ch, 0) for ch in allowed_chars}
    return freq

def plot_histogram(freq, output_pdf):
    """
    根据统计的频率数据绘制直方图，将频率转换为百分比，并保存为 PDF 文件。
    """
    total = sum(freq.values())
    # 计算百分比
    percent = {ch: (count / total * 100) for ch, count in freq.items()}
    
    # 排序保持字符顺序
    chars = list(percent.keys())
    percentages = [percent[ch] for ch in chars]
    
    plt.figure(figsize=(12, 6))
    plt.bar(chars, percentages)
    plt.xlabel("ASCII Character")
    plt.ylabel("Percentage (%)")
    plt.title("Frequency (%) of 97 ASCII Characters")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(output_pdf)
    plt.show()

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python ascii_frequency.py <待测.txt>")
        sys.exit(1)
    
    filename = sys.argv[1]
    freq = count_ascii_frequency(filename, ascii_count=97)
    
    # 构造输出 PDF 文件名，形如：待测_frequency.pdf
    base_name = os.path.splitext(os.path.basename(filename))[0]
    output_pdf = f"{base_name}_frequency.pdf"
    
    plot_histogram(freq, output_pdf)
