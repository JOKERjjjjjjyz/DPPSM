import matplotlib.pyplot as plt
from collections import Counter
import sys
import os
import random
import string

def count_ascii_frequency(text, ascii_count=97):
    """
    统计文本中 97 个 ASCII 字符（ASCII 码 32～128）的出现频率，
    返回一个字典，键为字符，值为出现次数。
    """
    allowed_chars = [chr(i) for i in range(32, 32 + ascii_count)]
    counter = Counter(ch for ch in text if ch in allowed_chars)
    freq = {ch: counter.get(ch, 0) for ch in allowed_chars}
    return freq

def generate_random_characters(n):
    """
    随机生成 n 个字符，其生成范围仅限于常见字符：
    包括字母、数字以及常用符号（常用于密码中）。
    """
    common_chars = string.ascii_letters + string.digits + "!@#$%^&*()_+-="
    random_chars = random.choices(common_chars, k=n)
    return ''.join(random_chars)

def plot_histogram(freq, output_pdf):
    """
    根据频数数据绘制直方图，将频数转换为百分比，并保存为 PDF 文件。
    """
    total = sum(freq.values())
    # 将频数转换为百分比
    percentages = {ch: (count / total * 100) for ch, count in freq.items()}
    chars = list(percentages.keys())
    values = [percentages[ch] for ch in chars]
    
    plt.figure(figsize=(12, 6))
    plt.bar(chars, values)
    plt.xlabel("ASCII Character")
    plt.ylabel("Percentage (%)")
    plt.title("Frequency (%) of 97 ASCII Characters")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(output_pdf)
    plt.show()

def main():
    if len(sys.argv) < 2:
        print("Usage: python character_random_trick.py <待测.txt>")
        sys.exit(1)
    
    filename = sys.argv[1]
    # 读取输入文件内容
    with open(filename, 'r', encoding='utf-8') as f:
        text = f.read()
    
    total_chars = len(text)
    # 生成额外随机字符，数量为原文本字符数的 20%
    num_random = int(0.2 * total_chars)
    random_text = generate_random_characters(num_random)
    
    # 合并原文本与生成的随机字符
    combined_text = text + random_text
    freq = count_ascii_frequency(combined_text, ascii_count=97)
    
    # 构造输出 PDF 文件名，如 待测.txt -> 待测_frequency_random_trick.pdf
    base_name = os.path.splitext(os.path.basename(filename))[0]
    output_pdf = f"{base_name}_frequency_random_trick.pdf"
    
    plot_histogram(freq, output_pdf)

if __name__ == '__main__':
    main()
