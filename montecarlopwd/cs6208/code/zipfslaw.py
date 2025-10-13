import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

def main():
    if len(sys.argv) < 2:
        print("Usage: python zipf.py <passwords.txt>")
        sys.exit(1)
    
    filename = sys.argv[1]
    
    # 1) 读取文件，每行一个密码，去掉空行和首尾空白
    with open(filename, 'r', encoding='utf-8') as f:
        passwords = [line.strip() for line in f if line.strip()]
    
    # 2) 统计密码出现频率，并按频数降序排序
    freq_counter = Counter(passwords)
    # freq_list 形如 [(pwd1, count1), (pwd2, count2), ...]
    freq_list = sorted(freq_counter.items(), key=lambda x: x[1], reverse=True)
    
    # rank: 1, 2, 3, ...
    ranks = np.arange(1, len(freq_list) + 1)
    # 对应的出现次数
    frequencies = np.array([pair[1] for pair in freq_list])
    
    # 3) 转换为 log10 空间
    log_ranks = np.log10(ranks)
    log_freqs = np.log10(frequencies)
    
    # 4) 用线性回归拟合: log(f) = A * log(r) + B
    #    其中根据 Zipf's Law: log(f_r) = log(C) - s * log(r)
    #    因此 A = -s, B = log(C).
    A, B = np.polyfit(log_ranks, log_freqs, 1)
    
    # 将 A, B 转化为 s, C
    s = -A            # s 应该是正值
    C = 10**B         # C = 10^(log(C))
    
    # 生成拟合直线的点：log(fitted) = A * log(r) + B
    fitted_line = A * log_ranks + B
    
    # 5) 画图：散点图(蓝色圆点) + 拟合直线(红色虚线)
    plt.figure(figsize=(8, 6))
    
    # (a) 原始数据散点 (log-log 坐标)
    plt.loglog(ranks, frequencies, 'bo', markersize=3, label='Data')
    
    # (b) 拟合直线
    plt.loglog(ranks, 10**fitted_line, 'r--', label=f'Fit: f(r) = {C:.2f}/r^{s:.2f}')
    
    plt.xlabel("Rank (r)")
    plt.ylabel("Frequency (f)")
    plt.title("Zipf's Law for Password Frequencies")
    plt.legend()
    plt.grid(True, which="both", ls="--")
    
    # 6) 保存为 PDF
    base_name = os.path.splitext(os.path.basename(filename))[0]
    output_pdf = f"{base_name}_zipf_weak.pdf"
    plt.savefig(output_pdf)
    plt.show()

if __name__ == '__main__':
    main()


# import sys
# import os
# import numpy as np
# import matplotlib.pyplot as plt
# from collections import Counter

# def main():
#     if len(sys.argv) < 2:
#         print("Usage: python zipf.py <passwords.txt>")
#         sys.exit(1)
    
#     filename = sys.argv[1]
    
#     # 1) 读取文件，每行一个 password，去掉空行和首尾空白
#     with open(filename, 'r', encoding='utf-8') as f:
#         passwords = [line.strip() for line in f if line.strip()]
    
#     # 2) 统计密码出现频率，并按频数降序排序
#     freq_counter = Counter(passwords)
#     freq_list = sorted(freq_counter.items(), key=lambda x: x[1], reverse=True)
    
#     # 构造 rank 和频数数组
#     ranks = np.arange(1, len(freq_list) + 1)
#     frequencies = np.array([pair[1] for pair in freq_list])
    
#     # 3) 转换到 log10 空间
#     log_ranks = np.log10(ranks)
#     log_freqs = np.log10(frequencies)
    
#     # 4) 限制只使用前 10^4 个 rank 进行拟合
#     limit = 10**4 if len(ranks) >= 10**4 else len(ranks)
#     log_ranks_fit = log_ranks[:limit]
#     log_freqs_fit = log_freqs[:limit]
    
#     # 线性回归拟合：log(f) = A*log(r) + B, 即 log(f) = log(C) - s*log(r)
#     A, B = np.polyfit(log_ranks_fit, log_freqs_fit, 1)
#     s = -A         # s 应为正值
#     C = 10**B      # C = 10^(B)
    
#     # 生成拟合直线
#     fitted_line = A * log_ranks + B

#     # 5) 绘图：散点图和拟合直线（均使用 log-log 坐标）
#     plt.figure(figsize=(8, 6))
#     plt.loglog(ranks, frequencies, 'bo', markersize=3, label='Data')
#     plt.loglog(ranks, 10**fitted_line, 'r--', label=f'Fit: f(r) = {C:.2f}/r^{s:.2f}')
#     plt.xlabel("Rank (r)")
#     plt.ylabel("Frequency (f)")
#     plt.title("Zipf's Law for Password Frequencies")
#     plt.legend()
#     plt.grid(True, which="both", ls="--")
    
#     # 6) 保存为 PDF 文件
#     base_name = os.path.splitext(os.path.basename(filename))[0]
#     output_pdf = f"{base_name}_zipf.pdf"
#     plt.savefig(output_pdf)
#     plt.show()

# if __name__ == '__main__':
#     main()

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

def main():
    if len(sys.argv) < 2:
        print("Usage: python zipf.py <passwords.txt>")
        sys.exit(1)
    
    filename = sys.argv[1]
    
    # 1) 读取文件，每行一个密码，去掉空行和首尾空白
    with open(filename, 'r', encoding='utf-8') as f:
        passwords = [line.strip() for line in f if line.strip()]
    
    # 2) 统计密码出现频率，并按频数降序排序
    freq_counter = Counter(passwords)
    freq_list = sorted(freq_counter.items(), key=lambda x: x[1], reverse=True)
    
    # 构造 rank 和频数数组
    ranks = np.arange(1, len(freq_list) + 1)
    frequencies = np.array([pair[1] for pair in freq_list])
    
    # 3) 转换为 log10 空间用于拟合
    log_ranks = np.log10(ranks)
    log_freqs = np.log10(frequencies)
    
    # 4) 用全部数据进行线性拟合: log(f) = A*log(r) + B
    A, B = np.polyfit(log_ranks, log_freqs, 1)
    s = -A            # 根据 Zipf's law：s = -A
    C = 10**B         # C = 10^B
    
    # 生成拟合直线的点（在完整数据上计算）
    fitted_line = A * log_ranks + B

    # 5) 绘制散点时进行分段采样：
    effective_cutoff = 10000  # 前10^4个点全部保留
    max_tail_points = 1000    # 后程最多绘制1000个点
    
    if len(ranks) <= effective_cutoff:
        ranks_plot = ranks
        frequencies_plot = frequencies
    else:
        # 前段全部保留
        ranks_head = ranks[:effective_cutoff]
        freq_head = frequencies[:effective_cutoff]
        # 后段均匀采样
        tail_ranks = ranks[effective_cutoff:]
        tail_freq = frequencies[effective_cutoff:]
        step = max(1, len(tail_ranks) // max_tail_points)
        tail_ranks_sample = tail_ranks[::step]
        tail_freq_sample = tail_freq[::step]
        
        # 合并前段和后段采样数据
        ranks_plot = np.concatenate([ranks_head, tail_ranks_sample])
        frequencies_plot = np.concatenate([freq_head, tail_freq_sample])
    
    # 6) 绘图：散点图（用采样后的数据）和拟合直线（用完整数据）
    plt.figure(figsize=(8, 6))
    
    # 使用 loglog 绘制散点图
    plt.loglog(ranks_plot, frequencies_plot, 'bo', markersize=3, label='Data (sampled)')
    # 绘制完整数据的拟合直线
    plt.loglog(ranks, 10**fitted_line, 'r--', label=f'Fit: f(r) = {C:.2f}/r^{s:.2f}')
    
    plt.xlabel("Rank (r)")
    plt.ylabel("Frequency (f)")
    plt.title("Zipf's Law for Password Frequencies")
    plt.legend()
    plt.grid(True, which="both", ls="--")
    
    # 7) 保存为 PDF 文件
    base_name = os.path.splitext(os.path.basename(filename))[0]
    output_pdf = f"{base_name}_zipf.pdf"
    plt.savefig(output_pdf)
    plt.show()

if __name__ == '__main__':
    main()
