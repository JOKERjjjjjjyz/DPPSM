from itertools import islice
import argparse
import csv
import sys
import math
import pandas as pd
# internal imports
import model
import ngram_chain
import pcfg
import numpy as np
import matplotlib.pyplot as plt
import shelve
import time

def generate_top_n_using_iter(model, n, threshold=float('inf')):
    """
    Generate the top N most probable passwords using the model's __iter__ method.
    
    :param model: An instance of a Markov model (e.g., NGramModel).
    :param n: The number of top passwords to generate.
    :param threshold: A threshold to limit the cumulative log probability.
    :return: A list of tuples (logprob, password) with the top N passwords.
    """
    # Use the model's __iter__ method to generate passwords
    # The __iter__ method automatically handles the priority queue internally
    top_passwords = list(islice(model.__iter__(threshold=float('inf'), min_length=2), n))
    
    return top_passwords

import numpy as np
import matplotlib.pyplot as plt

def zipf_law_linear_fit_plot(indexed_results, filename='linear_fit_plot.png'):
    """
    进行 Zipf 定律的线性拟合并绘制图表
    :param indexed_results: (排序, 频数) 的数组
    :param filename: 保存图表的文件名
    """
    # 打印一些中间数据供检查
    for j in range(1, 9):
        print(indexed_results[100 + 100 * j])
        input("Check the printed indexed_results values and press Enter to continue...")

    # 取前 800 个数据进行拟合
    top_1000_results = indexed_results[:800]

    # 获取 rank 和 frequency 数组
    ranks = np.array([rank for rank, logprob in top_1000_results])
    frequencies = np.array([logprob for rank, logprob in top_1000_results])

    # 计算 log(rank) 和 log(frequency)
    log_ranks = np.log10(ranks)
    log_frequencies = np.log10(frequencies)

    # 使用最小二乘法进行线性拟合
    slope, intercept = np.polyfit(log_ranks, log_frequencies, 1)

    # 计算拟合的 log(frequency) 值
    log_frequencies_pred = slope * log_ranks + intercept

    # 计算 R^2 值
    ss_res = np.sum((log_frequencies - log_frequencies_pred) ** 2)
    ss_tot = np.sum((log_frequencies - np.mean(log_frequencies)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    # 输出拟合的斜率、截距和 R^2 值
    print(f"Slope: {slope}, Intercept: {intercept}, R^2: {r_squared}")

    # 绘制图表
    plt.figure(figsize=(8, 6))
    plt.scatter(log_ranks, log_frequencies, color='blue', label='Data Points')
    plt.plot(log_ranks, log_frequencies_pred, color='red', label=f'Fit Line: y = {slope:.2f}x + {intercept:.2f}')

    # 设置图表的标题、标签和网格
    plt.title('Linear Fit of log(Frequency) vs. log(Rank)')
    plt.xlabel('log(Rank)')
    plt.ylabel('log(Frequency)')
    plt.legend()
    plt.grid(True)

    # 保存图表
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

# 示例调用
# zipf_law_linear_fit_plot(indexed_results, 'linear_fit_plot.png')


parser = argparse.ArgumentParser()
parser.add_argument('passwordfile', help='password training set')
parser.add_argument('--testfile', help='password testing set')
parser.add_argument('--min_ngram', type=int, default=2,
                    help='minimum n for n-grams')
parser.add_argument('--max_ngram', type=int, default=5,
                    help='maximum n for n-grams')
parser.add_argument('--backoff_threshold', type=int, default=10,
                    help='threshold for backoff')
parser.add_argument('--samplesize', type=int, default=10000,
                    help='sample size for Monte Carlo model')
args = parser.parse_args()

with open(args.passwordfile, 'rt') as f:
    train_data = [w.strip('\r\n') for w in f]

with open(args.testfile, 'rt') as f:
    test_data = [w.strip('\r\n') for w in f]

# training = ngram_chain.parse_textfile(args.passwordfile)

# shelf_path = './model/4gram_rockyou_2019/4gram_model.db'
# shelf_path = './model/backoff_model.db'

# #下面是画password model拟合zipf's law的代码, 这个代码不适用于backoff，他无法自己generate topn pwd：
# N=10000
# number = 100
# top_passwords = generate_top_n_using_iter(model, number)
# indexed_results = [(i + 1, 2 ** (-logprob)) for i, (logprob, _) in enumerate(top_passwords)]
# plot_cumulative_frequency(indexed_results, N, 'λG-graph')
models = {}

shelf_path = f"./model/4gram_rockyou_2019/4gram_model_rockyou2019.db"
# models['4gram']=ngram_chain.NGramModel.get_from_shelf(shelf_path, n=4)
start_time = time.time()
models['4gram'] = ngram_chain.NGramModel(train_data, 4)
print(time.time()-start_time)
samples = {name: list(model.sample(args.samplesize))
           for name, model in models.items()}
estimators = {name: model.PosEstimator(sample)
              for name, sample in samples.items()}
model_names = {name: name for name, model in models.items()}

guess_number = {}
minuslog_probabilities = {}
password_list = {}

for name, model in models.items():
    guess_number[name] = []
    minuslog_probabilities[name] = []
    password_list[name] = []

processed_passwords = set()

for password in test_data:
    if password in processed_passwords:
        continue
    for name, model in models.items():
        log_prob = model.logprob(password)
        estimation = estimators[name].position(log_prob)
        guess_number[name].append(estimation)
        minuslog_probabilities[name].append(log_prob)
        password_list[name].append(password)
    processed_passwords.add(password)

results = []

# 将数据转换为字典列表
for name in model_names:
    for i in range(len(guess_number[name])):
        results.append({
            'name': password_list[name][i],
            'log_prob': minuslog_probabilities[name][i],
            'guess_number': guess_number[name][i]
        })

# 转换为DataFrame
df = pd.DataFrame(results)

# 按照guess_number升序排列
df_sorted = df.sort_values(by='guess_number', ascending=True)

# 将结果保存成csv文件
output_file_path = './result/rock2019-rock2024-4gram.csv'  # 设置你的输出文件路径
df_sorted.to_csv(output_file_path, index=False)

print(f'Results saved to {output_file_path}')

