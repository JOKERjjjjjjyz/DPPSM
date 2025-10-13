from itertools import islice
import argparse
import csv
import sys
import math
import pandas as pd
# internal imports
import montecarlopwd.backoff_DP as backoff_DP
import model
import backoff
import ngram_chain
import pcfg
import numpy as np
import matplotlib.pyplot as plt
import shelve

from scipy.interpolate import interp1d

def plot_probability_threshold(minuslog_probabilities, filename="probability_threshold_plot.png"):
    """
    绘制 probability-threshold 图，保存为文件。
    :param probabilities: 密码概率的列表
    :param filename: 保存文件名
    """
    # 将概率列表排序，从小到大排列
    minuslog_probabilities.sort()

    unified_x = np.linspace(0, 200, 2000)

    # 计算 y 值，即 percentage
    y_values = np.linspace(1 / len(minuslog_probabilities), 1, len(minuslog_probabilities))

    # 对 y 进行插值，映射到统一的 x 轴上
    interp_func = interp1d(minuslog_probabilities, y_values, bounds_error=False, fill_value=(0, 1))
    interpolated_y = interp_func(unified_x)

    # 绘制图表
    plt.figure(figsize=(8, 6))
    plt.plot(unified_x, interpolated_y, marker='o', linestyle='-', color='blue')
    plt.xlabel('2^-x')
    plt.ylabel('Percentage')
    plt.xlim(0, 100)  # 设置 x 轴范围为 0 到 100
    plt.ylim(0, 1)    # 设置 y 轴范围为 0 到 1
    plt.title('Probability-Threshold Plot')
    plt.grid(True)

    # 保存图表为文件
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def plot_guess_number_graph(guess_numbers, filename="guess_number_graph.png"):
    """
    绘制 Guess Number graph，保存为文件。
    :param guess_numbers: Guess number 的列表
    :param filename: 保存文件名
    """
    # 将 Guess number 列表排序，表示猜测次数从小到大
    guess_numbers.sort()

    unified_x = np.linspace(0, 30, 1000)

    # 计算 y 值，即 percentage
    y_values = np.linspace(1 / len(guess_numbers), 1, len(guess_numbers))

    # 对 y 进行插值，映射到统一的 x 轴上
    x_values = np.log10(guess_numbers)
    interp_func = interp1d(x_values, y_values, bounds_error=False, fill_value=(0, 1))
    interpolated_y = interp_func(unified_x)

    # 绘制图表
    plt.figure(figsize=(8, 6))
    plt.plot(unified_x, interpolated_y, marker='o', linestyle='-', color='green')
    plt.xlabel('10^x')
    plt.ylabel('Percentage')
    plt.xlim(0, 30)  # 设置 x 轴范围为 0 到 100
    plt.ylim(0, 1)    # 设置 y 轴范围为 0 到 1
    plt.title('Guess Number Graph')
    plt.grid(True)

    # 保存图表为文件
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


def plot_guess_number_graph_compare(guess_numbers, model_names, filename="guess_number_comparison.png"):
    """
    绘制多条 Guess Number graph 进行比较，保存为文件。
    :param guess_numbers: 包含多个模型的 guess_number 字典
    :param model_names: 模型名称的列表
    :param filename: 保存文件名
    """
    # 创建统一的 x 轴
    unified_x = np.linspace(0, 30, 1000)

    # 设置图表
    plt.figure(figsize=(8, 6))

    # 为每个模型绘制一条曲线
    for name in model_names:
        # 获取每个模型的 guess_numbers 并排序
        guess_numbers[name].sort()
        y_values = np.linspace(1 / len(guess_numbers[name]), 1, len(guess_numbers[name]))

        # 对 y 进行插值，映射到统一的 x 轴上
        x_values = np.log10(guess_numbers[name])
        interp_func = interp1d(x_values, y_values, bounds_error=False, fill_value=(0, 1))
        interpolated_y = interp_func(unified_x)

        # 绘制曲线
        plt.plot(unified_x, interpolated_y, label=name)

    # 添加图例
    plt.legend()

    # 设置坐标轴标签和标题
    plt.xlabel('10^x')
    plt.ylabel('Percentage')
    plt.xlim(0, 30)  # 设置 x 轴范围为 0 到 100
    plt.ylim(0, 1)    # 设置 y 轴范围为 0 到 1
    plt.title('Guess Number Comparison')
    plt.grid(True)

    # 保存图表为文件
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()



def plot_probability_threshold_compare(minuslog_probabilities, model_names, filename="probability_threshold_comparison.png"):
    """
    绘制多条 probability-threshold 图进行比较，保存为文件。
    :param minuslog_probabilities: 包含多个模型的 minuslog_probabilities 字典
    :param model_names: 模型名称的列表
    :param filename: 保存文件名
    """
    # 创建统一的 x 轴
    unified_x = np.linspace(0, 100, 1000)

    # 设置图表
    plt.figure(figsize=(8, 6))

    # 为每个模型绘制一条曲线
    for name in model_names:
        # 获取每个模型的 minuslog_probabilities 并排序
        minuslog_probabilities[name].sort()
        y_values = np.linspace(1 / len(minuslog_probabilities[name]), 1, len(minuslog_probabilities[name]))

        # 对 y 进行插值，映射到统一的 x 轴上
        interp_func = interp1d(minuslog_probabilities[name], y_values, bounds_error=False, fill_value=(0, 1))
        interpolated_y = interp_func(unified_x)

        # 绘制曲线
        plt.plot(unified_x, interpolated_y, label=name)

    # 添加图例
    plt.legend()

    # 设置坐标轴标签和标题
    plt.xlabel('2^-x')
    plt.ylabel('Percentage')
    plt.xlim(0, 100)  # 设置 x 轴范围为 0 到 100
    plt.ylim(0, 1)    # 设置 y 轴范围为 0 到 1
    plt.title('Probability-Threshold Comparison')
    plt.grid(True)

    # 保存图表为文件
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


parser = argparse.ArgumentParser()
parser.add_argument('passwordfile', help='password training set')
parser.add_argument('--testfile', help='password testing set')
parser.add_argument('--epsilon', type=float, default=1.0,
                    help='total epsilon budget')
parser.add_argument('--min_ngram', type=int, default=2,
                    help='minimum n for n-grams')
parser.add_argument('--max_ngram', type=int, default=5,
                    help='maximum n for n-grams')
parser.add_argument('--backoff_threshold', type=int, default=10,
                    help='threshold for backoff')
parser.add_argument('--samplesize', type=int, default=10000,
                    help='sample size for Monte Carlo model')
args = parser.parse_args()

password_data = ngram_chain.parse_textfile(args.passwordfile)

# 进行 5-fold 交叉验证

with open(args.passwordfile, 'rt') as f:
    train_data = [w.strip('\r\n') for w in f]

with open(args.testfile, 'rt') as f:
    test_data = [w.strip('\r\n') for w in f]

models = {}
# models['DPbackoff']=backoff_DP_strongest.DP_BackoffModel(train_data, threshold=10, epsilon_total=args.epsilon)
# models['DPbackoff'].save_to_shelf()
shelf_path = f"./model/backoff_rockyou_2019/DP_backoff_model_rockyou2019_{args.epsilon}.db"
# models['DPbackoff']=backoff_DP_strongest.DP_BackoffModel(train_data, threshold=10, epsilon_total=args.epsilon,shelfname=shelf_path)
# models['DPbackoff'].save_to_shelf()
models['DPbackoff'] = backoff_DP.DP_BackoffModel.get_from_shelf(shelf_path, threshold = 10)
# shelf_path2 = f"./model/backoff_rockyou_2019/backoff_model_rockyou2019.db"
# models['backoff'] = backoff.BackoffModel.get_from_shelf(shelf_path2, threshold = 10)
# shelf_path3 = f"./model/4gram_rockyou_2019/4gram_model_rockyou2019.db"
# models['4gram'] = ngram_chain.NGramModel.get_from_shelf(shelf_path3, n = 4)
# models['backoff'] = backoff.BackoffModel(train_data, threshold = 10, shelfname=shelf_path2)
# models['backoff'].save_to_shelf()
# models['4gram'] = ngram_chain.NGramModel(train_data, n = 4, shelfname=shelf_path3)
# models['4gram'].save_to_shelf()
samples = {name: list(model.sample(args.samplesize))
           for name, model in models.items()}
estimators = {name: model.PosEstimator(sample)
              for name, sample in samples.items()}
model_names = {name: name for name, model in models.items()}
# train(train_data)
# estimate(test_data)
guess_number = {}
minuslog_probabilities = {}

for name, model in models.items():
    guess_number[name] = []
    minuslog_probabilities[name] = []

for password in test_data:
    for name, model in models.items():
        log_prob = model.logprob(password)
        estimation = estimators[name].position(log_prob)
        guess_number[name].append(estimation)
        minuslog_probabilities[name].append(log_prob)
# plot_probability_threshold(minuslog_probabilities, f"./fig/example/prob_thre_example_example.png")
# plot_guess_number_graph(guess_number, f"./fig/example/guess_number_example_example.png")

# plot_probability_threshold(minuslog_probabilities['DPbackoff'], f"./fig/rockyou2019/prob_thre_rockyou2019_cityday_DPbackoff_{args.epsilon}.png")
# plot_guess_number_graph(guess_number['DPbackoff'], f"./fig/rockyou2019/guess_number_rockyou2019_cityday_DPbackoff_{args.epsilon}.png")

# plot_probability_threshold_compare(minuslog_probabilities, model_names, f"./fig/rockyou2019/prob_thre_rockyou2019_cityday_DPbackoff_{args.epsilon}_compare.png")
# plot_guess_number_graph_compare(guess_number, model_names, f"./fig/rockyou2019/guess_number_rockyou2019_cityday_DPbackoff_{args.epsilon}_compare.png")
results = []

# 将数据转换为字典列表
for name in model_names:
    for i in range(len(guess_number[name])):
        results.append({
            'name': name,
            'log_prob': minuslog_probabilities[name][i],
            'guess_number': guess_number[name][i]
        })

# 转换为DataFrame
df = pd.DataFrame(results)

# 按照guess_number升序排列
df_sorted = df.sort_values(by='guess_number', ascending=True)

# 将结果保存成csv文件
output_file_path = './result/rock2019-cityday-DP-backoff.csv'  # 设置你的输出文件路径
df_sorted.to_csv(output_file_path, index=False)

print(f'Results saved to {output_file_path}')