from itertools import islice
import argparse
import csv
import sys
import math
# internal imports
import montecarlopwd.backoff_DP as backoff_DP
import model
import backoff
import ngram_chain
import pcfg
import numpy as np
import matplotlib.pyplot as plt
import shelve
import numpy as np
import matplotlib.pyplot as plt

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
    plt.xlabel('-log2(Probability)')
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


def plot_guess_number_graph_compare2(guess_numbers, model_names, filename="guess_number_comparison.png"):
    """
    绘制多条 Guess Number graph 进行比较，保存为文件。
    :param guess_numbers: 包含多个模型的 guess_number 字典
    :param model_names: 模型名称的列表
    :param filename: 保存文件名
    """

    plt.figure(figsize=(8, 6))

    # 为每个模型绘制一条曲线
    for name in model_names:
        guess_numbers[name].sort()

        # 计算 log2(Guess number)
        x_values = [np.log10(g) for g in guess_numbers[name]]

        # 计算 percentage，累积百分比
        n = len(guess_numbers[name])
        y_values = [(i + 1) / n for i in range(n)]
        # 获取每个模型的 guess_numbers 并排序

        # 绘制曲线
        plt.plot(x_values, y_values, label=name)

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
    # plt.figure(figsize=(8, 6))
    # for name in model_names:
    #     minuslog_probabilities[name].sort()

    #     # 计算 -log2(probability)
    #     x_values = minuslog_probabilities[name]

    #     # 计算 percentage，累积百分比
    #     n = len(minuslog_probabilities[name])
    #     y_values = [(i + 1) / n for i in range(n)]

    #     # 绘制图表
    #     plt.figure(figsize=(8, 6))
    #     plt.plot(x_values, y_values, label=name)
    # plt.legend()

    # # 设置坐标轴标签和标题
    # plt.xlabel('-log2(Probability)')
    # plt.ylabel('Percentage')
    # plt.xlim(0, 100)  # 设置 x 轴范围为 0 到 100
    # plt.ylim(0, 1)    # 设置 y 轴范围为 0 到 1
    # plt.title('Probability-Threshold Comparison')
    # plt.grid(True)

    # # 保存图表为文件
    # plt.savefig(filename, dpi=300, bbox_inches='tight')
    # plt.close()
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
    plt.xlabel('-log2(Probability)')
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


epsilon_values = [0.01, 0.1, 1, 2, 3, 5, 10]
# shelf_path = {}
# for epsilon in epsilon_values:
#     shelf_path[epsilon] = f"./model/backoff_rockyou_2019/DP_backoff_model_rockyou2019_{epsilon}.db"

# shelf_path[1] = f"./model/backoff_rockyou_2019/DP_backoff_model_rockyou2019_{epsilon}_test.db"

models = {}
for epsilon in epsilon_values:
    # 创建并存储模型
    # models[epsilon] = backoff_DP_strongest.DP_BackoffModel(
        # train_data, threshold=10, epsilon_total=epsilon, shelfname=shelf_path[epsilon])  
    models[epsilon] = backoff_DP.DP_BackoffModel(
        train_data, threshold=10, epsilon_total=epsilon)  

shelf_path2 = f"./model/backoff_rockyou_2019/backoff_model_rockyou2019.db"
models['backoff'] = backoff.BackoffModel.get_from_shelf(shelf_path2, threshold = 10)

shelf_path3 = f"./model/4gram_rockyou_2019/4gram_model_rockyou2019.db"
models['4gram'] = ngram_chain.NGramModel.get_from_shelf(shelf_path3, n = 4)

samples = {name: list(model.sample(args.samplesize))
           for name, model in models.items()}
estimators = {name: model.PosEstimator(sample)
              for name, sample in samples.items()}
model_names = {name: name for name, model in models.items()}

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

# plot_probability_threshold(minuslog_probabilities['DPbackoff'], f"./fig/rockyou2019/prob_thre_rockyou2019_cityday_DPbackoff_{args.epsilon}_test8.png")
# plot_guess_number_graph(guess_number['DPbackoff'], f"./fig/rockyou2019/guess_number_rockyou2019_cityday_DPbackoff_{args.epsilon}_test8.png")

plot_probability_threshold_compare(minuslog_probabilities, model_names, f"./fig/rockyou2019/prob_thre_rockyou2019_cityday_DPbackoff_epsilon_compare_new.png")
plot_guess_number_graph_compare(guess_number, model_names, f"./fig/rockyou2019/guess_number_rockyou2019_cityday_DPbackoff_epsilon_compare_new.png")
plot_guess_number_graph_compare2(guess_number, model_names, f"./fig/rockyou2019/guess_number_rockyou2019_cityday_DPbackoff_epsilon_compare_newnew.png")