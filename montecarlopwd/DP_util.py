import numpy as np
import random
import math
from tqdm import tqdm
from scipy.special import comb

def sample_utility(epsilon, delta_u, max_utility, num_bins):

    # 计算每个效用值的概率，范围从 0 到 max_utility
    epsilon1 = 100 * epsilon / num_bins
    # epsilon1 = epsilon / (1+math.sqrt(num_bins/math.log(1/0.05)))
    interval = int(math.log(1/0.05) * delta_u / epsilon1)
    epsilon2 = epsilon - epsilon1
    k = int(delta_u / epsilon2)
    start = k * num_bins
    # print(start,interval)
    utilities = np.arange(start, start + interval)
    prob_A = k * np.exp(-epsilon1 * k * num_bins / (2 * delta_u))
    probabilities = np.exp(-epsilon1 * utilities / (2 * delta_u))
    prob_B = np.sum(probabilities)
    # print(prob_A,prob_B)
    # 将概率归一化
    # input(prob_A)
    if random.random() < prob_A / (prob_A + prob_B):
        # 以 A / (A + B) 的概率执行的代码
        # print("Executed with A/(A + B) probability")
        print("here")
        comb_list = np.array([math.comb(i + num_bins - 1, num_bins - 1) for i in range(1, k * num_bins)])
        probabilities = comb_list / comb_list.sum()
        utilities_A = np.arange(1, k * num_bins)
        sampled_utility = np.random.choice(utilities_A, p=probabilities)
    else:
        # 其余情况下执行的代码
        # print("Executed with 1 - A/(A + B) probability")
        probabilities /= prob_B
        # input(prob_B)
        sampled_utility = np.random.choice(utilities, p=probabilities)
    # 根据概率分布采样效用值

    
    return sampled_utility

def generate_histogram_sample(sampled_utility, num_bins):
    """
    根据采样的效用值，均匀地从可能的实例中生成直方图。
    :param sampled_utility: 采样到的效用值
    :param num_bins: 直方图的 bin 数目（即直方图 x 轴的总可能数）
    :return: 生成的直方图
    """
    # 初始化一个全为0的直方图
    histogram_sample = np.zeros(num_bins)
    
    # 将效用值分布在直方图的随机 bin 上，均匀采样
    remaining_utility = sampled_utility
    # while remaining_utility > 0:
    #     bin_idx = random.randint(0, num_bins - 1)  # 随机选择一个 bin
    #     increment = min(remaining_utility, 1)  # 每次增加 1，直到用完剩余的效用值
    #     histogram_sample[bin_idx] += increment
    #     remaining_utility -= increment

    # for _ in tqdm(range(remaining_utility), desc="Distributing balls into bins"):
    #     bin_idx = random.randint(0, num_bins - 1)  # 随机选择一个 bin
    #     histogram_sample[bin_idx] += 1  # 每次增加 1
    start = time.time()
    cut_points = np.random.randint(0, remaining_utility + 1, size=num_bins - 1)
    cut_points = np.sort(cut_points)
    cut_points = np.concatenate(([0], cut_points, [remaining_utility]))
    histogram_sample = cut_points[1:] - cut_points[:-1]
    end = time.time()
    print(f"Split {remaining_utility} into {num_bins} using {end-start}seconds")

    # for i in range(num_bins):
    #     if random.random() < 0.5:
    #         histogram_sample[i] = -histogram_sample[i]

    for i in tqdm(range(num_bins), desc="Flipping Histogram Values"):
        if random.random() < 0.5:
            histogram_sample[i] = -histogram_sample[i]

    return histogram_sample

def apply_exponential_mechanism(state_histogram, epsilon, delta_u=32, max_utility=10000):
    """
    使用指数机制生成新的直方图，基于已有的直方图。
    :param state_histogram: 已有的直方图 (list of counts)
    :param epsilon: 差分隐私参数 epsilon
    :param delta_u: 效用函数的敏感度 Delta_u
    :param max_utility: 效用值的最大值
    :return: 新的直方图
    """
    num_bins = len(state_histogram)  # 直方图的总 bin 数
    
    # 通过指数机制对效用进行采样
    sampled_utility = sample_utility(epsilon, delta_u, max_utility, num_bins)
    
    # 生成一个新的直方图实例，基于采样到的效用值
    histogram_sample = generate_histogram_sample(sampled_utility, num_bins)
    
    # 将生成的直方图和已有的直方图相加，形成新的直方图
    new_histogram = np.array(state_histogram) + histogram_sample
    
    new_histogram = np.maximum(new_histogram, 1)

    return new_histogram

def AboveThreshold(count, threshold_DP, epsilon = 1.0, delta_u = 32):
    b = delta_u / epsilon
    Lap_count = np.random.laplace(0, 4*b, size=1).item()
    # Lap_threshold = np.random.laplace(0, 2*b, size=1).item()
    if (count + Lap_count >= threshold_DP):
        return True
    else:
        return False

import numpy as np
import time

def method_multinomial(remaining_utility, num_bins):
    start = time.time()
    histogram_sample = np.random.multinomial(remaining_utility, [1/num_bins]*num_bins)
    end = time.time()
    return end - start

def method_split_points(remaining_utility, num_bins):
    start = time.time()
    cut_points = np.random.randint(0, remaining_utility + 1, size=num_bins - 1)
    cut_points = np.sort(cut_points)
    cut_points = np.concatenate(([0], cut_points, [remaining_utility]))
    histogram_sample = cut_points[1:] - cut_points[:-1]
    end = time.time()
    return end - start

# remaining_utility = 10**11
# num_bins = 10**10

# # time_multinomial = method_multinomial(remaining_utility, num_bins)
# # print(f"Multinomial Time: {time_multinomial:.4f} seconds")
# time_split_points = method_split_points(remaining_utility, num_bins)
# print(f"Split Points Time: {time_split_points:.4f} seconds")
import numpy as np
from scipy.optimize import minimize

def compute_L(gamma, k=5000000):
    """
    计算表达式 L 的值。

    参数:
    gamma (float): 变量 γ
    k (int, optional): 常数 k，默认为 32,000,000

    返回:
    float: 表达式 L 的值
    """
    if gamma <= 0:
        return np.inf  # 避免 γ <= 0 的情况
    term1 = 4 * k * gamma
    term2 = (6.1 + 1 / (5 * k * gamma)) / gamma
    L = term1 + term2
    return L

def find_optimal_gamma(k=5000000):
    """
    寻找使 L 最小的 γ 值。

    参数:
    k (int, optional): 常数 k，默认为 32,000,000

    返回:
    result.x (float): 最优的 γ 值
    result.fun (float): 最小的 L 值
    """
    # 初始猜测值
    initial_gamma = 1e-4

    # 使用 minimize 函数进行优化
    result = minimize(
        compute_L,
        x0=initial_gamma,
        args=(k,),
        method='L-BFGS-B',
        bounds=[(1e-12, None)]  # γ 必须大于 0
    )

    return result.x[0], result.fun

if __name__ == "__main__":
    optimal_gamma, minimum_L = find_optimal_gamma()
    print(f"Optimal γ: {optimal_gamma}")
    print(f"Minimum L: {minimum_L}")
    print(compute_L(gamma=0.2))
    print(32000000/(4*math.log(2)*62*62))