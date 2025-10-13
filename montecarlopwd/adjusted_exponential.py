import DP_util
import math
import numpy as np
from tqdm import tqdm

# 参数设置
delta = 0.05
k = 3036095  # 可以根据需要调整k的值
delta_u = 62
epsilon = 10  # 确保epsilon > 1
Exp = 0
Lap = 0

for i in tqdm(range(1)):
    Exp += DP_util.sample_utility(epsilon=epsilon, delta_u=delta_u,max_utility=10000,num_bins=k)

Exp /= 20

# for i in tqdm(range(10)):
#     noise = np.random.laplace(loc=0.0, scale=delta_u/epsilon, size=k)
#     Lap += np.sum(np.abs(noise))

# Lap /= 10

# print(Exp, Lap)
print(Exp)