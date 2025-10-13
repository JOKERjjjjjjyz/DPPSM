# from scipy.stats import norm
# import math
# delta = 0.05
# quantile = norm.ppf(1-delta)  # 0.95分位数

# k = 527669751
# delta_u = 62
# epsilon = 0.9
# Lap = (delta_u / epsilon) * (k + math.sqrt(k/delta))
# # if epsilon > 1:
# #     epsilon1 = 0.001
# #     epsilon2 = epsilon - epsilon1
# # else:
# #     epsilon1 = epsilon / (0.000001 * k)
# #     epsilon2 = epsilon - epsilon1
# epsilon1 = epsilon / (1 + math.sqrt(k / 3))
# epsilon2 = epsilon - epsilon1
# Exp = math.log(1/delta) * delta_u / epsilon1 + k * delta_u / epsilon2 

# print(Lap, Exp)


# import math

# # 参数设置
# delta = 0.05
# k = 527669751  # 可以根据需要调整k的值
# delta_u = 62
# epsilon = 0.9  # 确保epsilon > 1

# # 计算Laplace机制的误差
# Lap = (delta_u / epsilon) * (k + math.sqrt(k / delta))

# # 计算A和B
# A = math.log(1 / delta) * delta_u  # A = ln(1/δ) * Δu
# B = k * delta_u                    # B = k * Δu

# # 计算C
# C = B / A  # C = B / A

# # 计算√C
# sqrt_C = math.sqrt(C)

# # 计算最优的ε1占总ε的比例r
# r_numerator = 1 - sqrt_C
# r_denominator = 1 - C

# # 检查分母是否为零，避免除以零的错误
# if r_denominator == 0:
#     print("无法计算r，分母为零。请检查参数设置。")
# else:
#     r = r_numerator / r_denominator

#     # 确保r在(0,1)范围内
#     if r <= 0 or r >= 1:
#         print("计算得到的r不在(0,1)范围内，无法继续计算。请调整参数。")
#     else:
#         # 计算ε1和ε2
#         epsilon1 = r * epsilon
#         epsilon2 = epsilon - epsilon1

#         # 检查ε1和ε2是否大于零
#         if epsilon1 <= 0 or epsilon2 <= 0:
#             print("计算得到的ε1或ε2不大于零，无法继续计算。")
#         else:
#             # 计算Exp机制的误差
#             Exp = (A / epsilon1) + (B / epsilon2)

#             # 输出结果
#             print(f"ε1 = {epsilon1}")
#             print(f"ε2 = {epsilon2}")
#             print(f"Laplace机制的误差（Lap）: {Lap}")
#             print(f"指数机制的误差（Exp）: {Exp}")

#             # 比较Exp和Lap的误差
#             if Exp < Lap:
#                 print("在此参数配置下，Exp的误差小于Lap的误差。")
#             else:
#                 print("在此参数配置下，Exp的误差不小于Lap的误差。")

import math
import matplotlib.pyplot as plt
import numpy as np

# 参数设置
delta = 0.05
delta_u = 62
epsilon = 0.9  # 确保epsilon > 1

# 计算函数
def calculate_error_bound_decrease(k, delta, delta_u, epsilon):
    Lap = (delta_u / epsilon) * (k + math.sqrt(k / delta))
    A = math.log(1 / delta) * delta_u
    B = k * delta_u
    C = B / A
    sqrt_C = math.sqrt(C)
    
    r_numerator = 1 - sqrt_C
    r_denominator = 1 - C
    
    if r_denominator == 0:
        return None
    r = r_numerator / r_denominator

    if r <= 0 or r >= 1:
        return None
    
    epsilon1 = r * epsilon
    epsilon2 = epsilon - epsilon1

    if epsilon1 <= 0 or epsilon2 <= 0:
        return None

    Exp = (A / epsilon1) + (B / epsilon2)
    error_bound_decrease = (Lap - Exp) / Lap
    return error_bound_decrease

# 生成数据
k_values = np.logspace(2, 14, num=100, base=10)  # 从10^2到10^14的k值
x_values = np.log10(k_values)  # 横坐标取log10(k)
y_values = [calculate_error_bound_decrease(k, delta, delta_u, epsilon) for k in k_values]

# 过滤None值
x_values = [x for x, y in zip(x_values, y_values) if y is not None]
y_values = [y for y in y_values if y is not None]

# 绘图
plt.figure(figsize=(10, 6))
plt.plot(x_values, y_values, label="Error Bound Decrease")
plt.xlabel("log10(k)")
plt.ylabel("Error Bound Decrease (Lap - Exp) / Lap")
plt.title("Error Bound Decrease")
plt.legend()
plt.grid(True)
plt.savefig("error_bound_decrease.png", format="png", dpi=300)

import math
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

# 参数设置
delta = 0.05
delta_u = 62
k = 10**4  # 固定 k = 10^8
epsilon_values_small = [0.001, 0.01, 0.1]  # 小 ε 值
epsilon_values_large = [1, 2, 3, 4, 5, 6]  # 大 ε 值

# 计算误差边界减少量的函数
def calculate_error_bound_decrease(k, delta, delta_u, epsilon):
    Lap = (delta_u / epsilon) * (k + math.sqrt(k / delta))
    A = math.log(1 / delta) * delta_u
    B = k * delta_u
    C = B / A
    sqrt_C = math.sqrt(C)
    
    r_numerator = 1 - sqrt_C
    r_denominator = 1 - C
    
    if r_denominator == 0:
        return None
    r = r_numerator / r_denominator

    if r <= 0 or r >= 1:
        return None
    
    epsilon1 = r * epsilon
    epsilon2 = epsilon - epsilon1

    if epsilon1 <= 0 or epsilon2 <= 0:
        return None

    Exp = (A / epsilon1) + (B / epsilon2)
    error_bound_decrease = (Lap - Exp) / Lap
    return error_bound_decrease

# 生成数据
y_values_small = [calculate_error_bound_decrease(k, delta, delta_u, epsilon) for epsilon in epsilon_values_small]
y_values_large = [calculate_error_bound_decrease(k, delta, delta_u, epsilon) for epsilon in epsilon_values_large]

# 绘图
fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(12, 6), gridspec_kw={'width_ratios': [1, 2]})

# 绘制小 epsilon 部分
ax1.plot(epsilon_values_small, y_values_small, marker='o', linestyle='-', label="Error Bound Decrease")
ax1.set_xscale('log')
ax1.set_xlabel("ε (small values)")
ax1.set_ylabel("Error Bound Decrease (Lap - Exp) / Lap")
ax1.grid(True)

# 绘制大 epsilon 部分
ax2.plot(epsilon_values_large, y_values_large, marker='o', linestyle='-', label="Error Bound Decrease")
ax2.set_xticks(epsilon_values_large)
ax2.set_xlabel("ε (large values)")
ax2.grid(True)

# 设置图表标题
fig.suptitle("Error Bound Decrease for fixed k=10^8")

# 调整布局
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("error_bound_decrease2.png", format="png", dpi=300)

import math
import matplotlib.pyplot as plt
import numpy as np

# 参数设置
delta = 0.05
delta_u = 62
k = 10**8  # 固定 k 值

# 计算函数
def calculate_error_bound_decrease_fixed_k(delta, delta_u, epsilon, k):
    Lap = (delta_u / epsilon) * (k + math.sqrt(k / delta))
    A = math.log(1 / delta) * delta_u
    B = k * delta_u
    C = B / A
    sqrt_C = math.sqrt(C)
    
    r_numerator = 1 - sqrt_C
    r_denominator = 1 - C
    
    if r_denominator == 0:
        return None
    r = r_numerator / r_denominator

    if r <= 0 or r >= 1:
        return None
    
    epsilon1 = r * epsilon
    epsilon2 = epsilon - epsilon1

    if epsilon1 <= 0 or epsilon2 <= 0:
        return None

    Exp = (A / epsilon1) + (B / epsilon2)
    error_bound_decrease = (Lap - Exp) / Lap
    return error_bound_decrease

# 生成数据
epsilon_values = range(1, 11)  # ε从1到10
y_values = [calculate_error_bound_decrease_fixed_k(delta, delta_u, epsilon, k) for epsilon in epsilon_values]

# 过滤None值
epsilon_values = [e for e, y in zip(epsilon_values, y_values) if y is not None]
y_values = [y for y in y_values if y is not None]

# 绘图
plt.figure(figsize=(10, 6))
plt.plot(epsilon_values, y_values, marker='o', label="Error Bound Decrease")
plt.xlabel("ε")
plt.ylabel("Error Bound Decrease (Lap - Exp) / Lap")
plt.title("Error Bound Decrease for Fixed k = 10^8 with Varying ε")
plt.legend()
plt.grid(True)
plt.savefig("error_bound_decrease3.png", format="png", dpi=300)

import math
import matplotlib.pyplot as plt
import numpy as np

# 参数设置
delta = 0.05
delta_u = 62
epsilon = 0.9  # 请确认是否应大于1

# 计算函数
def calculate_error_bound_decrease(k, delta, delta_u, epsilon):
    try:
        Lap = (delta_u / epsilon) * (k + math.sqrt(k / delta))
        A = math.log(1 / delta) * delta_u
        B = k * delta_u
        C = B / A
        sqrt_C = math.sqrt(C)

        r_numerator = 1 - sqrt_C
        r_denominator = 1 - C

        if r_denominator == 0:
            return None
        r = r_numerator / r_denominator

        if r <= 0 or r >= 1:
            return None

        epsilon1 = r * epsilon
        epsilon2 = epsilon - epsilon1

        if epsilon1 <= 0 or epsilon2 <= 0:
            return None

        Exp = (A / epsilon1) + (B / epsilon2)
        error_bound_decrease = (Lap - Exp) / Lap
        return error_bound_decrease
    except (ValueError, ZeroDivisionError):
        return None

# 生成数据
k_values = np.logspace(2, 14, num=100, base=10)  # 从10^2到10^14的k值
x_values = np.log10(k_values)  # 横坐标取log10(k)
y_values = [calculate_error_bound_decrease(k, delta, delta_u, epsilon) for k in k_values]

# 过滤None值
filtered_x = []
filtered_y = []
for x, y in zip(x_values, y_values):
    if y is not None:
        filtered_x.append(x)
        filtered_y.append(y)

# 标记整数点
integer_k = [10**i for i in range(2, 15)]
integer_x = [math.log10(k) for k in integer_k]
integer_y = [calculate_error_bound_decrease(k, delta, delta_u, epsilon) for k in integer_k]

# 过滤计算结果为None的整数点
integer_plot_x = []
integer_plot_y = []
for x, y in zip(integer_x, integer_y):
    if y is not None:
        integer_plot_x.append(x)
        integer_plot_y.append(y)

# 绘图
plt.figure(figsize=(10, 6))
plt.plot(filtered_x, filtered_y, label="Error Bound Decrease")  # 连续曲线
plt.scatter(integer_plot_x, integer_plot_y, color="black", marker='x', label="Integer Points 2-14")  # 只在特定整数点上标记
plt.xlabel("log10(k)")
plt.ylabel("Error Bound Decrease (Lap - Exp) / Lap")
plt.title("Error Bound Decrease")
plt.legend()
plt.grid(True)
plt.savefig("error_bound_decrease4.png", format="png", dpi=300)

