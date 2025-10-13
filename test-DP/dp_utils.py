"""
自定义的 Rényi 差分隐私（RDP）会计器，基于 Opacus 的实现。

功能：
- 计算子采样高斯机制（Sampled Gaussian Mechanism, SGM）的 Rényi 差分隐私（RDP）。
- 累积多轮迭代的 RDP，并将其转换为 (ε, δ)-差分隐私参数。

使用条件：
- 每轮的噪声倍数 σ 和采样率 q 均保持不变。
- 总迭代次数为 T。

依赖：
- numpy
- scipy

示例用法：
    q = 0.01
    sigma = 1.5
    T = 10000
    delta = 1e-5
    orders = np.arange(2, 1001, 0.1)

    rdp = [compute_rdp(q, sigma, alpha) for alpha in orders]
    rdp_total = [r * T for r in rdp]
    epsilon, opt_alpha = get_privacy_spent(orders, rdp_total, delta)
    print(f"Privacy Spent: ε = {epsilon}, at α = {opt_alpha}")
"""

import math
import warnings
from typing import List, Tuple, Union

import numpy as np
from scipy import special


########################
# LOG-SPACE ARITHMETIC #
########################

def _log_add(logx: float, logy: float) -> float:
    """
    在对数空间中执行加法操作，返回 log(exp(logx) + exp(logy))。

    Args:
        logx: 第一个数的对数值。
        logy: 第二个数的对数值。

    Returns:
        log(exp(logx) + exp(logy)) 的值。
    """
    a, b = min(logx, logy), max(logx, logy)
    if a == -np.inf:
        return b
    return math.log1p(math.exp(a - b)) + b  # log1p(x) = log(x + 1)


def _log_sub(logx: float, logy: float) -> float:
    """
    在对数空间中执行减法操作，返回 log(exp(logx) - exp(logy))。

    Args:
        logx: 被减数的对数值，应大于 logy。
        logy: 减数的对数值，应小于 logx。

    Returns:
        log(exp(logx) - exp(logy)) 的值。

    Raises:
        ValueError: 如果 logx < logy。
    """
    if logx < logy:
        raise ValueError("The result of subtraction must be non-negative.")
    if logy == -np.inf:
        return logx
    if logx == logy:
        return -np.inf  # 0 在对数空间中表示为 -inf
    try:
        return math.log(math.expm1(logx - logy)) + logy  # expm1(x) = exp(x) - 1
    except OverflowError:
        return logx


def _log_erfc(x: float) -> float:
    """
    计算 log(erfc(x))，用于非整数 α 的 RDP 计算。

    Args:
        x: 输入值。

    Returns:
        log(erfc(x)) 的值。
    """
    return math.log(2) + special.log_ndtr(-x * math.sqrt(2))


##########################
# RDP 计算核心函数       #
##########################

def _compute_log_a_for_int_alpha(q: float, sigma: float, alpha: int) -> float:
    """
    计算整数 α 时的 log(A_alpha)。

    Args:
        q: 采样率。
        sigma: 噪声倍数。
        alpha: Rényi 阶数（整数）。

    Returns:
        log(A_alpha) 的值。
    """
    log_a = -np.inf  # 初始化为 -inf，对应 log(0)
    for i in range(alpha + 1):
        log_coef_i = (
            math.log(special.comb(alpha, i))
            + i * math.log(q)
            + (alpha - i) * math.log(1 - q)
        )
        s = log_coef_i + (i * (i - 1)) / (2 * sigma ** 2)
        log_a = _log_add(log_a, s)
    return log_a


def _compute_log_a_for_frac_alpha(q: float, sigma: float, alpha: float) -> float:
    """
    计算非整数 α 时的 log(A_alpha)。

    Args:
        q: 采样率。
        sigma: 噪声倍数。
        alpha: Rényi 阶数（非整数）。

    Returns:
        log(A_alpha) 的值。
    """
    # 参考 Opacus 的实现，这里使用近似方法
    # 具体实现复杂，需参考相关论文和源码
    # 这里简化为对整数 α 的近似计算，实际应用中应使用更精确的方法
    # 仅供示例
    return _compute_log_a_for_int_alpha(q, sigma, int(math.floor(alpha)))


def _compute_log_a(q: float, sigma: float, alpha: float) -> float:
    """
    计算 log(A_alpha)。

    Args:
        q: 采样率。
        sigma: 噪声倍数。
        alpha: Rényi 阶数。

    Returns:
        log(A_alpha) 的值。
    """
    if float(alpha).is_integer():
        return _compute_log_a_for_int_alpha(q, sigma, int(alpha))
    else:
        return _compute_log_a_for_frac_alpha(q, sigma, alpha)


def _compute_rdp(q: float, sigma: float, alpha: float) -> float:
    """
    计算子采样高斯机制的 α 阶 RDP。

    Args:
        q: 采样率。
        sigma: 噪声倍数。
        alpha: Rényi 阶数。

    Returns:
        RDP 在 α 阶下的值。
    """
    if q == 0:
        return 0
    if sigma == 0:
        return np.inf
    if q == 1.0:
        return alpha / (2 * sigma ** 2)
    if np.isinf(alpha):
        return np.inf
    log_a = _compute_log_a(q, sigma, alpha)
    return log_a / (alpha - 1)


def compute_rdp(
    *,
    q: float,
    noise_multiplier: float,
    steps: int,
    orders: Union[List[float], float],
) -> Union[List[float], float]:
    """
    计算子采样高斯机制的 RDP 保证。

    Args:
        q: 采样率。
        noise_multiplier: 噪声倍数 σ。
        steps: 迭代次数。
        orders: Rényi 阶数列表或单个值。

    Returns:
        每个 Rényi 阶数对应的 RDP 值乘以迭代次数。
    """
    if isinstance(orders, float):
        rdp = _compute_rdp(q, noise_multiplier, orders)
    else:
        rdp = np.array([_compute_rdp(q, noise_multiplier, order) for order in orders])
    return rdp * steps


def get_privacy_spent(
    *,
    orders: Union[List[float], float],
    rdp: Union[List[float], float],
    delta: float,
) -> Tuple[float, float]:
    """
    将 RDP 转换为 (ε, δ)-差分隐私参数。

    Args:
        orders: Rényi 阶数列表或单个值。
        rdp: 每个 Rényi 阶数对应的 RDP 值列表或单个值。
        delta: 目标 δ 值。

    Returns:
        最小的 ε 和对应的最佳 α。
    """
    orders_vec = np.atleast_1d(orders)
    rdp_vec = np.atleast_1d(rdp)

    if len(orders_vec) != len(rdp_vec):
        raise ValueError(
            f"Input lists must have the same length.\n"
            f"\torders_vec = {orders_vec}\n"
            f"\trdp_vec = {rdp_vec}\n"
        )

    # 计算每个 α 下的 ε
    eps = (
        rdp_vec
        - (np.log(delta) + np.log(orders_vec)) / (orders_vec - 1)
        + np.log((orders_vec - 1) / orders_vec)
    )

    # 处理没有隐私保护的情况
    if np.isnan(eps).all():
        return np.inf, np.nan

    # 找到最小的 ε
    idx_opt = np.nanargmin(eps)  # 忽略 NaNs
    if idx_opt == 0 or idx_opt == len(eps) - 1:
        extreme = "smallest" if idx_opt == 0 else "largest"
        warnings.warn(
            f"Optimal order is the {extreme} alpha. Please consider expanding the range of alphas to get a tighter privacy bound."
        )
    return eps[idx_opt], orders_vec[idx_opt]


#############################
# 示例用法和主程序         #
#############################

def main():
    """
    主函数示例，展示如何使用自定义的 RDP 会计器。
    """
    # 示例参数
    q = 0.0001         # 采样率
    sigma = 8     # 噪声倍数
    T = 100000         # 总迭代次数
    delta = 1e-5      # 目标 δ 值

    # 定义 Rényi 阶数的范围
    # 可以根据需要调整范围和步长
    orders = np.arange(2, 1201, 0.1)  # 从 2 到 1000，步长为 0.1

    # 计算每个 α 下的 RDP
    print("Computing RDP for each alpha...")
    rdp = compute_rdp(q=q, noise_multiplier=sigma, steps=T, orders=orders)

    # 将 RDP 转换为 (ε, δ)-DP
    print("Converting RDP to (ε, δ)-DP...")
    epsilon, opt_alpha = get_privacy_spent(orders=orders, rdp=rdp, delta=delta)

    # epsilon *= 32
    print(f"Privacy Spent: ε = {epsilon:.4f}, at α = {opt_alpha:.1f}")


if __name__ == "__main__":
    main()
