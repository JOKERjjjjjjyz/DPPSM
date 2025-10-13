# dp_utils.py
import numpy as np
import scipy.special
        
class MomentsAccountant:
    def __init__(self, total_examples, max_moment_order=32):
        """
        Initialize the Moments Accountant.
        
        Args:
            total_examples: Total number of examples in the dataset.
            max_moment_order: Maximum order of moments to compute (default is 32).
        """
        self._total_examples = total_examples
        self._max_moment_order = max_moment_order
        # Store the cumulative log moments for each moment order
        self._log_moments = np.zeros(max_moment_order)

    def _compute_log_moment(self, sigma, q, moment_order):
        if moment_order <= 1:
            return 0
        else:
            main_term = (moment_order * (moment_order - 1) * q ** 2) / (2 * sigma ** 2)
            higher_order = (moment_order ** 3 * q ** 3) / (6 * sigma ** 3)  # 示例
            return main_term + higher_order

    def accumulate_privacy_spending(self, sigma, num_examples):
        """
        Accumulate privacy spending for the given batch size.
        
        Args:
            sigma: Noise standard deviation.
            num_examples: Number of examples in the current batch.
        """
        q = num_examples / self._total_examples  # 采样概率 q
        for moment_order in range(1, self._max_moment_order + 1):
            # 累加每个阶数下的对数矩 y_i(L)
            log_moment = self._compute_log_moment(sigma, q, moment_order)
            self._log_moments[moment_order - 1] += log_moment

    def _compute_delta(self, log_moments, eps):
        """
        Compute the delta value for a given epsilon using the log moments.
        
        Args:
            log_moments: Array of log moments.
            eps: The target epsilon value.
        
        Returns:
            The corresponding delta value, using the Chernoff bound formula:
            delta = exp(sum(y_i(L)) - L * epsilon)
        """
        min_delta = 1.0
        for moment_order in range(1, len(log_moments) + 1):
            log_moment = log_moments[moment_order - 1]
            # 应用切尔诺夫界公式 delta = exp(sum(y_i(L)) - L * epsilon)
            if log_moment < moment_order * eps:
                delta = np.exp(log_moment - moment_order * eps)
                min_delta = min(min_delta, delta)
        return min_delta

    def get_privacy_spent(self, target_eps=None, target_deltas=None):
        """
        Compute (epsilon, delta) privacy guarantee using the tail bound.
        
        Args:
            target_eps: List of target epsilon values to compute corresponding delta values.
            target_deltas: List of target delta values to compute corresponding epsilon values.
            Note: Either target_eps or target_deltas must be specified.
        
        Returns:
            List of (epsilon, delta) pairs with the optimal (minimum) delta for each epsilon.
        """
        if target_eps is not None:
            eps_deltas = []
            for eps in target_eps:
                delta = self._compute_delta(self._log_moments, eps)
                eps_deltas.append((eps, delta))
            return eps_deltas
        elif target_deltas is not None:
            eps_deltas = []
            for delta in target_deltas:
                eps = self._compute_epsilon(self._log_moments, delta)
                eps_deltas.append((eps, delta))
            return eps_deltas
        else:
            raise ValueError("Either target_eps or target_deltas must be specified.")

    def _compute_epsilon(self, log_moments, delta):
        """
        Compute the epsilon value for a given delta using the log moments.
        
        Args:
            log_moments: Array of log moments.
            delta: The target delta value.
        
        Returns:
            The corresponding epsilon value.
        """
        min_eps = float('inf')
        for moment_order in range(1, len(log_moments) + 1):
            log_moment = log_moments[moment_order - 1]
            # 通过 log_moment 和 delta 计算 epsilon
            eps = (log_moment - np.log(delta)) / moment_order
            min_eps = min(min_eps, eps)
        return min_eps

class GaussianMomentsAccountant(MomentsAccountant):
    def __init__(self, total_examples, max_moment_order=32):
        super(GaussianMomentsAccountant, self).__init__(total_examples, max_moment_order)
        # Generate binomial coefficient table up to max_moment_order
        self._binomial_table = np.zeros((max_moment_order + 1, max_moment_order + 1))
        for n in range(max_moment_order + 1):
            for k in range(n + 1):
                self._binomial_table[n, k] = scipy.special.binom(n, k)

    def _differential_moments(self, sigma, s, t):
        """
        Compute 0 to t-th differential moments for Gaussian variable.

        Args:
            sigma: The noise sigma (standard deviation of the Gaussian noise).
            s: The shift parameter (0 or 1).
            t: The maximum order of moments.

        Returns:
            An array of moments from order 0 to t.
        """
        assert t <= self._max_moment_order, "The order t is out of the upper bound."

        # Binomial coefficients up to order t
        binomials = self._binomial_table[:t + 1, :t + 1]

        # Compute signs (-1)^{i - j}
        signs = np.fromfunction(lambda i, j: (-1) ** (i - j), (t + 1, t + 1))

        # Exponents for the Gaussian distribution
        exponents = np.array([j * (j + 1 - 2 * s) / (2.0 * sigma ** 2) for j in range(t + 1)])

        # Compute y[i, j] = binomial[i, j] * signs[i, j] * exp(exponents[j])
        y = binomials * signs * np.exp(exponents)

        # Sum over j to get z[i] = sum_j y[i, j]
        z = np.sum(y, axis=1)
        return z

    def _compute_log_moment(self, sigma, q, moment_order):
        """
        Compute the high moment of privacy loss for the Gaussian mechanism.

        Args:
            sigma: The noise sigma (standard deviation of the Gaussian noise).
            q: The sampling ratio (batch_size / total_examples).
            moment_order: The order of the moment to compute.

        Returns:
            The logarithm of the moment generating function.
        """
        t = moment_order
        assert t <= self._max_moment_order, "The moment order is out of the upper bound."

        # Binomial coefficients for moment_order
        binomials = self._binomial_table[t, :t + 1]

        # Compute q to the powers of [0, 1, ..., t]
        qs = q ** np.arange(t + 1)

        # Compute differential moments for s = 0 and s = 1
        moments0 = self._differential_moments(sigma, 0.0, t)
        moments1 = self._differential_moments(sigma, 1.0, t)

        # Compute terms for s = 0 and s = 1
        term0 = np.sum(binomials * qs * moments0)
        term1 = np.sum(binomials * qs * moments1)

        # Combine terms to get the moment
        moment = q * term0 + (1.0 - q) * term1

        # Return the logarithm of the moment
        return np.log(moment)

def sample_utility(epsilon, delta_u, max_utility):
    """
    通过指数机制对效用值进行采样。
    :param epsilon: 差分隐私参数 epsilon
    :param delta_u: 效用函数的敏感度 Delta_u
    :param max_utility: 效用值的上限，假设 utility 是从 0 到 max_utility 的整数值
    :return: 采样到的效用值
    """
    # 计算每个效用值的概率，范围从 0 到 max_utility
    utilities = np.arange(0, max_utility + 1)
    probabilities = np.exp(-epsilon * utilities / (2 * delta_u))
    
    # 将概率归一化
    probabilities /= np.sum(probabilities)
    
    # 根据概率分布采样效用值
    sampled_utility = np.random.choice(utilities, p=probabilities)
    
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
    while remaining_utility > 0:
        bin_idx = random.randint(0, num_bins - 1)  # 随机选择一个 bin
        increment = min(remaining_utility, 1)  # 每次增加 1，直到用完剩余的效用值
        histogram_sample[bin_idx] += increment
        remaining_utility -= increment

    for i in range(num_bins):
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
    sampled_utility = sample_utility(epsilon, delta_u, max_utility)
    
    # 生成一个新的直方图实例，基于采样到的效用值
    histogram_sample = generate_histogram_sample(sampled_utility, num_bins)
    
    # 将生成的直方图和已有的直方图相加，形成新的直方图
    new_histogram = np.array(state_histogram) + histogram_sample
    
    new_histogram = np.maximum(new_histogram, 1)

    return new_histogram

def AboveThreshold(count, threshold_DP, epsilon = 0.1, delta_u = 66):
    b = delta_u / epsilon
    Lap_count = np.random.laplace(0, 4*b, size=1).item()
    # Lap_threshold = np.random.laplace(0, 2*b, size=1).item()
    if (count + Lap_count >= threshold_DP):
        return True
    else:
        return False

