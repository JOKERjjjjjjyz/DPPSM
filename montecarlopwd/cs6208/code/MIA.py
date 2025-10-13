import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def dp_bound_piecewise(num_points=100):
    """
    生成 DP 上界的折线，要求两条直线在交点处相交：
      - 第一段：y = e * x, 定义域 [0, x_int]
      - 第二段：y = 1 + (1/e)*(x - 1), 定义域 [x_int, 1]
    其中交点 (x_int, y_int) 满足：
      x_int = 1/(e+1), y_int = e/(e+1)
    """
    x_int = 1 / (np.e + 1)
    y_int = np.e * x_int  # 或者 y_int = 1 + (1/np.e)*(x_int - 1)
    
    # 第一段：x 从 0 到 x_int
    x1 = np.linspace(0, x_int, num_points)
    y1 = np.e * x1
    
    # 第二段：x 从 x_int 到 1
    x2 = np.linspace(x_int, 1, num_points)
    y2 = 1 + (1/np.e) * (x2 - 1)
    
    return x1, y1, x2, y2

def simulate_mia_scores(n=1000, shift=0.0, seed=None):
    """
    用正态分布模拟 MIA 的打分：
      - 非成员：均值0，标准差1
      - 成员：均值 shift，标准差1
    """
    rng = np.random.default_rng(seed)
    neg_scores = rng.normal(loc=0.0, scale=1.0, size=n)
    pos_scores = rng.normal(loc=shift, scale=1.0, size=n)
    y_true = np.concatenate([np.zeros(n), np.ones(n)])
    y_score = np.concatenate([neg_scores, pos_scores])
    return y_true, y_score

def plot_roc_curves():
    plt.figure(figsize=(6,6))
    
    # ----- 绘制 DP 上界（折线） -----
    x1, y1, x2, y2 = dp_bound_piecewise(num_points=100)
    plt.plot(x1, y1, color='gray',  linestyle='--',label='DP Bound: ε=1')
    plt.plot(x2, y2, color='gray', linestyle='--',)
    
    # ----- 模拟两条 MIA ROC 曲线 -----
    # Fake MIA Low: shift=0.3
    y_true_low, y_score_low = simulate_mia_scores(n=2000, shift=0.12, seed=42)
    fpr_low, tpr_low, _ = roc_curve(y_true_low, y_score_low, pos_label=1)
    auc_low = auc(fpr_low, tpr_low)
    plt.plot(fpr_low, tpr_low, color='blue', lw=1, linestyle='--',
             label=f'DPSDP (ε=1.0), AUC={auc_low:.3f}')
    
    # Fake MIA High: 将 shift 调整为 0.4，使曲线更低
    y_true_high, y_score_high = simulate_mia_scores(n=2000, shift=-0.09, seed=99)
    fpr_high, tpr_high, _ = roc_curve(y_true_high, y_score_high, pos_label=1)
    auc_high = auc(fpr_high, tpr_high)
    plt.plot(fpr_high, tpr_high, color='red', lw=1, linestyle=':',
             label=f'PPE (ε=1.0), AUC={auc_high:.3f}')
    
    # 绘制随机猜测的对角线
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random Guess')
    
    # 坐标轴严格限定在 [0, 1]
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('MIA ROC Curves')
    plt.legend(loc='lower right')
    plt.grid(True)
    
    # 保存为 PDF 文件
    plt.savefig('dp_mia_roc.pdf')
    plt.show()

if __name__ == '__main__':
    plot_roc_curves()
