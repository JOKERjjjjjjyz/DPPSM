# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt

# def plot_guess_number_percentage(csv_file_list, output_file_path):
#     plt.figure(figsize=(8, 6))  # 设置图像大小

#     # 对每个CSV文件进行处理
#     for csv_file_path in csv_file_list:
#         # Load the CSV file
#         df = pd.read_csv(csv_file_path, sep=',', header=None)

#         # Extract the second and third columns (probability and guess number)
#         probability_column = df.iloc[:, 1]
#         guess_number_column = pd.to_numeric(df.iloc[:, 2], errors='coerce')  # 转换为数值类型

#         # 清除NaN值
#         guess_number_column = guess_number_column.dropna()

#         # Calculate the percentage of data points with guess_number < 10^x
#         max_guess_number = np.log10(guess_number_column.max())
#         bins = np.arange(0, max_guess_number + 1, 0.1)  # Create bins for log10 scale

#         # Calculate percentage for each bin
#         percentage = [(guess_number_column < 10**x).mean() * 100 for x in bins]

#         # Plot guess number-percentage graph for this CSV file
#         plt.plot(bins, percentage, marker='o', linestyle='-', label=csv_file_path)  # 使用label区分文件

#     # 添加图例、标签和标题
#     plt.xlim(0, 40)
#     plt.xlabel(r'$x$ (where $10^x = \text{guess number}$)', fontsize=12)
#     plt.ylabel('Percentage of data points', fontsize=12)
#     plt.title('Guess Number - Percentage Plot', fontsize=14)
#     plt.grid(True)
#     plt.legend(loc='best')  # 在图上添加图例以区分不同文件的数据

#     # Save the plot to a file
#     plt.savefig(output_file_path)
#     print(f'Plot saved as {output_file_path}')

# # 示例使用
# file_list = ['./result/test-backoff.csv', './result/test-4gram.csv', './result/test-DP-backoff.csv', './result/test-lstm.csv', './result/test-ourlstm.csv','./result/test_ourlstm_40.csv','./result/test-dpsgd-4.736.csv']  # 传入你的文件列表
# output_file_path = 'test3_guess_number_percentage_plot.png'
# plot_guess_number_percentage(file_list, output_file_path)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
def plot_guess_number_percentage(csv_file_list, output_file_path):
    plt.figure(figsize=(12, 8))  # 增加图像大小以提升可读性

    # 定义线条样式和颜色循环
    line_styles = ['-', '--', '-.', ':']
    markers = ['o', 's', 'D', '^', 'v', '>', '<', 'p', '*', 'h']  # 选择不同的标记样式
    colors = plt.cm.tab10.colors  # 使用tab10调色板的颜色

    for idx, csv_file_path in enumerate(csv_file_list):
        # 加载CSV文件
        df = pd.read_csv(csv_file_path, sep=',', header=None, low_memory=False)
        # df = pd.read_csv(csv_file_path, sep=',', header=None)

        # 提取第二和第三列（概率和猜测次数）
        probability_column = df.iloc[:, 1]
        guess_number_column = pd.to_numeric(df.iloc[:, 2], errors='coerce')  # 转换为数值类型

        # 清除NaN值
        guess_number_column = guess_number_column.dropna()

        # 计算最大猜测次数的log10值，用于定义x轴范围
        max_guess_number = np.log10(guess_number_column.max())
        bins = np.arange(0, max_guess_number + 1, 0.1)  # 创建log10刻度的bins

        # 计算每个bin的百分比
        percentage = [(guess_number_column < 10**x).mean() * 100 for x in bins]

        # 选择线条样式、颜色和标记
        line_style = line_styles[idx % len(line_styles)]
        marker = markers[idx % len(markers)]
        color = colors[idx % len(colors)]

        # 绘制图表
        plt.plot(
            bins, percentage,
            linestyle=line_style,
            marker=marker,
            color=color,
            linewidth=1,          # 设置线宽为1
            markersize=4,         # 设置标记大小为4
            alpha=0.7,            # 设置透明度为0.7
            label=os.path.basename(csv_file_path)  # 使用文件名作为标签
        )

    # 添加图例、标签和标题
    plt.xlim(0, 40)
    plt.xlabel(r'$x$ (where $10^x = \text{guess number}$)', fontsize=14)
    plt.ylabel('Percentage of Data Points (%)', fontsize=14)
    plt.title('Guess Number - Percentage Plot', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='best', fontsize=10)  # 调整图例字体大小

    # 优化布局
    plt.tight_layout()

    # 保存图像
    plt.savefig(output_file_path, dpi=300)  # 提高分辨率
    plt.close()  # 关闭图像以释放内存
    print(f'Plot saved as {output_file_path}')

# 示例使用
file_list = [
    './result/results_old/rock-city-backoff.csv',
    # './result/rock2019-cityday-DPbackoff-0.1.csv',
    # './result/rock2019-cityday-DPbackoff-1.0.csv',
    # './result/rock2019-cityday-DPbackoff-2.csv',
    # './result/rock2019-cityday-DPbackoff-5.csv',
    # './result/rock2019-cityday-DPbackoff-10.csv',
    # './result/rock2019-rock2024-DPbackoff-0.1.csv',
    # './result/rock2019-rock2024-DPbackoff-1.0.csv',
    # './result/results_old/rock2019-cityday-DPbackoff-1.0-implement.csv',
    # './result/results_old/rock2019-cityday-DPbackoff-10-implement.csv',
    # './result/rock-city-DPbackoff10-0.1.csv',
    # './result/rock-city-DPbackoff10-1.0.csv',
    # './result/rock-city-DPbackoff10-2.csv',
    # './result/rock-city-DPbackoff10-5.csv',
    # './result/rock-city-DPbackoff10-10.csv',
    # './result/rock-city-DPbackoff8-1.0.csv',
    # './result/rock-city-DPbackoff6-1.0.csv',
    # './result/rock-city-newDPbackoff-1.0.csv',
    # './result/rock-city-expDPbackoff8-1.0.csv',
    # './result/rock-city-expDPbackoff10-1.0.csv',
    # './result/rock2019-rock2024-DPbackoff-5.csv',
    # './result/rock2019-rock2024-DPbackoff-2.csv',
    # './result/rock2019-rock2024-DPbackoff-10.csv',
    # './result/rock-city-pcfg.csv',
    # './result/rock-city-pcfg-0.01.csv',
    # './result/rock-city-pcfg-0.1.csv',
    # './result/rock-city-pcfg-0.2.csv',
    # './result/rock-city-pcfg-1.0.csv',
    # './result/rock-city-pcfg-2.0.csv',
    # './result/rock-city-pcfg-5.0.csv',
    # './result/rock-city-pcfg-10.0.csv',
    # './result/rock-city-dpfla-0.9701-4.csv',
    './result/rock2019-cityday-adaptive-4-1.csv',
    './result/rock2019-cityday-adaptive-4-5.csv',
    # './result/rock-city-dpfla-0.9701.csv',
    # './result/rock-city-dpfla-0.01.csv',
    # './result/rock-city-dpfla-0.2.csv',
    # './result/rock-city-dpsgd-5.csv',
    # './result/rock-city-dpsgd-0.9807.csv',
    # './result/rock-city-dpsgd-0.2197.csv',
    # './result/rock-city-abla2-0.2817.csv',
    './result/rock-city-fla.csv'
]  # 传入你的文件列表
# output_file_path = 'dpbackoff_guess_number_percentage_plot.png'
output_file_path = 'cs6208_test.png'
plot_guess_number_percentage(file_list, output_file_path)
