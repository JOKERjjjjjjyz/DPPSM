from itertools import islice
import argparse
import csv
import sys
import math
import pandas as pd
# internal imports
import backoff
import model
import ngram_chain
import pcfg
import numpy as np
import matplotlib.pyplot as plt
import shelve
import time

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
start = time.time()
# shelf_path = f"./model/4gram_rockyou_2019/4gram_model_rockyou2019.db"
# shelf_path = f"./model/backoff_rockyou_2019/backoff_model_rockyou2019.db"
# model_backoff=ngram_chain.NGramModel(train_data, 4, shelfname=shelf_path)
# model_backoff.save_to_shelf()
# models['backoff'] = backoff.BackoffModel.get_from_shelf(shelf_path, threshold = 10)
models['backoff'] = backoff.BackoffModel(train_data, threshold = 10)
input(time.time() - start)
samples = {name: list(model.sample(args.samplesize))
           for name, model in models.items()}
estimators = {name: model.PosEstimator(sample)
              for name, sample in samples.items()}
model_names = {name: name for name, model in models.items()}
# train(train_data)
# estimate(test_data)
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
        continue  # 如果密码已经处理过，跳过
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
output_file_path = './result/rock2019-rock2024-backoff.csv'  # 设置你的输出文件路径
# output_file_path = './result/rock-city-backoff.csv'
df_sorted.to_csv(output_file_path, index=False)

print(f'Results saved to {output_file_path}')