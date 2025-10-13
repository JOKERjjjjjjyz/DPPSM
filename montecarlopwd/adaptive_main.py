from itertools import islice
import argparse
import csv
import sys
import math
import pandas as pd
# internal imports
import model
import ngram_chain_adaptive
import pcfg
import numpy as np
import matplotlib.pyplot as plt
import shelve

parser = argparse.ArgumentParser()
parser.add_argument('passwordfile', help='password training set')
parser.add_argument('--testfile', help='password testing set')
parser.add_argument('--min_ngram', type=int, default=2,
                    help='minimum n for n-grams')
parser.add_argument('--max_ngram', type=int, default=5,
                    help='maximum n for n-grams')
parser.add_argument('--backoff_threshold', type=int, default=10,
                    help='threshold for backoff')
parser.add_argument('--samplesize', type=int, default=10000,
                    help='sample size for Monte Carlo model')
args = parser.parse_args()

with open(args.passwordfile, 'rt') as f:
    train_data = [w.strip('\r\n') for w in f]

with open(args.testfile, 'rt') as f:
    test_data = [w.strip('\r\n') for w in f]

models = {}

shelf_path = f"./model/4gram_rockyou_2019/adaptive_4gram_model_0.2_rockyou2019.db"
models['4gram']=ngram_chain_adaptive.NGramModel.get_from_shelf(shelf_path, n=4)

# models['4gram'] = ngram_chain_adaptive.NGramModel(words=train_data, n=4,shelfname= shelf_path)
# models['4gram'].save_to_shelf()
samples = {name: list(model.sample(args.samplesize))
           for name, model in models.items()}
estimators = {name: model.PosEstimator(sample)
              for name, sample in samples.items()}
model_names = {name: name for name, model in models.items()}

guess_number = {}
minuslog_probabilities = {}
password_list = {}

for name, model in models.items():
    guess_number[name] = []
    minuslog_probabilities[name] = []
    password_list[name] = []

processed_passwords = set()

for password in test_data:
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
output_file_path = './result/rock2019-rock2024-adaptive-4gram-0.2.csv'  # 设置你的输出文件路径
df_sorted.to_csv(output_file_path, index=False)

print(f'Results saved to {output_file_path}')

