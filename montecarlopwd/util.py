import json

def load_ngram_flag_table(file):
    """从 JSON 文件加载 n-gram flag table"""
    with open(file, 'r') as f:
        return json.load(f)

def check_ngram_in_table(ngram, flag_table):
    """检查 n-gram 是否存在于 flag table 中"""
    return flag_table.get(ngram, False)

def generate_ngrams(password, n_min=2, n_max=34):
    """生成指定范围的 n-grams"""
    ngrams = []
    for n in range(n_min, min(n_max + 1, len(password) + 1)):
        for i in range(len(password) - n + 1):
            ngrams.append(password[i:i + n])
    return ngrams

# 主逻辑
def check_password_ngrams(password, flag_table):
    ngrams = generate_ngrams(password)  # 生成 n-grams
    discarded_count = 0  # 被丢弃的 n-gram 计数
    remaining_count = 0  # 剩余的 n-gram 计数

    # 检查每个 n-gram 是否存在于 flag table 中
    for ngram in ngrams:
        if check_ngram_in_table(ngram, flag_table):
            remaining_count += 1  # 统计剩余的 n-grams
        else:
            print(ngram)
            discarded_count += 1  # 统计被丢弃的 n-grams

    print(f"密码: {password}")
    print(f"生成的 n-grams 总数: {len(ngrams)}")
    print(f"被丢弃的 n-gram 对数: {discarded_count}")
    print(f"剩余的 n-gram 对数: {remaining_count}")

# 使用示例
flag_table = load_ngram_flag_table('ngram_flag_table.json')
print("read done")
password_to_check = "123456"  # 示例密码

check_password_ngrams(password_to_check, flag_table)

password_to_check = "qwer123456hh"  # 示例密码

check_password_ngrams(password_to_check, flag_table)
