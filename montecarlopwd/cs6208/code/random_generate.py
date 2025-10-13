import random

def generate_random_text(filename, n=100000, ascii_count=97):
    # 定义允许的字符列表：ASCII 码从 32 到 32+97-1，即 32 至 128
    allowed_chars = [chr(i) for i in range(32, 32 + ascii_count)]
    # 随机生成 n 个字符
    random_chars = random.choices(allowed_chars, k=n)
    text = ''.join(random_chars)
    # 将结果写入文件
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(text)

if __name__ == '__main__':
    generate_random_text('random.txt')
