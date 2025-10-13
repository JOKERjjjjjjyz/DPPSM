import csv
import math

input_file = "UNCM/usacombo-rockyou2024.csv"
output_file = "UNCM/usacombo-rockyou2024-processed.csv"

with open(input_file, "r", encoding="utf-8") as fin, \
     open(output_file, "w", encoding="utf-8", newline="") as fout:

    reader = csv.reader(fin)
    writer = csv.writer(fout)

    header = next(reader)
    writer.writerow(header)  # 保留表头

    for row in reader:
        try:
            val = float(row[2])  # 第3列（index=2）
            if math.isinf(val):
                continue  # 忽略无穷大的行
        except ValueError:
            continue  # 非数字（比如 inf 字符串）也跳过
        writer.writerow(row)
