import csv

# 设置输入输出文件名
# input_file = "UNCM/usacombo-cit0day.txt"
# output_file = "UNCM/usacombo-cit0day.csv"
# input_file = "UNCM/usacombo-rockyou2024.txt"
# output_file = "UNCM/usacombo-rockyou2024.csv"
input_file = "UNCM/usacombo-rockyou2024.txt"
output_file = "UNCM/usacombo-rockyou2024.csv"

with open(input_file, "r", encoding="utf-8") as fin, \
     open(output_file, "w", encoding="utf-8", newline="") as fout:

    writer = csv.writer(fout)
    writer.writerow(["Password", "Log_Prob", "Strength"])  # 写表头

    for line in fin:
        # 按空格分隔，允许多个空格
        parts = line.strip().split()
        if len(parts) == 3:
            password, log_prob, strength = parts
            writer.writerow([password, log_prob, strength])
