# DPPSM
python ngram_dp_main.py ./dataset/rockyou2019/rockyou_new.txt --testfile=./dataset/testset/cityday_less.txt
ngram_dp is Laplace
ngram_dp_new is exponential

python mia_data.py rockyou2019/rockyou_new.txt rockyou2019/rockyou_new_distinct.txt rockyou2019/train_set.txt rockyou2019/ground_truth_members.txt rockyou2019/ground_truth_non_members.txt --ratio 0.2