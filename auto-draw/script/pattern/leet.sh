#!/bin/bash
# set -aux
dataset=$1

model=ckl-pcfg

BASE=/disk/xm/auto-draw
FILE=${BASE}/result/pwdlist/${dataset}-${model}.txt

OUTPUT=${BASE}/result/pwdlist/leet/${dataset}-${model}.txt
MODEL=${BASE}/result/leet.pkl
DATASET=/disk/yjt/working_data/roccsdnn_new.txt
# DATASET=/disk/yjt/working_data/roccsdnn_test.txt


python3 ${BASE}/src/leet_v2.py \
-c ${DATASET} \
-p ${FILE} \
-o ${OUTPUT} \
-l ${MODEL}
# -m ${MODEL} 
# -l ${MODEL}


