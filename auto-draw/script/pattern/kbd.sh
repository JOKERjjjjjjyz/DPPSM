#!/bin/bash
# set -aux
data=$1

model=ckl-pcfg

BASE=/disk/xm/auto-draw
# FILE=${BASE}/result/intermediated/Cracked_${data}.txt
# FILE=${BASE}/result/diff-lstm/Expend-${data}.txt
FILE=${BASE}/result/pwdlist/${data}-${model}.txt
OUTPUT=${BASE}/result/pwdlist/keyboard/${data}-${model}.txt

python3 ${BASE}/src/kbd.py \
-o ${OUTPUT} \
-p ${FILE}
