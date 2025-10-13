#!/bin/bash
# set -aux
data=$1

model=pcfg

BASE=/disk/xm/auto-draw
# FILE=${BASE}/result/intermediated/Cracked_${data}.txt
# FILE=${BASE}/result/diff-lstm/Expend-${data}.txt
FILE=${BASE}/result/pwdlist/${data}-${model}.txt
OUTPUT=${BASE}/result/pwdlist/syllable/${data}-${model}.txt

python3 ${BASE}/src/SyllablePwdFilter.py \
-o ${OUTPUT} \
-i ${FILE} \
--detail \
--syllable ${BASE}/voc/EngPattern.txt \
--pinyin ${BASE}/voc/chinese.txt
