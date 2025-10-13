#!/bin/bash

BASE=/disk/xm/auto-draw

# 4-gram, ckl-backoff, pcfg, ckl-pcfg, fla, ckl-fla
MODELS=("4-gram" "ckl-backoff" "pcfg" "ckl-pcfg" "fla" "ckl-fla")

DATASET=("cityday" "neopets" "178" "youku")

THRESHOLD=1000_000

# FLA 0 2
# ZXCVBN 0 1

model_len=${#MODELS[*]}
data_len=${#DATASET[*]}

for((i=0;i<$model_len;i++))
do
    MODEL=${MODELS[$i]}
    for((j=0;j<$data_len;j++))
    do
        data=${DATASET[$j]}
        echo "MODEL: ${MODEL}, data: ${data}, threshold: ${THRESHOLD}"
        PWDS=/disk/xm/auto-draw/result/pwdlist/${data}-${MODEL}.txt
        OUTPUT=${BASE}/result/evaluate/weak/${data}-${MODEL}.txt
        FLA=/disk/xm/lstm/new/${data}.txt
        ZXCVBN=/disk/xm/auto-draw/result/zxcvbn/${data}.txt
        MAPPER=${FLA}
        python ${BASE}/src/evaluate_password.py -i ${MAPPER} -o ${OUTPUT} -p $PWDS --pwd-index 0 --guess-index 2 --threshold ${THRESHOLD}
    done
done

# THRESHOLD=1000_000_000_000_00

# for((i=0;i<$model_len;i++))
# do
#     MODEL=${MODELS[$i]}
#     for((j=0;j<$data_len;j++))
#     do
#         data=${DATASET[$j]}
#         echo "MODEL: ${MODEL}, data: ${data}, threshold: ${THRESHOLD}"
#         PWDS=/disk/xm/auto-draw/result/pwdlist/${data}-${MODEL}.txt
#         OUTPUT=${BASE}/result/evaluate/${data}-${MODEL}.txt
#         FLA=/disk/xm/lstm/new/${data}.txt
#         ZXCVBN=/disk/xm/auto-draw/result/zxcvbn/${data}.txt
#         MAPPER=${FLA}
#         python ${BASE}/src/evaluate_password.py -i ${MAPPER} -o ${OUTPUT} -p $PWDS --pwd-index 0 --guess-index 2 --threshold ${THRESHOLD}
#     done
# done



