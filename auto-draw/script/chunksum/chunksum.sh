#!/bin/bash

ave_len=$1
model=$2
data=$3
#chunknum=$4
train_data=rockyou


# if [ $data = "178" ] || [ $data = "youku" ];then
#     train_data=csdnn
# elif [ $data = "cityday" ] || [ $data = "neopets"] || [$data = "neocityday" ];then
#     train_data=rockyou
# fi

echo $train_data

# if [ $model = "backoff" ];then
#     CRACKED=/disk/xm/subword-markov/result_backoff/${data}1.816+.txt
# elif [ $model = "pcfg" ];then
#     CRACKED=/disk/xm/ckl-pcfg/resultless/${data}4.516+.txt
# elif [ $model = "fla" ];then
#     CRACKED=/disk/xm/LSTM/result/csv/${data}-long-subword.tsv
# fi

if [ $model = "backoff" ];then
    CRACKED=/disk/xm/subword-markov/result_backoff/${data}1.832+.txt
elif [ $model = "pcfg" ];then
    CRACKED=/disk/xm/ckl-pcfg/resultless/${data}4.532+.txt
elif [ $model = "fla" ];then
    CRACKED=/disk/xm/LSTM/result/csv/${data}-long-long-subword.tsv
fi

# if [ $model = "backoff" ];then
#     CRACKED=/disk/xm/subword-markov/result_backoff/${data}1.8.txt
# elif [ $model = "pcfg" ];then
#     CRACKED=/disk/xm/ckl-pcfg/resultless/${data}4.5.txt
# elif [ $model = "fla" ];then
#     CRACKED=/disk/xm/LSTM/result/csv/${data}-subword.tsv
# fi


# echo $CRACKED

# echo "$train_data"

BASE=/disk/xm/auto-draw
#PWD=/disk/xm/data_xm/16+//deduplicate/${data}_less.txt
#PWD=/disk/xm/data_xm/deduplicate/${data}_less.txt
PWD=/disk/xm/data_xm/32+/${data}_less.txt
CHUNKED=/disk/xm/auto-draw/result/chunksum/long/${data}_${model}_${ave_len}_chunked.txt
#CHUNKED=/disk/xm/auto-draw/result/chunksum/all/${data}_${model}_${ave_len}_chunked.txt
#OUTPUT=/disk/xm/auto-draw/result/chunksum/${data}_${model}_${ave_len}_${chunknum}.txt
echo ${CHUNKED}

# CRACKED=/disk/xm/LSTM/result/csv/${data}-long-subword.tsv
# CRACKED=/disk/xm/subword-markov/result_backoff/${data}1.816+.txt
# CRACKED=/disk/xm/ckl-pcfg/resultless/${data}4.516+.txt

# python ${BASE}/src/apply_bpe.py -i ${PWD} \
# -c /disk/xm/bpe/code/${train_data}_${ave_len}.txt \
# -o ${CHUNKED}

python ${BASE}/src/apply_weight.py -i ${PWD} \
-m /disk/xm/bpe/voc/${train_data}_${ave_len}.txt \
-o ${CHUNKED}

# set -aux

# python ${BASE}/src/chunked_impact.py --pwd ${PWD} \
# --crack ${CRACKED} \
# --chunked ${CHUNKED} \
# --chunknum ${chunknum} \
# -o ${OUTPUT}

#for calculating avg_chunk_num
set -aux
python ${BASE}/src/chunked_avg_len.py -i ${CHUNKED}