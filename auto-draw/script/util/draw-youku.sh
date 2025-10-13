#!/bin/bash

BASE=/disk/xm/auto-draw
# set -aux

MINAUTO_OUTPUT=${BASE}/result/minauto-youku.txt

DATA=(\
    ${MINAUTO_OUTPUT} \
    "/disk/xm/subword-markov/result_backoff/youku1.2.txt" \
    "/disk/xm/subword-markov/result_backoff/youku1.5.txt" \
    "/disk/xm/subword-markov/result_backoff/youku1.8.txt" \
    # "/disk/xm/subword-markov/result_backoff/xato2.0.txt" \
    "/disk/xm/character-markov/4-gram/youku.txt" \
    "/disk/xm/character-markov/backoff/youku.txt" \
    "/disk/xm/lstm/youku.txt" \
    "/home/xm/test/result+/rocomen_10/youku__result.txt" \
)

LABELS=(\
    "Min_auto" \
    "CKL_Backoff(1.2)" \
    "CKL_Backoff(1.5)" \
    "CKL_Backoff(1.8)" \
    "4-gram" \
    "backoff" \
    "LSTM" \
    "OMEN(4-gram)" \
)

COLORS=(\
    "black" \
    "red" \
    "orange" \
    "chocolate" \
    "royalblue" \
    "green" \
    "skyblue" \
    "darkorchid" \
    "slategrey" \
    "yellow" \
)

TEST=/disk/xm/data_xm/deduplicate/youku_less.txt

JSONS=()

len=${#DATA[*]}
# len=1

echo "Generating json file..."

# min-auto

JSON=${BASE}/config/youku.json

python3 ${BASE}/src/minauto-json.py \
    -i ${JSON} \
    -d ${TEST} \
    -o ${MINAUTO_OUTPUT} 

for((i=0;i<$len-1;i++))
do
    dataset=${DATA[$i]}
    label=${LABELS[$i]}
    color=${COLORS[$i]}
    JSONS+="${BASE}/json/${i}.json "
    # python3 ${BASE}/src/convert.py -l "$label" -f $dataset -s "${BASE}/json/${i}.json" -t ${TEST} --idx-guess 3 --idx-pwd 4 -c $color
done


i=$((${len}-1))
dataset=${DATA[$i]}
label=${LABELS[$i]}
color=${COLORS[$i]}
JSONS+="${BASE}/json/${i}.json "
# python3 ${BASE}/src/convert.py -l "$label" -f $dataset -s "${BASE}/json/${i}.json" -t ${TEST} --idx-guess 0 --idx-pwd 1 -c $color --gc-split ' : '

echo "Generating done"

echo "Drawing..."

python3 ${BASE}/src/curver.py -f ${JSONS[*]}  \
--suffix .pdf                               \
-s ${BASE}/figure/youku_result.pdf               \
-x 'Guesses'                               \
-y 'Cracked (%)'                            \
--show-text                                 \
--legend-loc 'upper left'                   \
--xtick-direction 'in'                      \
--ytick-direction 'in'                      \
--grid-linestyle 'dash'                     \
--xlim-high 100_000_000_000_000                  \
--xlim-low 100                           \
--ylim-high 95                              \
--ylim-low 0.00001                                \


echo "Drawing done"
