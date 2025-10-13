#!/bin/bash

BASE=/home/guest/zsf/jyz/auto-draw
# set -aux
data=$1


if [ "$data" == "cityday" ]; then
    DATA=(
        "/home/guest/zsf/jyz/montecarlopwd/result/nn_distinct/rock-city-dpfla-0.9701.csv" \
        "/home/guest/zsf/jyz/montecarlopwd/result/nn_distinct/rock-city-dpfla-0.2.csv" \
        "/home/guest/zsf/jyz/montecarlopwd/result/nn_distinct/rock-city-dpfla-0.01.csv" \
        # "/home/xm/test/crack_file/cityday_crack.txt" \
    )
elif [ "$data" == "rockyou2024" ]; then
    DATA=(
        "/home/guest/zsf/jyz/montecarlopwd/result/nn_distinct/rock2019-rock2024-dpfla-0.9701.csv" \
        "/home/guest/zsf/jyz/montecarlopwd/result/nn_distinct/rock2019-rock2024-dpfla-0.2.csv" \
        "/home/guest/zsf/jyz/montecarlopwd/result/nn_distinct/rock2019-rock2024-dpfla-0.01.csv" \
        # "/home/xm/test/crack_file/rockyou2024_crack.txt" \
    )
else
    echo "Error: Invalid value for \$data. Only 'cityday' and 'rockyou2024' are allowed."
    exit 1
fi

# "CKL_Backoff (1.5)" \
# "CKL_Backoff (1.2)" \
# "LSTM" \

LABELS=(\
    "ε=1" \
    "ε=0.2" \
    "ε=0.01" \
    # "Backoff" \
    # "OMEN (4-gram)" \
)

COLORS=(\
    "black" \
    "red" \
    "royalblue" \
    # "green" \
    # "chocolate" \
    # "darkorchid" \
)

GUESS_IDX=( \
    2 \
    2 \
    2 \
    # 3 \
    # 2 \
)

PWD_IDX=( \
    0 \
    0 \
    0 \
    # 0 \
    # 0 \
)

SPLIT=( \
    "," \
    "," \
    "," \
    # "\t" \
    # " : " \
)

WEIGHTS=( \
    0.5 \
    0.5 \
    0.5 \
    # 1.5 \
    # 1.5 \
)

LINE_STYLE=(\
    # "solid" \
    # "solid" \
    # "solid" \
    "dashed" \
    "dashed" \
    "dashed" \
    # "dashed" \
)

MARKERS=( \
    # "x" \
    # "o" \
    # "+" \
    none \
    none \
    none \
    # "1" \
    # "2" \
    # "3" \
    # "4" \
)

TEST="/home/guest/zsf/jyz/montecarlopwd/dataset/testset/"$data"_less.txt"

JSONS=()

len=${#DATA[*]}
# len=1

echo "Generating json file..."

# min-auto


# for((i=0;i<$len;i++))
# do
#     json_filename=$(basename "${DATA[$i]}" .csv).json
#     JSONS+="${BASE}/json/${json_filename} "
# done

for((i=0;i<$len;i++))
do
    json_filename=$(basename "${DATA[$i]}" .csv).json
    dataset=${DATA[$i]}
    label=${LABELS[$i]}
    color=${COLORS[$i]}
    # JSONS+="${BASE}/json/${DATA[$i]}.json "
    # JSONS+="${BASE}/json/${json_filename} "
    JSONS+="${BASE}/json_no_marker/${json_filename} "
    # python3 ${BASE}/src/convert_v2.py \
    python3 ${BASE}/src/convert_dppsm.py \
    -l "$label" \
    -f $dataset \
    -s "${BASE}/json_no_marker/${json_filename}" \
    -t ${TEST} \
    --idx-guess ${GUESS_IDX[$i]} \
    --idx-pwd ${PWD_IDX[$i]} \
    -c $color \
    --gc-split "${SPLIT[$i]}" \
    --line-style ${LINE_STYLE[$i]} \
    --line-width ${WEIGHTS[$i]} \
    --force-update \
    --marker-size "5" \
    --marker ${MARKERS[$i]} \
    --need-sort \
    --mark-idx 100 1000 10000 1000_00 1000_000 10_000_000 100_000_000 1000_000_000 10_000_000_000 100_000_000_000 1000_000_000_000 10_000_000_000_000 100_000_000_000_000
done
echo "JSONS paths: $JSONS"
echo "Generating done"

echo "Drawing..."

# python3 ${BASE}/src/lines.py -f ${JSONS[*]}  \
python3 ${BASE}/src/lines_no_marker.py -f ${JSONS[*]}  \
--suffix .pdf                               \
-s ${BASE}/figure/DPPSM/$data"_dpfla_epsilon_result.pdf"               \
-x 'Guesses'                               \
-y 'Cracked (%)'                            \
--show-text                                 \
--legend-loc 'none'                   \
--xtick-direction 'in'                      \
--ytick-direction 'in'                      \
--grid-linestyle 'dash'                     \
--xlim-high 100_000_000_000_000                  \
--xlim-low 100                           \
--ylim-high 80                              \
--ylim-low 0.00001                                \
--xlabel-size 20 \
--ylabel-size 20 \
--tick-size 20 \
--legend-fontsize 20 \
--tight \
--show-grid \
--no-boarder "top" "right" \
--fig-size '6.4 4.0'

echo "Drawing done"
