#!/bin/bash

BASE=/home/guest/zsf/jyz/auto-draw
# set -aux
data=$1

if [ "$data" == "cityday" ]; then
    DATA=(
        "/home/guest/zsf/jyz/montecarlopwd/result/rock-city-backoff.csv" \
        "/home/guest/zsf/jyz/montecarlopwd/result/rock2019-cityday-4gram.csv" \
        # "/home/guest/zsf/jyz/montecarlopwd/result/rock2019-cityday-4gram-dp-1.0.csv" \
        # "/home/guest/zsf/jyz/montecarlopwd/result/rock2019-cityday-4gram-dp-new-1.0.csv" \
        # "/home/guest/zsf/jyz/montecarlopwd/result/rock2019-cityday-4gram-dp-new-0.2.csv" \
        # "/home/guest/zsf/jyz/montecarlopwd/result/rock2019-cityday-4gram-dp-new-0.01.csv" \
        # "/home/guest/zsf/jyz/montecarlopwd/result/rock2019-cityday-4gram-dp-2.0.csv" \
        "/home/guest/zsf/jyz/montecarlopwd/result/rock2019-cityday-4gram-dp-1.0.csv" \
        "/home/guest/zsf/jyz/montecarlopwd/result/rock2019-cityday-4gram-dp-0.2.csv" \
        "/home/guest/zsf/jyz/montecarlopwd/result/rock2019-cityday-4gram-dp-0.01.csv" \
        # "/home/guest/zsf/jyz/montecarlopwd/result/rock2019-rock2024-newDPbackoff10gram-1.0.csv" \
        # "/home/guest/zsf/jyz/montecarlopwd/result/rock-city-DPbackoff10gram-0.2.csv" \
        # "/home/guest/zsf/jyz/montecarlopwd/result/rock-city-DPbackoff10gram-0.01.csv" \
        # "/home/guest/zsf/jyz/montecarlopwd/result/rock2019-cityday-adaptive-4gram.csv" \
        # "/home/guest/zsf/jyz/montecarlopwd/result/rock2019-cityday-adaptive-4gram-0.33.csv" \
        "/home/guest/zsf/jyz/montecarlopwd/result/rock2019-cityday-adaptive-4gram-0.2.csv" \
        # "/home/guest/zsf/jyz/montecarlopwd/result/rock2019-cityday-DPbackoff10gram-new-1.0.csv" \
        "/home/guest/zsf/jyz/montecarlopwd/result/rock-city-DPbackoff10gram-1.0.csv" \
        # "/home/guest/zsf/jyz/montecarlopwd/result/rock-city-DPbackoff10gram-5.csv" \
        # "/home/guest/zsf/jyz/montecarlopwd/result/rock-city-DPbackoff10gram-10.csv" \
    )
elif [ "$data" == "rockyou2024" ]; then
    DATA=(
        "/home/guest/zsf/jyz/montecarlopwd/result/rock2019-rock2024-backoff.csv" \
        "/home/guest/zsf/jyz/montecarlopwd/result/rock2019-rock2024-4gram.csv" \
        # "/home/guest/zsf/jyz/montecarlopwd/result/rock2019-rock2024-4gram-dp-2.0.csv" \
        "/home/guest/zsf/jyz/montecarlopwd/result/rock2019-rock2024-4gram-dp-1.0.csv" \
        "/home/guest/zsf/jyz/montecarlopwd/result/rock2019-rock2024-4gram-dp-0.2.csv" \
        "/home/guest/zsf/jyz/montecarlopwd/result/rock2019-rock2024-4gram-dp-0.01.csv" \
        # "/home/guest/zsf/jyz/montecarlopwd/result/rock2019-rock2024-4gram-dp-new-2.0.csv" \
        # "/home/guest/zsf/jyz/montecarlopwd/result/rock2019-rock2024-4gram-dp-new-1.0.csv" \
        # "/home/guest/zsf/jyz/montecarlopwd/result/rock2019-rock2024-4gram-dp-new-0.2.csv" \
        # "/home/guest/zsf/jyz/montecarlopwd/result/rock2019-rock2024-4gram-dp-new-0.01.csv" \
        "/home/guest/zsf/jyz/montecarlopwd/result/rock2019-rock2024-adaptive-4gram-0.2.csv" \
        "/home/guest/zsf/jyz/montecarlopwd/result/rock2019-rock2024-DPbackoff10gram-1.0.csv" \
        # "/home/guest/zsf/jyz/montecarlopwd/result/rock2019-rock2024-DPbackoff10gram-1.0.csv" \
        # "/home/guest/zsf/jyz/montecarlopwd/result/rock2019-rock2024-DPbackoff10gram-0.2.csv" \
        # "/home/guest/zsf/jyz/montecarlopwd/result/rock2019-rock2024-DPbackoff10gram-0.01.csv" \
        # "/home/guest/zsf/jyz/montecarlopwd/result/rock2019-rock2024-DPbackoff10gram-5.csv" \
        # "/home/guest/zsf/jyz/montecarlopwd/result/rock2019-rock2024-DPbackoff10gram-10.csv" \
    )
else
    echo "Error: Invalid value for \$data. Only 'cityday' and 'rockyou2024' are allowed."
    exit 1
fi

# "CKL_Backoff (1.5)" \
# "CKL_Backoff (1.2)" \
# "LSTM" \

LABELS=(\
    "Backoff" \
    "Markov" \
    "DP-Markov (ε=1)" \
    "DP-Markov (ε=0.2)" \
    "DP-Markov (ε=0.01)" \
    # "newDP4gram ε=1" \
    "AdaptiveMarkov" \
    # "newDPBackoff ε=1"
    "DP-Backoff (ε=1)" \
    # "DPBackoff ε=0.2" \
    # "DPBackoff ε=0.01" \
    # "Adaptive Markov 0.33" \
    # "Adaptive Markov" \

)

COLORS=(\
    "darkorchid" \
    "black" \
    "red" \
    "royalblue" \
    "green" \
    "chocolate" \
    # "gold", \
    # "cyan", \
    # "magenta", \
    "gold" \
)

GUESS_IDX=( \
    2 \
    2 \
    2 \
    2 \
    2 \
    2 \
    2 \
)

PWD_IDX=( \
    0 \
    0 \
    0 \
    0 \
    0 \
    0 \
    0 \
)

SPLIT=( \
    "," \
    "," \
    "," \
    "," \
    "," \
    "," \
    "," \
)

WEIGHTS=( \
    1.0 \
    1.0 \
    1.0 \
    1.0 \
    1.0 \
    1.0 \
    1.0 \
)

LINE_STYLE=(\
    # "dashed" \
    "dashed" \
    "dashed" \
    "dashed" \
    "dashed" \
    "dashed" \
    "dashed" \
    "dashed" \
)

MARKERS=( \
    # "x" \
    # "o" \
    # "+" \
    none \
    none \
    none \
    none \
    none \
    none \
    none \
    # "4" \
)

TEST="/home/guest/zsf/jyz/montecarlopwd/dataset/testset/"$data"_less.txt"

JSONS=()

len=${#DATA[*]}
# len=1

echo "Generating json file..."

# min-auto


for((i=0;i<$len;i++))
do
    json_filename=$(basename "${DATA[$i]}" .csv).json
    JSONS+="${BASE}/json_no_marker/${json_filename} "
done

# for((i=0;i<$len;i++))
# do
#     json_filename=$(basename "${DATA[$i]}" .csv).json
#     dataset=${DATA[$i]}
#     label=${LABELS[$i]}
#     color=${COLORS[$i]}
#     # JSONS+="${BASE}/json/${DATA[$i]}.json "
#     # JSONS+="${BASE}/json/${json_filename} "
#     JSONS+="${BASE}/json_no_marker/${json_filename} "
#     # python3 ${BASE}/src/convert_v2.py \
#     python3 ${BASE}/src/convert_dppsm.py \
#     -l "$label" \
#     -f $dataset \
#     --upper 10000000000000000000000000000000000000000 \
#     -s "${BASE}/json_no_marker/${json_filename}" \
#     -t ${TEST} \
#     --idx-guess ${GUESS_IDX[$i]} \
#     --idx-pwd ${PWD_IDX[$i]} \
#     -c $color \
#     --gc-split "${SPLIT[$i]}" \
#     --line-style ${LINE_STYLE[$i]} \
#     --line-width ${WEIGHTS[$i]} \
#     --force-update \
#     --marker-size "5" \
#     --marker ${MARKERS[$i]} \
#     --need-sort \
#     --mark-idx 100 1000 10000 1000_00 1000_000 10_000_000 100_000_000 1000_000_000 10_000_000_000 100_000_000_000 1000_000_000_000 10_000_000_000_000 100_000_000_000_000
# done
# echo "JSONS paths: $JSONS"
# echo "Generating done"

echo "Drawing..."

# python3 ${BASE}/src/lines.py -f ${JSONS[*]}  \
python3 ${BASE}/src/lines_no_marker.py -f ${JSONS[*]}  \
--suffix .pdf                               \
-s ${BASE}/figure/DPPSM/$data"_dpbackoff_result_extend2.pdf"               \
-x 'Guesses'                               \
-y 'Cracked (%)'                            \
--show-text                                 \
--xtick-direction 'in'                      \
--ytick-direction 'in'                      \
--grid-linestyle 'dash'                     \
--xlim-high 100_000_000_000_000_000_000_000_000_000_000_000_000                  \
--xlim-low 100                           \
--ylim-high 100                              \
--ylim-low 0.00001                                \
--xlabel-size 12 \
--ylabel-size 12 \
--tick-size 12 \
--legend-loc "upper left" \
--legend-fontsize 8 \
--tight \
--show-grid \
--no-boarder "top" "right" \
--fig-size '6.4 4.0'

echo "Drawing done"
