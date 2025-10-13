BASE=/disk/xm/auto-draw

dataset=$1
MODEL=pcfg

SOURCE="/disk/xm/data_xm/deduplicate/${dataset}_less.txt"
# CRACK="/disk/xm/subword-markov/result_backoff/${dataset}1.8.txt"
# CRACK="/disk/xm/character-markov/4-gram/${dataset}.txt"
# CRACK="/disk/xm/lstm/${dataset}.txt"
# CRACK="/disk/xm/LSTM/result/txt/${dataset}.txt"
# CRACK="/disk/xm/ckl-pcfg/resultless/${dataset}4.5.txt"
CRACK="/disk/xm/semantic_pcfg/resultless/${dataset}result.txt"
GUESS_INDEX=3
MAX_GUESS=1000_000_000_000_00
OUTPUT=${BASE}/result/pwdlist/${dataset}-${MODEL}.txt

python ${BASE}/src/ExtractPwdFromResult.py \
-i ${CRACK} \
-d ${SOURCE} \
--guess-idx ${GUESS_INDEX} \
--max-guess ${MAX_GUESS} \
-o ${OUTPUT}