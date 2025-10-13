#AVES=(1.8 4.5)
AVES=(4.5)
MODELS=(backoff pcfg fla)
# DATAS=(neopets 178)
DATAS=(neopets cityday 178 youku)
# DATAS=(neopets)
#CHUNK_LEN=("==2" "==3" "==4" "==5" "==6" "==7" "==8" "==9" "==10" "==11" "==12" "==13" "==14" "==15")
#CHUNK_LEN=("<=5" "==6" "==7" "==8" ">=9")


len1=${#AVES[*]}
len2=${#MODELS[*]}
len3=${#DATAS[*]}
#len4=${#CHUNK_LEN[*]}

for((i=0;i<$len1;i++))
do
    for((j=0;j<$len2;j++))
    do
        for((k=0;k<$len3;k++))
        do
            echo "Model: ${MODELS[$j]}, DATA: ${DATAS[$k]}, AVE_LEN:${AVES[$i]}"
            #for((a=0;a<$len4;a++))
            #do
                ave=${AVES[$i]}
                model=${MODELS[$j]}
                data=${DATAS[$k]}
                chunk_len=${CHUNK_LEN[$a]}
                #bash /disk/xm/auto-draw/script/chunksum/chunksum.sh $ave $model $data $chunk_len
                bash /disk/xm/auto-draw/script/chunksum/chunksum.sh $ave $model $data
           # done
        done
    done
done

