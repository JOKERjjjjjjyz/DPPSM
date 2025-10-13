import argparse
from collections import defaultdict
import math

def argmax(chunks, key):
    index = 0
    for i, chunk in enumerate(chunks):
        if(key(chunks[index]) < key(chunk)):
            index  = i
    return index

def max_weight_match(pwd, voc, mapper, max_len):
    index = 0
    chunks = []
    while index < len(pwd):
        flag = False
        res = []
        for i in range(min(max_len,len(pwd)-index-1), 0,-1):
            if pwd[index:index+i] in voc:
                flag = True
                chunk = pwd[index:index+i]
                res.append((chunk, mapper[chunk]))
                # break
        if not flag:
            chunks.append(pwd[index])
            index += 1
        else:
            chunk_max = argmax(res, lambda x:x[1])
            chunks.append(res[chunk_max][0])
            index += len(res[chunk_max][0])
    return chunks

def max_forward_match(pwd, voc, mapper, max_len):
    index = 0
    chunks = []
    while index < len(pwd):
        flag = False
        res = []
        for i in range(min(max_len,len(pwd)-index-1), 0,-1):
            if pwd[index:index+i] in voc:
                flag = True
                chunk = pwd[index:index+i]
                chunks.append(chunk)
                index += len(chunk)
                break
        if not flag:
            chunks.append(pwd[index])
            index += 1
    return chunks

def apply_chunks(pwd, voc, mapper, max_len):
    return max_weight_match(pwd, voc, mapper, max_len)

def read_mapper(file, freq):
    mapper = defaultdict(int)
    for line in file:
        ss = line.strip("\r\n").split(" ")
        chunk = ss[0] if ss[0][-1] != '\x01' else ss[0][:-1]
        count = int(ss[1]) if freq else 1
        mapper[chunk] = max(count,mapper[chunk])
    return mapper

def create_argparser():
    cli = argparse.ArgumentParser("Apply chunks to password list(max forward matchinig)")
    cli.add_argument("-i",dest="input",required=True, help="Password list(not chunked)",type=argparse.FileType("r"))
    cli.add_argument("-m",dest="map",default=True,type=argparse.FileType("r"),help="vocabulary mapper")
    cli.add_argument("-o",dest="output",required=True,help="output file",type=argparse.FileType("w"))
    return cli.parse_args()

def main():
    args = create_argparser()
    voc_mapper = read_mapper(args.map, True)
    voc = set(filter(lambda x:len(x)>=3,voc_mapper.keys()))
    # print(voc)
    max_len = max(list(map(lambda x:len(x),voc)))
    total_chunk_count = 0
    total_pwd_count = 0
    for line in args.input:
        pwd, _, cnt, guesses, _, _ = line.strip("\r\n").split('\t')
        cnt = int(cnt)
        chunks = apply_chunks(pwd, voc, voc_mapper, max_len)
        args.output.write(f"{' '.join(chunks)}\t{cnt}\n")
        if(int(guesses)<=1000_000_000_00):
            total_chunk_count += len(chunks) * cnt
            total_pwd_count += cnt
        else:
            continue
        # print(f"{pwd}\t{chunks}")
    avgchunkcount = total_chunk_count / total_pwd_count
    args.output.write(f"{total_chunk_count} / {total_pwd_count} = {avgchunkcount:7.4f}\n")
    print(f"{total_chunk_count} / {total_pwd_count} = {avgchunkcount:7.4f}")

if __name__ == '__main__':
    main()
