import argparse
import os
from collections import defaultdict
from typing import Text, TextIO

def read_chunked_file(file:TextIO):
    chunked_map = {}
    for line in file:
        line = line.strip("\r\n ")
        ss = line.split(" ")
        chunks = [x if x[-1] is not "\x01" else x[:-1] for x in ss]
        chunked_map["".join(chunks)] = chunks
    return chunked_map

def read_pwd_file(file:TextIO):
    pwds = []
    for line in file:
        line = line.strip("\r\n")
        pwds.append(line)
    return pwds

def read_cracked(file: TextIO, crack_idx):
    crack_file = {}
    for line in file:
        line = line.strip("\r\n")
        ss = line.split("\t")
        crack_file[ss[0]] = float(ss[crack_idx])
    return crack_file

def create_args():
    cli = argparse.ArgumentParser("Check impact of chunk number")
    cli.add_argument("--pwd", dest="pwd", type=argparse.FileType("r"), required=True)
    cli.add_argument("--chunked", dest="chunked", type=argparse.FileType("r"),required=True)
    cli.add_argument("--crack", dest="crack", type=argparse.FileType("r"), required=True)
    cli.add_argument("-o","--output", dest="output", type=argparse.FileType("w"), required=True)
    cli.add_argument("--chunknum", dest="chunknum", type=str, default="==3")
    cli.add_argument("--crack-idx", dest="crack_idx", type=int, default="3")
    cli.add_argument("--threshold", dest="threshold", type=int, default=100_000_000_000_000)
    return cli.parse_args()

def compare_chunknum(chunknum, config):
    num = int(config[2:])
    if config[:2] == "==":
        return chunknum == num
    if config[:2] == "<=":
        return chunknum <= num
    if config[:2] == ">=":
        return chunknum >= num
    return True

def main():
    args = create_args()
    chunked_map = read_chunked_file(args.chunked)
    pwds = read_pwd_file(args.pwd)
    crack_map = read_cracked(args.crack, args.crack_idx)
    total_num = 0
    crack_num = 0
    error_num = 0
    #sum_total=0
    for pwd in pwds:
        # assert pwd in chunked_map, f"{pwd} not in chunked file"
        if pwd in chunked_map:
            chunks = chunked_map[pwd]
        else:
            chunks = [x for x in pwd]
        chunk_num = len(chunks)
        # chunk number is equal to 
        if compare_chunknum(chunk_num, args.chunknum):
            total_num += 1
            # assert pwd in crack_map, f"{pwd} not in crack file{args.crack.name}"
            if pwd not in crack_map:
                error_num += 1
                args.output.write(f"{pwd}\t{chunks} not cracked\n")
                continue
            if crack_map[pwd] < args.threshold:
                crack_num += 1
                args.output.write(f"{pwd}\t{chunks} cracked\n")
            else:
                args.output.write(f"{pwd}\t{chunks} not cracked\n")
        pass
    #sum_total += total_num
    total_num = total_num if total_num > 0 else 1
    args.output.write(f"Chunknum: {args.chunknum[2:]}\tTotal: {total_num}\tCracked: {crack_num}({crack_num/total_num})\tError number: {error_num}")
    print(f"Chunknum: {args. [2:]}\tTotal: {total_num}\tCracked: {crack_num}({crack_num/total_num})\tError number: {error_num}")

        
if __name__ == '__main__':
    main()
