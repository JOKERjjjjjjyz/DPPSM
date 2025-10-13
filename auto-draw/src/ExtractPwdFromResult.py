import argparse
import json
import sys
from collections import defaultdict
from typing import TextIO, Tuple, Callable, Iterable, Dict
import re


def read_dict(vocab:TextIO):
    mapper = defaultdict(int)
    for line in vocab:
        line = line.strip("\n\r")
        mapper[line] += 1
    return mapper

def read_set(result:TextIO, splitter, pwd_idx, guess_idx, max_guess):
    ans = set()
    for line in result:
        line = line.strip("\r\n")
        ss = line.split(splitter)
        if(float(ss[guess_idx]) < max_guess):
            ans.add(ss[pwd_idx])
    return ans

def output_pwds(mapper, ans, output):
    for pwd in ans:
        repeat = mapper[pwd]
        for i in range(repeat):
            output.write(f"{pwd}\n")

def main():
    cli = argparse.ArgumentParser("Extract password from result file")
    cli.add_argument("-i","--input",dest="input",help="crack result from model",required=True,type=argparse.FileType("r"))
    cli.add_argument("-d","--data",dest="data",help="password file(one password per line)",required=True,type=argparse.FileType("r"))
    cli.add_argument("--pwd-idx",dest="pwd_idx",help="password index in crack file",default=0,type=int)
    cli.add_argument("--guess-idx",dest="guess_idx",help="guess index in crack file",default=1,type=int)
    cli.add_argument("--splitter",dest="splitter",help="crack file splitter",default="\t")
    cli.add_argument("--max-guess",dest="max_guess",default=1000_000_000_000_00,type=int)
    cli.add_argument("-o","--output",dest="output",type=argparse.FileType("w"))
    args = cli.parse_args()
    pwds = read_dict(args.data)
    ans = read_set(args.input, args.splitter, args.pwd_idx, args.guess_idx, args.max_guess)
    output_pwds(pwds, ans, args.output)
    pass


if __name__ == '__main__':
    main()