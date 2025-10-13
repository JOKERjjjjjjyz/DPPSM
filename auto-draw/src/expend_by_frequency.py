from collections import defaultdict
import os,sys
import argparse
from typing import TextIO, Tuple, Callable

def read_dict(f_dict:TextIO):
    res = defaultdict(int)
    for line in f_dict:
        line = line.strip('\r\n')
        res[line] += 1
    return res

def handle(pwd_dict, pwds, output):
    for pwd in pwds:
        for _ in range(pwd_dict[pwd]):
            output.write(f"{pwd}\n")

def main():
    cli = argparse.ArgumentParser("Expend password list by frequency")
    cli.add_argument("-i", "--input", required=True, dest="input", type=argparse.FileType('r'),
                     help="password list(with other infomation)")
    cli.add_argument("-o", "--output", required=True, dest="output", type=argparse.FileType('w'),
                     help="output pwds list(one password per line)")
    cli.add_argument("-d", "--dict", required=True, dest="dict", type=argparse.FileType('r'),
                     help="password dictionary which to expend")
    args = cli.parse_args()
    pwd_dict = read_dict(args.dict)
    pwds = [line.strip('\r\n').split('\t')[0] for line in args.input]
    handle(pwd_dict, pwds, args.output)
    pass

if __name__ == '__main__':
    main()