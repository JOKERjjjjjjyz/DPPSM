import collections
import os,sys
import argparse
from typing import TextIO, Tuple, Callable

def initCount(counts, spliter:str):
    if counts == None:
        return {}
    pwdMap = {}
    for line in counts:
        ss = line.strip("\r\n").split(spliter)
        pwd = ss[0]
        count = int(float(ss[1]))
        pwdMap[pwd] = count
    return pwdMap


def getCoverage(pwds:TextIO,spliter:str,index:int, threshold:int, pwdMap):
    total = 0
    coverage = 0
    for line in pwds:
        ss = line.strip("\r\n").split(spliter)
        try:
            guessing = int(float(ss[index]))
        except Exception:
            continue
        count = 1
        if ss[0] in pwdMap:
            count = pwdMap[ss[0]]
        if guessing < threshold:
            coverage = coverage + count
        total = total + count
    if total == 0:
        print(f'Exception: total is 0')
        return
    print(f'total: {total},\ncoverage number: {coverage},\ncoverage rate:{coverage/total}\nthreshold:{threshold}')


def main():
    cli = argparse.ArgumentParser("Password coverage counter")
    cli.add_argument("-i", "--input", required=True, dest="input", type=argparse.FileType('r'),
                     help="password list(one password a line)")
    cli.add_argument("--split", required=False, dest="split", default="\t", type=str,
                     help="how to split a line in password file, default is '\\t'")
    cli.add_argument("--strength-index", required=False, dest="index", default=1, type=int,
                     help="strength column index")
    cli.add_argument("--threshold", required=False, dest="threshold", default=100_000_000_000_000, type=int,
                    help="Guessing number threshold")
    cli.add_argument("--count-file", required=False, dest="count", default=None, type=argparse.FileType('r'), 
                    help="count file of each password. default is None")
    args = cli.parse_args()
    pwdMap = initCount(args.count, args.split)
    getCoverage(args.input, args.split, args.index, args.threshold, pwdMap)


if __name__ == '__main__':
    main()
