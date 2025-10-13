import argparse
import json
import sys
from collections import defaultdict
from typing import TextIO, Tuple, Callable, List
import random

def sample(pwds:List[str], output:TextIO, number:int):
    random.shuffle(pwds)
    size = min(number, len(pwds))
    out = pwds[:size]
    pwdCountSet = defaultdict()
    for pwd in out:
        if pwd not in pwdCountSet:
            pwdCountSet[pwd] = 1
        else:
            pwdCountSet[pwd] = pwdCountSet[pwd] + 1
    pwdWeightSet = defaultdict()
    for pwd,count in pwdCountSet.items():
        pwdWeightSet[pwd] = 0
    for pwd in pwds:
        if pwd in pwdWeightSet:
            pwdWeightSet[pwd] = pwdWeightSet[pwd] + 1
    out = [pwd for pwd in set(out)]
    out.sort(key=lambda pwd:pwdCountSet[pwd],reverse=True)
    for pwd in out:
        output.write(f'{pwd}\t{pwdCountSet[pwd]}\t{pwdWeightSet[pwd]}\n')

def main():
    cli = argparse.ArgumentParser("Sample N passwords from given password file")
    cli.add_argument("-i", "--input", required=True, dest="input", type=argparse.FileType('r'),
                     help="password list(one password a line)")
    cli.add_argument("-n", "--number", required=True, dest="number", type=int,default=10000, 
                     help="Number of password need to sample")
    cli.add_argument("-o", "--output", required=False, dest="output", type=argparse.FileType('w'), default=sys.stdout,
                     help="output file")
    args = cli.parse_args()
    pwds = [line.strip('\r\n') for line in args.input]
    sample(pwds, args.output, args.number)


if __name__ == '__main__':
    main()
