from collections import defaultdict
import os,sys
import argparse
from typing import TextIO, Tuple, Callable

def read_dict(f_dict):
    pwd_dict = defaultdict(int)
    for line in f_dict:
        line = line.strip('\r\n')
        pwd_dict[line] += 1
    return pwd_dict

def count(pwds, pwd_dict, max_num):
    total = sum(list(map(lambda x:x[1], pwd_dict.items())))
    cracked = 0
    for pwd,guess in pwds:
        guess = float(guess)
        if guess>max_num:
            continue
        cracked += pwd_dict[pwd]
    print(f"Total: {total}\nCracked: {cracked}\nRate: {cracked/total} ")


def main():
    cli = argparse.ArgumentParser("Password counter")
    cli.add_argument("-i", "--input", required=True, dest="input", type=argparse.FileType('r'),
                     help="password list of one model")
    cli.add_argument("-p", "--pwd", required=True, dest="password", type=argparse.FileType('r'),
                     help="password list(one password a line)")
    cli.add_argument("-n", "--number", required=False, dest="number", type=int, default=1000_000_000_000_00,
                     help="Max guessing number of model")
    cli.add_argument("--guess-idx", required=False, dest="guess_idx", type=int, default=-1,
                     help="Guess index of password guessing model")
    args = cli.parse_args()
    if args.guess_idx > 0:
        pwds = [(line.strip('\r\n').split('\t')[0],line.strip('\r\n').split('\t')[args.guess_idx]) for line in args.input]
    else:
        pwds = [(line.strip('\r\n').split('\t')[0],1) for line in args.input]
    pwd_dict = read_dict(args.password)
    count(pwds, pwd_dict, args.number)

if __name__ == '__main__':
    main()
