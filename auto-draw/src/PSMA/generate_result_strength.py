import argparse
import json
import sys
from collections import defaultdict
from typing import TextIO, Tuple, Callable, List
import random

def initCountFile(f:TextIO):
    return [line.strip('\r\n').split('\t') for line in f]

def initMeterResult(f:TextIO, index:int, sample_pwds:list):
    pwds = defaultdict()
    for line in f:
        line = line.strip('\r\n')
        ss = line.split('\t')
        pwd = ss[0]
        if pwd == '':
            continue
        strength = ss[index]
        pwds[pwd] = strength

    return pwds

def generate(name:str, inputFile:TextIO, outputFile:TextIO, countFile:TextIO, index:int):
    pwdItems = initCountFile(countFile)
    pwds = [pwd[0] for pwd in pwdItems]

    meters = initMeterResult(inputFile, index, pwds)
    outputFile.write(f'count	strength	weight	{name}	password\n')

    
    for pwd,count,weight in pwdItems:
        # 不存在直接报错
        if pwd == '':
            continue
        strength = meters[pwd]
        count = int(float(count))
        weight = int(float(weight))
        # outputFile.write(f'{count}\t{-count}\t{count}\t{strength}\t{pwd}\n')
        outputFile.write(f'{count}\t{-weight}\t{weight}\t{strength}\t{pwd}\n')


def main():
    cli = argparse.ArgumentParser("Convert raw password file to password with frequency strength")
    cli.add_argument("-i", "--input", required=True, dest="input", type=argparse.FileType('r'),
                     help="password list(one password a line)")
    cli.add_argument("-n", "--name", required=True, dest="name", type=str,
                     help="Name of password meter")
    cli.add_argument("-o", "--output", required=False, dest="output", type=argparse.FileType('w'), default=sys.stdout,
                     help="output file")
    cli.add_argument("-c", "--count", required=True, dest="count", type=argparse.FileType('r'),
                     help="password count file")
    cli.add_argument("--strength-idx", required=False, dest="strength", type=int, default=1, 
                    help="strength index in input file")
    args = cli.parse_args()
    generate(args.name, args.input,args.output, args.count, args.strength)

if __name__ == '__main__':
    main()
