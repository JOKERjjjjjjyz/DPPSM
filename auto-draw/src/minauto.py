#!/usr/bin/env python3

"""
Password Min Auto Tools
Finding minimal guessing number of all passwords
"""

import argparse
import json
import sys
from collections import defaultdict
from typing import TextIO, Tuple, Callable, Iterable, Dict

def init_targets(targets: TextIO):
    pwd_rank = {}
    for line in targets:
        line = line.strip("\r\n")
        num, _ = pwd_rank.get(line, (0, 0))
        pwd_rank[line] = (num + 1, sys.float_info.max)
    return pwd_rank

def parse_rank(pwd_rank: Dict, model: TextIO, key: Callable):
    for line in model:
        if len(line.strip('\r\n')) == 0:
            continue
        pwd, guess = key(line)
        if pwd not in pwd_rank:
            # raise Exception("Wrong password set:  not in password set"+pwd)
            sys.stderr.write("Wrong password set:  not in password set "+pwd+"\n")
            continue
        num, rank = pwd_rank[pwd]
        if guess < rank:
            pwd_rank[pwd] = (num, guess)

def minAuto(data:TextIO, target1:TextIO, target2:TextIO, out:str, minBound:int, maxBound:int, title:bool, key1:Callable, key2:Callable):
    bag = init_targets(data)
    data.close()
    parse_rank(bag,target1,key1)
    parse_rank(bag,target2,key2)
    target1.close()
    target2.close()
    
    output = sys.stdout
    if out:
        try:
            output = open(out,"w+")
        except Exception as e:
            raise e

    result = sorted(bag.items(), key = lambda x : x[1][1])
    number = 0
    for line in result:
        number = number + line[1][0]
        output.write(f"%s\t%s\t%d\t%d\t%d\n" % (line[0],'0.0',line[1][0],line[1][1],number))
    if output != sys.stdout:
        output.close()
    

def main():
    cli = argparse.ArgumentParser("min-auto")
    cli.add_argument("-t", "--target", required=True, dest="target", type=argparse.FileType("r"), help="Input file, one data one row")
    cli.add_argument("-i", "--input", required=True, dest="input", type=argparse.FileType("r"), help="Input file, one data one row")
    cli.add_argument("-d","--data", required=True, dest="data", type=argparse.FileType("r"), help="Data file, one password per line")
    cli.add_argument("-o","--output",default=None,type=str,dest="output",required=False,help="Output file, default standard output")
    cli.add_argument("--split-t", dest="split_t", default='\t', required=False, type=str, help="target split words, default '\\t'")
    cli.add_argument("--split-i", dest="split_i", default='\t', required=False, type=str, help="input split words, default '\\t'")
    cli.add_argument("--target-pwd-index", dest="tpwd_index", default=0,required=False, type=int, help="password index in target file")
    cli.add_argument("--target-guess-index", dest="tguess_index", default=2,required=False, type=int, help="guess index in target file")
    cli.add_argument("--input-pwd-index", dest="ipwd_index", default=0,required=False, type=int, help="password index in input file")
    cli.add_argument("--input-guess-index", dest="iguess_index", default=2,required=False, type=int, help="guess index in input file")
    cli.add_argument("--min-bound", default=0, dest="min", type=int, help="Minimal guess number")
    cli.add_argument("--max-bound", default=10**18, dest="max", type=int, help="Maximal guess number")
    cli.add_argument("--title", dest="title", default=False, action="store_true", help="Print table tile,default false")
    args = cli.parse_args()

    split_t = args.split_t.replace('\\\\', '\\')
    split_i = args.split_i.replace('\\\\', '\\')

    def key1(line):
        try:
            items = line.strip("\r\n").split(split_t)
            return items[args.tpwd_index],int(items[args.tguess_index])
        except:
            raise Exception("Wrong data format:"+line)
    def key2(line):
        try:
            items = line.strip("\r\n").split(split_i)
            return items[args.ipwd_index],int(items[args.iguess_index])
        except:
            raise Exception("Wrong data format:"+line)
    
    minAuto(args.data, args.target,args.input,args.output, args.min, args.max, args.title, key1, key2)

if __name__ == "__main__":
    main()
