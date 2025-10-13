#!/usr/bin/env python3

"""
Password Min Auto Tools
Finding minimal guessing number of all passwords
"""

import argparse
import json
import sys
from collections import defaultdict
from typing import TextIO, Tuple, Callable, Iterable, Dict, List

errorCount = 0

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
            # sys.stderr.write("Wrong password set:  not in password set "+pwd+"\n")
            global errorCount
            errorCount = errorCount + 1
            continue
        num, rank = pwd_rank[pwd]
        if guess < rank:
            pwd_rank[pwd] = (num, guess)

def minAuto(data:TextIO, target:List[Dict], out:str, minBound:int, maxBound:int, title:bool):
    bag = init_targets(data)
    data.close()
    global errorCount
    for dataset in target:
        errorCount = 0
        split = dataset['split'].replace('\\\\', '\\')
        if split == "\\t":
            split = "\t"
        def key(line):
            try:
                items = line.strip("\r\n").split(split)
                return items[int(dataset['pwd_index'])],int(items[int(dataset['guess_index'])])
            except:
                raise Exception("Wrong data format:"+line)
        with open(dataset['path']) as f:
            parse_rank(bag,f,key)
            if(errorCount != 0):
                print('error password: ',dataset['path'],':',errorCount)
    
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
        if number >= minBound and number <= maxBound:
            output.write(f"%s\t%s\t%d\t%d\t%d\n" % (line[0],'0.0',line[1][0],line[1][1],number))
    if output != sys.stdout:
        output.close()
    

def main():
    cli = argparse.ArgumentParser("min-auto-json")
    cli.add_argument("-i", "--input", required=True, dest="input", type=argparse.FileType("r"), help="Input json configure file")
    cli.add_argument("-d","--data", required=True, dest="data", type=argparse.FileType("r"), help="Data file, one password per line")
    cli.add_argument("-o","--output",default=None,type=str,dest="output",required=False,help="Output file, default standard output")
    cli.add_argument("--min-bound", default=0, dest="min", type=int, help="Minimal guess number")
    cli.add_argument("--max-bound", default=10**18, dest="max", type=int, help="Maximal guess number")
    cli.add_argument("--title", dest="title", default=False, action="store_true", help="Print table tile,default false")
    args = cli.parse_args()

    data = json.load(args.input)
    
    minAuto(args.data, data, args.output, args.min, args.max, args.title)

if __name__ == "__main__":
    main()

# [{"path":"","pwd_index":"","guess_index":"","split":""}]
