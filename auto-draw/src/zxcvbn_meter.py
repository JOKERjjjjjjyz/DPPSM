import sys, os
import argparse
from typing import TextIO, Tuple, Callable, List, Dict, Tuple
from zxcvbn import zxcvbn

def measure(inputFile:TextIO, output:TextIO, spliter:str, idx:int, info_index:List[int]):
    list = []
    for line in inputFile:
        if len(line) <= 1:
            continue
        line = line[:-1]
        pwd = line
        pair = []
        if spliter != None:
            ss = line.split(spliter)
            pwd = ss[idx]
            pair.append(pwd)
            for index in info_index:
                pair.append(ss[index])
        else:
            pair.append(pwd)
        res = zxcvbn(pwd)
        pair.append(res['guesses'])
        list.append(tuple(pair))
    length = len(info_index) + 1
    list = sorted(list, key = lambda x: x[length], reverse=True)
    for pair in list:
        output.write("%s" % (pair[0]))
        for i in range(1, len(pair)-1):
            output.write("\t%s" % (pair[i]))
        output.write("\t%f\n" % (pair[len(pair)-1]))
        pass
    
    if output != sys.stdout:
        output.close()


def main():
    cli = argparse.ArgumentParser("Zxcvbn: password evaluation")
    cli.add_argument("-i", "--input", required=True, dest="input", type=argparse.FileType('r'),
                     help="password list(one password a line)")
    cli.add_argument("-o", "--output", required=False, dest="output", type=argparse.FileType('w'), default=sys.stdout,
                     help="output file with password and password guessing number")
    cli.add_argument("--split", required=False, dest="split", default=None, type=str,
                     help="how to split a line in password file, default is ' '. (subword mode need)")
    cli.add_argument("--pwd-index", required=False, dest="pwd_index", default=0, type=int, 
                    help="password index in passsword file")
    cli.add_argument("--indexes", required=False, dest="indexes", default=[], type=int, nargs="+", 
                    help="other password infomation")
    args = cli.parse_args()
    if args.split == '\\t':
        args.split = '\t'
    
    measure(args.input, args.output, args.split, args.pwd_index, args.indexes)


if __name__ == '__main__':
    main()
