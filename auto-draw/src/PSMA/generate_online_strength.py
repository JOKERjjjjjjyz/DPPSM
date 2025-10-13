import argparse
import json
import sys
from collections import defaultdict
from typing import TextIO, Tuple, Callable, List

def couting(inputFile:TextIO):
    pwds = defaultdict()
    for line in inputFile:
        line = line.strip('\r\n')
        if line in pwds:
            pwds[line] = pwds[line] + 1
        else:
            pwds[line] = 1
    return pwds

def generate(inputFile:TextIO, output:TextIO):
    pwds = couting(inputFile)
    # title
    # output.write(f'pwd\tcount\n')
    dict=sorted(pwds.items(), key=lambda d:d[1], reverse = True)
    for pwd,count in dict:
        output.write(f'{pwd}\t{count}\n')

def main():
    cli = argparse.ArgumentParser("Convert raw password file to password with frequency strength")
    cli.add_argument("-i", "--input", required=True, dest="input", type=argparse.FileType('r'),
                     help="password list(one password a line)")
    cli.add_argument("-o", "--output", required=False, dest="output", type=argparse.FileType('w'), default=sys.stdout,
                     help="output file")
    args = cli.parse_args()
    generate(args.input,args.output)

if __name__ == '__main__':
    main()
