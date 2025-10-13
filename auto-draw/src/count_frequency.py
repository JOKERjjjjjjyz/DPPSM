import collections
import os,sys
import argparse
from typing import TextIO, Tuple, Callable

def count(inputFile:TextIO, output:TextIO, split=" "):
    counts = collections.Counter(ss for line in inputFile for ss in line[:-1].split(split))

    for subword,count in counts.most_common():
        output.write('%10s %10d\n' % (subword,count))
    
    inputFile.close()
    if output != sys.stdout:
        output.close()
    pass

def main():
    cli = argparse.ArgumentParser("Password segment counter")
    cli.add_argument("-i", "--input", required=True, dest="input", type=argparse.FileType('r'),
                     help="password list(one password a line)")
    cli.add_argument("-o", "--output", required=False, dest="output", type=argparse.FileType('w'), default=sys.stdout,
                     help="output vocaburary list")
    cli.add_argument("--split", required=False, dest="split", default=" ", type=str,
                     help="how to split a line in password file, default is ' '")
    args = cli.parse_args()
    count(args.input,args.output,args.split)
    pass

if __name__ == '__main__':
    main()