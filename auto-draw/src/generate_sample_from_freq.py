import collections
import os,sys
import argparse
from typing import TextIO, Tuple, Callable

def count(inputFile:TextIO, output:TextIO, split=" "):
    for line in inputFile:
        #print(line)
        #print(split+str("dffs"))
        pwd, count, _ = line.split(split)
        #print(str(pwd), "--",str(count))
        for i in range(int(count)):
            output.write('%s\n' % (pwd))    
    inputFile.close()
    if output != sys.stdout:
        output.close()
    pass

def main():
    cli = argparse.ArgumentParser("Password segment counter")
    cli.add_argument("-i", "--input", required=True, dest="input", type=argparse.FileType('r'),
                     help="password list(password, freq, xx per line)")
    cli.add_argument("-o", "--output", required=False, dest="output", type=argparse.FileType('w'), default=sys.stdout,
                     help="output pwds list")
    cli.add_argument("--split", required=False, dest="split", default="\t", type=str,
                     help="how to split a line in password file, default is ' '")
    args = cli.parse_args()
    count(args.input,args.output,args.split)
    pass

if __name__ == '__main__':
    main()