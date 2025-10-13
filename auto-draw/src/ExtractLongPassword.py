#!/usr/bin/env python3

"""
Password extract long password from given data set
"""

import argparse
import json
import sys
from collections import defaultdict
from typing import TextIO, Tuple, Callable, Iterable, Dict


def extract_long_password(pwds, maxlen, output):
    for pwd in pwds:
        if(len(pwd) >= maxlen):
            output.write(f"{pwd}\n")

def main():
    cli = argparse.ArgumentParser("Long password extractor")
    cli.add_argument("-i", "--input", required=True,type=argparse.FileType('r'))
    cli.add_argument("-o", "--output", required=True,type=argparse.FileType('w'))
    cli.add_argument("--len", type=int, default=16, help="Extract password length >= #len")
    args = cli.parse_args()
    pwds =[line.strip("\r\n") for line in args.input]
    extract_long_password(pwds, args.len, args.output)

if __name__ == '__main__':
    main()

