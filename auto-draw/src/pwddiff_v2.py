#!/usr/bin/env python3

"""
Password Comparison Tools
Finding different passwords from target dataset with specific rules
"""

import argparse
import json
import sys
from collections import defaultdict
from typing import TextIO, Tuple, Callable, Iterable, Dict
import re

def readPatternList(patterns : TextIO):
    list = []
    if patterns == None:
        return list
    for line in patterns:
        line = line.strip("\r\n")
        if len(line) <= 0:
            continue
        if line[0] == "#":
            continue
        pattern = re.compile(line)
        list.append(pattern)
    patterns.close()
    return list

def checkPattern(list:Iterable[any], pwd:str):
    if(len(list) == 0):
        return True
    for pattern in list:
        if pattern.match(pwd) != None:
            return True
    return False

def getCharClassPattern(char_class:str):
    template = ""
    if char_class == "":
        char_class = "LUDS"
    ps = {'L':'a-z', "U":'A-Z', "D":'0-9', "S":r'~!@#$%^&*()_+\-={}\[\]|;\':",\./<>?\\'}
    s = set()
    char_class = char_class.upper()
    if re.match("^[LUDS]*$",char_class) == None:
        print(f'Character class not match requirements. char_class: {char_class}')
        exit(-1)
    for ch in char_class:
        s.add(ch)
    for ch in s:
        template = template + ps[ch]
    template = '^['+template+']*$'
    return re.compile(template)

def read_plain_text(file: TextIO):
    res = defaultdict(int)
    for line in file:
        line = line.strip('\r\n')
        res[line] += 1
    return res


def diff(finput:TextIO, target:TextIO, output:TextIO, 
        max_len:int, min_len:int, char_class:str, patterns:TextIO, 
        count:bool, key:Callable, target_key:Callable, 
        sort:str, max_guess:int, pwd_only:bool, pwds:Dict[any,any]):
    sortDict={"pwd":0, "freq":1, "guess":2}
    pwdList = defaultdict(int)
    pwdGuess = defaultdict(int)
    targetList = set()
    patternList = readPatternList(patterns)
    ccPattern = getCharClassPattern(char_class)

    for line in target:
        pwd,guess = target_key(line)
        if max_guess > 0 and guess > max_guess:
            continue
        targetList.add(pwd)
    target.close()
    print(len(targetList), len(pwds))
    total = 0
    for line in finput:
        pwd,guess = key(line)
        if pwd in targetList:
            continue
        total += 1
        if len(pwd) < min_len or len(pwd) > max_len:
            continue
        if ccPattern.match(pwd) == None:
            continue
        if not checkPattern(patternList, pwd):
            continue
        if max_guess > 0 and guess > max_guess:
            continue
        
        pwdGuess[pwd] = guess
        # add frequency to list
        pwdList[pwd] += pwds[pwd]
    finput.close()

    result = sorted(pwdList.items(), key=lambda x: x[sortDict[sort]], reverse=True)
    if total != 0:
        print(f"Total: {total}\nPwds: {len(result)}\nRate: {len(result)/total}")
    # output.write('%15s    %15s    %15s\n' % ('<password>','<frequency>','<guess number>'))
    for pwd,freq in result:
        if pwd_only:
            output.write('%s\n' % (pwd))
        else:
            output.write('%s\t%d\t%f\n' % (pwd,freq,pwdGuess[pwd]))
    if output is not sys.stdout:
        output.close()
    pass

def main():

    cli = argparse.ArgumentParser("Password Comparison Tools")
    cli.add_argument("-i","--input",required=True,dest="input",type=argparse.FileType('r'),
                    help="Cracked password data, each line one password")
    cli.add_argument("-t","--target",required=True,dest="target",type=argparse.FileType('r'),
                    help="Target password need to compare")
    cli.add_argument("--password",required=True,dest="password",type=argparse.FileType('r'),
                    help="Password set need to compare")
    cli.add_argument("-o","--output", required=False, dest="output",type=argparse.FileType('w'), default=sys.stdout, 
                    help="Output file, default console output")
    cli.add_argument("--max-len", required=False, dest="max_len", type=int, default=256,
                    help="Max length of passwords which you need")
    cli.add_argument("--min-len", required=False, dest="min_len", type=int, default=0,
                    help="Min length of passwords which you need")
    cli.add_argument("--char-class", required=False, dest="char_class", type=str, default="LUDS", 
                    help="Character class of password which you need. \n\tL: lower case letters.\n\tU: upper case letters. \n\t D: digital numbers. \n\tS: special characters")
    cli.add_argument("-p", "--patterns", required=False, dest="patterns", type=argparse.FileType('r'), default=None, 
                    help="Pattern file. Per regular expression each line")
    cli.add_argument("--count", required=False, dest="count", action="store_true", default=False, 
                    help="Count password frequency, default false.")
    cli.add_argument("--input-pwd-index",required=False, dest="input_pwd",type=int, default=0, 
                    help="Password index in input file")
    cli.add_argument("--target-pwd-index",required=False, dest="target_pwd",type=int, default=0, 
                    help="Password index in target file")
    cli.add_argument("--split-i", required=False, dest="split_i", default="\t", type=str,
                    help="how to split a line in input file, default is '\\t'")
    cli.add_argument("--split-t", required=False, dest="split_t", default="\t", type=str,
                    help="how to split a line in target file, default is '\\t'")
    cli.add_argument("--sort-by", required=False, dest="sort_by", type=str, default="pwd", choices=["pwd","freq"], 
                    help="Output order, default password")
    cli.add_argument("--input-guess-index", required=False, dest="input_guess", type=int, default=-1, 
                    help="Guessed password number index in input file")
    cli.add_argument("--target-guess-index", required=False, dest="target_guess", type=int, default=-1, 
                    help="Guessed password number index in target file")
    cli.add_argument("--max-guess", required=False, dest="max_guess", type=float, default=-1,
                    help="Max number of guessed passwords")
    cli.add_argument("--pwd-only", required=False, dest="pwd_only", default=False, action="store_true", 
                    help="print password only")

    args = cli.parse_args()
    split = args.split_i.replace('\\\\', '\\')
    split_t = args.split_t.replace('\\\\', '\\')

    pwds = read_plain_text(args.password)

    def my_key(line: str):
        try:
            split_line = line.strip("\r\n").split(split)
            guess = 0
            if args.input_guess != -1:
                guess =  float(split_line[args.input_guess])
            return split_line[args.input_pwd], guess
        except Exception as e:
            print(e, file=sys.stderr)
            print(f"input file in {line}", end="", file=sys.stderr)
            print(f"Your split is '{split}',\n"
                  f"    input_pwd is '{args.input_pwd}',\n"
                  f"    input_guess is '{args.input_guess}',\n", file=sys.stderr)
            sys.exit(-1)

    def target_key(line: str):
        try:
            split_line = line.strip("\r\n").split(split_t)
            guess = 0
            if args.input_guess != -1:
                guess =  float(split_line[args.target_guess])
            return split_line[args.target_pwd], guess
        except Exception as e:
            print(e, file=sys.stderr)
            print(f"target file in {line}", end="", file=sys.stderr)
            print(f"Your split is '{split_t}',\n"
                  f"    target_pwd is '{args.target_pwd}',\n"
                  f"    target_freq is '{args.target_freq}'", file=sys.stderr)
            sys.exit(-1)
    
    diff(args.input, args.target, args.output, 
        args.max_len, args.min_len, args.char_class, 
        args.patterns, args.count, my_key, target_key, 
        args.sort_by, args.max_guess, args.pwd_only, pwds)

    

if __name__ == '__main__':
    main()
