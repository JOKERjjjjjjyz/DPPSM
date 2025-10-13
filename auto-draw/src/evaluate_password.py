import sys, os
import argparse
from typing import TextIO, Tuple, Callable, List, Dict, Tuple

def read_dict(f_dict: TextIO, key:Callable):
    res = {}
    for line in f_dict:
        line = line.strip('\r\n')
        pwd, guess = key(line)
        res[pwd] = guess
    return res

def measure(mapper, output:TextIO, pwds:List[str], threshold:int):
    ans = []
    total = 0
    error_data = 0
    for pwd in pwds:
        if pwd not in mapper:
            error_data += 1
            # print(f"{pwd} not in mapper!")
            continue
        total += 1
        guess = mapper[pwd]
        if guess < threshold:
            ans.append(pwd)
    for pwd in ans:
        output.write(f"{pwd}\t{mapper[pwd]}\n")
    assert total is not 0
    print(f"Total: {total}, Count: {len(ans)}, Coverage: {len(ans)/total}, error data: {error_data}")

def main():
    cli = argparse.ArgumentParser("Password evaluation for models(FLA and Zxcvbn)")
    cli.add_argument("-i", "--input", required=True, dest="input", type=argparse.FileType('r'),
                     help="password evaluation value")
    cli.add_argument("-p", "--password", required=True, dest="password", type=argparse.FileType('r'),
                     help="password list(one password a line)")
    cli.add_argument("-o", "--output", required=False, dest="output", type=argparse.FileType('w'), default=sys.stdout,
                     help="output file with password and password guessing number")
    cli.add_argument("--split", required=False, dest="split", default='\t', type=str,
                     help="how to split a line in password file, default is ' '. (subword mode need)")
    cli.add_argument("--pwd-index", required=False, dest="pwd_index", default=0, type=int, 
                    help="password index in passsword evaluation file")
    cli.add_argument("--guess-index", required=True, dest="guess_index", default=0, type=int, 
                    help="guess index in passsword evaluation file")
    cli.add_argument("--threshold", required=False, dest="threshold", default=1000_000_000, type=int, 
                    help="threshold of password we evaluate")
    args = cli.parse_args()
    if args.split == '\\t':
        args.split = '\t'
    def key(line):
        try:
            ss = line.split(args.split)
            return ss[args.pwd_index], float(ss[args.guess_index])
        except Exception as e:
            print(f"Line: '{line}' not match password index '{args.pwd_index}' and guess index '{args.guess_index}'")
    mapper = read_dict(args.input, key)
    pwds = [line.strip('\r\n').split(args.split)[0] for line in args.password]
    measure(mapper, args.output, pwds, args.threshold)

if __name__ == '__main__':
    main()
