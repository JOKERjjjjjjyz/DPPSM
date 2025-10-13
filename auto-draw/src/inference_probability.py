import argparse
import math
import sys

from hamcrest import none
from pandas import value_counts

def read_prob(input_f, input_pwd, spliter, prob_index, max_prob):
    # total = 0
    satisfy = 0
    dict = {}
    with open(input_f, "r") as f:
        for line in f:
            line = line.strip('\r\n')
            ss = line.split("\t")
            pwd = ss[0]
            # print(ss)
            prob = float(ss[prob_index])
            dict[pwd]=prob
    tmp_pwd = None
    tmp_logits = sys.maxsize
    # print(tmp_logits)
    with open(input_pwd, "r") as f:
        for line in f:
            line = line.strip('\r\n')
            if (dict[line] < tmp_logits):
                tmp_logits = dict[line]
                tmp_pwd = line
    return tmp_pwd, tmp_logits

def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("-s",dest="spliter", default="'\\t",type=str,help="Spliter of current file")
    cli.add_argument("-c", dest="cracked",type=str,help="Input cracked password file")
    cli.add_argument("-p", dest="prob",type=float,help="Probabilities of threshold")
    cli.add_argument("-t", dest="pwds",type=str,help="Input password")
    args = cli.parse_args()
    pwds, logits = read_prob(args.cracked, args.pwds, args.spliter, 1, args.prob)
    print(f"Threshold: {pwds, logits}")

if __name__ == '__main__':
    main()
