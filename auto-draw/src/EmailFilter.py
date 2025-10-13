

#!/usr/bin/env python3
"""
Select passwords with email patterns from input dataset 
"""
import argparse
import json
import sys
from collections import defaultdict
from typing import TextIO, Tuple, Callable, List
import re

class Record:
    def __init__(self, pwd:str, freq:int=1):
        self.pwd = pwd
        self.pattern = ""
        self.freq = freq

class BlockList:
    def __init__(self, blocks:List[str]=[]):
        self.list = self.getDefaultList()
        self.list = self.list + blocks
        self.table = set()
        for item in self.list:
            self.table.add(item)

    def isNotBlockSegment(self, pwd:str)->bool:
        if isinstance(pwd, Tuple):
            pwd = pwd[0]
        if pwd is None or len(pwd) < 1:
            return False
        # 长度为2的默认为国家域名
        if len(pwd.split(".")[-1]) == 2:
            return True
        # 长度大于2的需要为国际域名
        return pwd.split(".")[-1] in self.table 

    def getDefaultList(self)->List[str]:
        list = [
            # 国际域名
            "com",
            "edu",
            "gov",
            "int",
            "mil",
            "net",
            "org",
            "biz",
            "info",
            "pro",
            "name",
            "museum",
            "coop",
            "aero",
            "xxx",
            "idv"
        ]
        return list


class PwdFilter:
    def __init__(self, block:BlockList=BlockList()):
        self.block = block
        self.total = 0
        self.filter_number = 0

    def isValid(self, pwd:Record)->bool:
        return True

    def filter(self, pwdList:List[Record])->List[Record]:
        return list(filter(lambda x : self.isValid(x), pwdList))

    def finish(self):
        print("Total password checked: %d\nFilter password number: %d\nemail pattern password: %d, %5.2f\n" % (self.total, self.filter_number, self.total-self.filter_number, (self.total-self.filter_number) / self.total * 100))

class DatePwdFilter(PwdFilter):
    def getPwd(self, result):
        if isinstance(result, Tuple):
            return result[0]
        return result

    def isValid(self, record:Record)->bool:
        patterns_all = [
            r'[A-Za-z\d]+([-_.][A-Za-z\d]+)*@([A-Za-z\d]+[-.])+[A-Za-z\d]{4}',
            r'[A-Za-z\d]+([-_.][A-Za-z\d]+)*@([A-Za-z\d]+[-.])+[A-Za-z\d]{3}',
            r'[A-Za-z\d]+([-_.][A-Za-z\d]+)*@([A-Za-z\d]+[-.])+[A-Za-z\d]{2}'
        ]
        self.total = self.total + record.freq
        for pattern in patterns_all:
            # s = re.fullmatch(pattern,record.pwd)
            # if s is None:
            #     continue
            # s = s.group()
            # if self.block.isNotBlockSegment(s):
            #     print(s, "valid",record.pwd,  pattern)
            #     record.pattern = self.getPwd(s)
            #     return True
            s = re.match(pattern,record.pwd)
            if s is None:
                continue
            s = s.group()
            # print(s)
            if s is None:
                continue
            if self.block.isNotBlockSegment(s):
                # print(s, "valid",record.pwd,  pattern)
                record.pattern = self.getPwd(s)
                return True
        self.filter_number = self.filter_number + record.freq
        return False

def main():
    cli = argparse.ArgumentParser("Search all passwords match date pattern")
    cli.add_argument("-i", "--input", required=False, dest="input", default=sys.stdin, type=argparse.FileType('r'),
                     help="input password file. one password one line")
    cli.add_argument("-o", "--output", required=False, dest="output", default=sys.stdout, type=argparse.FileType("w"),
                     help="output password which match date pattern")
    cli.add_argument("--detail", required=False, action="store_true", default=False, dest="detail",
                    help="show detail information from filter")
    cli.add_argument("--freq", required=False, default=-1, dest="freq", type=int, 
                    help="frequency index. input one line with password and frequency")
    cli.add_argument("--split", required=False, dest="split", default="\t", type=str,
                     help="how to split a line in password file, default is '\\t'")
    args = cli.parse_args()
    spliter = args.split.replace('\\\\', '\\')
    if spliter == '\\t':
        spliter = '\t'
    list = []
    if args.freq <= 0:
        list = [Record(pwd.strip("\r\n").split(spliter)[0]) for pwd in args.input]
    else:
        list = [Record(pwd.strip("\r\n").split(spliter)[0], freq=int(float(pwd.strip("\r\n").split(spliter)[args.freq]))) for pwd in args.input]
    f = DatePwdFilter()
    total_pwd = sum([item.freq for item in list])
    result = f.filter(list)
    writer = args.output
    total_cnt= sum([item.freq for item in result])
    for item in result:
        if args.detail:
            writer.write(f"{item.pwd}\t{item.freq}\t{item.pattern}\n")
        else:
            writer.write(f"{item.pwd}\t{item.freq}\t{item.freq / total_pwd * 100:7.4f}\n")
    if args.detail:
        f.finish()
    print(f"containing date: {total_cnt / total_pwd * 100:5.2f}", file=sys.stderr)
    pass

if __name__ == "__main__":
    main()
