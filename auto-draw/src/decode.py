import collections
import os,sys
import argparse
from typing import TextIO, Tuple, Callable, List, Dict
import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold

PASSWORD_END = '\n'

class PasswordSet:
    def __init__(self, inputFile:TextIO, maxlen=10):
        self.list = []
        for line in inputFile:
            self.list.append(line[:-1])
        self.size = len(self.list)
        self.encodes = np.zeros((len(self.list),maxlen))
        self.points = np.zeros((len(self.list),2))

class CharacterTable:
    def __init__(self, charbag:List[str], max_len=10):
        self.charbag = charbag
        # self.charbag.append(PASSWORD_END)
        self.charbag.insert(0,PASSWORD_END)
        self.char_indices = dict((c, i) for i, c in enumerate(self.charbag))
        self.indices_char = dict((i, c) for i, c in enumerate(self.charbag))
        self.maxlen = max_len

    def padding(self,astring:str) -> List[str]:
        maxlen = self.maxlen
        # 截断
        if len(astring) > maxlen:
            astring = astring[len(astring) - maxlen:]
        list = []
        for ch in astring:
            list.append(ch)
        for _ in range(maxlen - len(astring)):
            list.append(PASSWORD_END)
        return list

    def encode(self,astring:str):
        list = self.padding(astring)
        X = np.zeros((self.maxlen), dtype=np.int16)
        for index,ss in enumerate(list):
            X[index] = self.char_indices[ss]
        return X


    @staticmethod
    def fromPasswordSet(pwdSet:PasswordSet, maxlen=10):
        list = []
        counts = collections.Counter(ss for line in pwdSet.list for ss in line)
        for subword,_ in counts.most_common():
            list.append(subword)
        return CharacterTable(list,maxlen)

class SubwordTable(CharacterTable):
    def __init__(self, subwordbag:List[str], max_len=10, spliter=" "):
        super().__init__(subwordbag, max_len)
        self.charbag = subwordbag
        self.char_indices = dict((c, i) for i, c in enumerate(self.charbag))
        self.indices_char = dict((i, c) for i, c in enumerate(self.charbag))
        self.maxlen = max_len
        self.spliter = spliter

    def padding(self, astring) -> List[str]:
        maxlen = self.maxlen
        # 截断
        subs = astring.split(self.spliter)
        length = len(subs)
        if length > maxlen:
            subs = subs[length-maxlen:]
            return subs
        for _ in range(maxlen - length):
            subs.append(PASSWORD_END)
        return subs

    @staticmethod
    def fromPasswordSet(pwdSet:PasswordSet, maxlen=10, spliter=" "):
        list = []
        counts = collections.Counter(ss for line in pwdSet.list for ss in line.split(spliter))
        for subword,_ in counts.most_common():
            list.append(subword)
        return SubwordTable(list,maxlen, spliter)
            

def draw(pwdSet:PasswordSet, out:str):
    x_min, x_max = pwdSet.points.min(0), pwdSet.points.max(0)
    X_norm = (pwdSet.points - x_min) / (x_max - x_min)
    # X_norm = pwdSet.points
    plt.switch_backend('agg')
    plt.figure(figsize=(8, 8))
    plt.scatter(X_norm[:,0], X_norm[:,1], alpha=0.4)
    # plt.xticks([])
    # plt.yticks([])
    plt.savefig(out, dpi=300)
    pass

def decode(pwdSet:PasswordSet, maxlen:int, ctable:CharacterTable):
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
    for index,pwd in enumerate(pwdSet.list):
        pwdSet.encodes[index] = ctable.encode(pwd)
    print('Encode over')
    pwdSet.points = tsne.fit_transform(pwdSet.encodes)
    print('TSNE over')
    pass

def main():
    cli = argparse.ArgumentParser("Password decode and 2D display")
    cli.add_argument("-i", "--input", required=True, dest="input", type=argparse.FileType('r'),
                     help="password list(one password a line)")
    cli.add_argument("-o", "--output", required=True, dest="output", type=str,
                     help="2D figure")
    cli.add_argument("--split", required=False, dest="split", default=" ", type=str,
                     help="how to split a line in password file, default is ' '. (subword mode need)")
    cli.add_argument("--max-len", required=False, dest="max_len", default=10, type=int, 
                    help="max password/subwords length")
    cli.add_argument("--mode", required=False, dest="mode", default="subword", type=str, choices=["subword","character"], 
                     help="password mode. subword means password split into segment. character level just need one password a line")
    cli.add_argument("--debug", required=False, dest="debug", default=None, type=argparse.FileType('w'),
                     help="debug output. None for no debug information")
    args = cli.parse_args()
    pwdSet = PasswordSet(args.input,args.max_len)
    if args.mode == "character":
        ctable = CharacterTable.fromPasswordSet(pwdSet, args.max_len)
    else:
        ctable = SubwordTable.fromPasswordSet(pwdSet, args.max_len)
    # ctable = CharacterTable.fromPasswordSet(pwdSet)
    decode(pwdSet, args.max_len, ctable)
    if args.debug != None:
        debugFile = args.debug
        debugFile.write(ctable.charbag.__str__())
        debugFile.write('encode:\n')
        for nn in pwdSet.encodes:
            debugFile.write(nn.__str__()+'\n')
        debugFile.write('tsne:\n')
        for nn in pwdSet.points:
            debugFile.write(nn.__str__()+'\n')
        debugFile.close()
    draw(pwdSet, args.output)
    pass

if __name__ == '__main__':
    main()


