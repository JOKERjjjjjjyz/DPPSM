import numpy as np

import os
import re
from PIL import Image
from os import path
from wordcloud import WordCloud
import string
import argparse
from typing import TextIO, Tuple, Callable, List
# Keyboard check script
from kbd import detect_keyboard_walk

class PatternChecker:
    def check(self,pwd)->bool:
        return True

class DatePatternChecker(PatternChecker):
    def __init__(self):
        self.table = self.getDefaultList()

    def isNotBlockSegment(self, pwd:str)->bool:
        if isinstance(pwd, Tuple):
            pwd = pwd[0]
        return pwd not in self.table 

    def getDefaultList(self)->List[str]:
        list = [
            "111111",
            "123123",
            "111000",
            "112233",
            "100200",
            "111222",
            "121212",
            "520520",
            "110110",
            "123000",
            "101010",
            "111333",
            "110120",
            "102030",
            "110119",
            "121314",
            "521125",
            "120120",
            "101203",
            "122333",
            "121121",
            "101101",
            "131211",
            "100100",
            "321123",
            "110112",
            "112211",
            "111112",
            "520521",
            "110111",
            "11111111",
            "123123123",
            "12121212",
            "11223344",
            "1111111111",
            "123123",
            "111111111",
            "5201314"
        ]
        return list
    
    def check(self,pwd)->bool:
        patterns = [
            r'^((19|20)\d{2})$',
            r'^(0[123456789]0[123456789])$',
            r'^([12]\d{1}0[123456789])$',
            r'^(3[01]0[123456789])$',
            r'^([0][123456789]1[012])$',
            r'^([12]\d{1}1[012])$',
            r'^(3[01]1[012])$',
            r'^0[123456789]0[123456789]$',
            r'^0[123456789][12]\d{1}$',
            r'^0[123456789]3[01]$',
            r'^1[012][0][123456789]$',
            r'^1[012][12]\d{1}$',
            r'^1[012]3[01]$',
        ]
        for pattern in patterns:
            ss = re.findall(pattern,pwd)
            for s in ss:
                if self.isNotBlockSegment(s):
                    return True
        return False

class ManglingPatternChecker(PatternChecker):
    def __init__(self):
        self.table = set(self.getDefaultManglingDict())

    def getDefaultManglingDict(self)->List[str]:
        list = [
            "iluv",
            "4ever",
            "4life",
            "luv",
            "iloveu",
            "4eva",
            "520",
            "521",
            "1314",
            "5201314"
        ]
        return list

    def check(self,pwd)->bool:
        if pwd in self.table:
            return True
        return False

class KeyBoardPatternChecker(PatternChecker):
    def check(self,pwd)->bool:
        session,found_list = detect_keyboard_walk(pwd,min_keyboard_run=4)
        if len(found_list) > 0:
            return True
        return False

class DictPatternChecker(PatternChecker):
    def __init__(self, check_subword=False):
        self.table = set()
        self.check_subword = check_subword

    def set_dict(self,word_list:TextIO):
        self.table = set([line.strip('\r\n') for line in word_list])

    def check(self,pwd)->bool:
        if pwd.lower() in self.table:
            return True
        if not self.check_subword:
            return False
        for i in range(len(pwd)):
            for j in range(i+2,len(pwd)):
                if pwd[i:j].lower() in self.table:
                    return True
        return False

def readWord(file, split='\t', word_min_length=3):
    words = {}
    bag = set([c for c in string.printable])
    for line in file:
        line = line[:-1]
        ss = line.split(split)
        if(len(ss) < 2):
            continue
        if(ss[0][-1:] not in bag):
            ss[0] = ss[0][:-1]
        if(len(ss[0]) < word_min_length):
            continue
        words[ss[0]] = max(int(ss[1]),words.get(ss[0],0))
    return words


def groupWord(word_dict, max_words, layers=10, base_line=100, step=10):
    max_words = min(len(word_dict), max_words)
    words = sorted(word_dict.items(), key=lambda x:x[1],reverse=True)
    words = words[:max_words]
    words = sorted(words, key=lambda x:x[1])
    ans = {}
    group_size = len(words)//10
    length = 0
    for word, freq in words:
        print(word,freq, base_line)
        ans[word] = base_line
        length += 1
        if length >= group_size:
            length = 0
            base_line += step
    
    return ans

# Mangling Rule
# Keyboard
# Date
# Syllable & Word
patterns = {
    "date":(DatePatternChecker(),"navy"),
    "mangling":(ManglingPatternChecker(),"darkred"),
    "keyboard":(KeyBoardPatternChecker(),"mediumslateblue"),
    "word":(DictPatternChecker(False),"green"),
    "syllable":(DictPatternChecker(False),"orange"),
    "default":(PatternChecker(),"grey")
}

def color_map(*args, **kwargs):
    pwd = args[0]
    orders = ["mangling","word","syllable","date","keyboard","default"]
    for pattern in orders:
        detecter,color = patterns[pattern]
        if detecter.check(pwd):
            return color
    return "red"

def getWordCloud(random_color=True, max_words=150):
    if random_color:
        return WordCloud(background_color="white", max_words=max_words, width=1000,height=800,max_font_size=200, relative_scaling=.8)
    return WordCloud(background_color="white", max_words=max_words, width=1000,height=800, min_font_size=20,max_font_size=120, relative_scaling=.3,color_func=color_map)
    
def saveImage(text,filename,random_color=True, max_words=150):
    wc = getWordCloud(random_color, max_words)
    # generate word cloud
    wc.generate_from_frequencies(text)
    wc.to_file(filename)

def main():
    cli = argparse.ArgumentParser("Password word cloud generator")
    cli.add_argument("-i", "--input", required=True, dest="input", type=str,
                     help="vocabulary list")
    cli.add_argument("-o", "--output", required=True, dest="output", type=str,
                     help="output image path (png)")
    cli.add_argument("--min-length", required=False,default=3 , dest="min_length", type=int, 
                     help="min length of word")
    cli.add_argument("--split", required=False, dest="split", default="\t", type=str,
                     help="how to split a line in vocabulary file, default is '\\t'")
    cli.add_argument("--syllable", required=False, dest="syllable", type=argparse.FileType('r'),
                     help="Chinese syllable table.")
    cli.add_argument("--word", required=False, dest="word", type=argparse.FileType('r'),
                     help="English word table.")
    cli.add_argument("--random-color", required=False, default=False, action="store_true",dest="random_color",
                     help="Randomly select color for word")
    max_words = 150
    args = cli.parse_args()
    if not args.random_color:
        patterns["syllable"][0].set_dict(args.syllable)
        patterns["word"][0].set_dict(args.word)
    with open(args.input,'r') as file:
        words = readWord(file,split=args.split,word_min_length=args.min_length)
        # words = groupWord(words, max_words, 10, 100, 2)
        saveImage(words,args.output,args.random_color, max_words)
    pass


if(__name__ == "__main__"):
    main()
