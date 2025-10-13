import numpy as np
import getopt,sys

def main(args):
    infile = ''
    outfile = ''
    try:
        opts,args = getopt.getopt(args,"i:o:")
    except getopt.GetoptError:
        print('cmd: -i [input file] -o [out file]')
        sys.exit(0)
    for opt,arg in opts:
        if opt == "-i":
            infile = arg
        if opt == "-o":
            outfile = arg
    set = []
    with open(infile,"r") as f:
        for line in f:
            if len(line) <= 33:
                set.append(line)
    
    number = int(len(set)/10)
    np.random.shuffle(set)
    with open(outfile,"w") as f:
        for i in range(number):
            f.write(set[i])
    


if __name__ == "__main__":
    main(sys.argv[1:])
