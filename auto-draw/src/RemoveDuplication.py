import sys,getopt

def main(args):
    infile = ''
    testfile = ''
    outfile = ''
    try:
        opts,args = getopt.getopt(args,"t:i:o:")
    except getopt.GetoptError:
        print("py -i infile -t testfile -o outfile")
        sys.exit(0)
    for opt,arg in opts:
        if opt == "-i":
            infile = arg
        elif opt == "-o":
            outfile = arg
        elif opt == "-t":
            testfile = arg
    
    print("train file:",infile)
    print("test file:",testfile)
    print("output file",outfile)
    train = set()
    with open(infile,"r") as f:
        for line in f:
            train.add(line)
    print("train set size:",len(train))
    with open(testfile,"r") as test:
        with open(outfile,"w") as out:
            for line in test:
                if line in train:
                    continue
                else:
                    out.write(line)
    

if __name__ == "__main__":
    main(sys.argv[1:])
