#!/usr/bin/env python3
import sys

if __name__ == "__main__":
    input_file_name = ""

    input_file_name = sys.argv[1]
    output_file_name = sys.argv[2]
    #print(input_file_name)
    
    input_res = open(input_file_name, 'r')
    output_res = open(output_file_name,'w')
    data_points = []
    count = 0
    for line in input_res:
        # print(line)
        line = line.strip("\r\n")
        segs = line.split("\t")
        for i in range(int(segs[2])):
            #if (int(segs[3])<int(10**14)):
            output_res.write(segs[0]+'\n')
        #print(segs[0])
    output_res.close() 
        
        #if len(segs[0]) > 32:
            # print(segs[0])
            #count+=1
    #print("total count: ", count)

