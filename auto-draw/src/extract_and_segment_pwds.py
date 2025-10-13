#!/usr/bin/env python3
import sys

if __name__ == "__main__":
    input_file_name = ""
    output_file_name = ""
    if len(sys.argv) < 3:
        print("Miss input file！")
    else:
        input_file_name = sys.argv[1]
        output_file_name = sys.argv[2]
        print(input_file_name)
    
    input_res = open(input_file_name, 'r')
    output_res = open(output_file_name, 'w')
    data_points = []
    for line in input_res:
        line = line.strip("\r\n")
        line = ' '.join(line.split())
        segs = line.split(" ")

        # print(segs)
        if len(segs) == 3:
            password, freq, guess_number = segs
            output_res.write("{0}\n".format(password))
