#!/usr/bin/env python3
import sys

if __name__ == "__main__":
    input_file_name = ""

    input_file_name = sys.argv[1]
    #output_file_name = sys.argv[2]
    #print(input_file_name)
    
    input_res = open(input_file_name, 'r')
    #output_res = open(output_file_name,'w')
    data_points = []
    count = 0
    total = 0
    count2=0
    count3=0
    for line in input_res:
        # print(line)
        line = line.strip("\r\n")
        #passwords, frequency, guess, zxcvbn = line.split("\t")
        passwords, probability, frequency, zxcvbn, guess, percentage = line.split("\t") 
        total += float(frequency)
        if float(zxcvbn)>=10**6 and float(zxcvbn)<=10**14:
            count += float(frequency)
        # if float(zxcvbn)>=10**10:
        #     count2 += float(frequency)
        if float(zxcvbn)>=10**14:
            count3 += float(frequency)
    result6= count/total
    result14= count3/total
    result= result14+result6

    print("6:" + str(result6) + "\n" + "/14:" +str(result14)+ "\n"+"total:"+str(result))