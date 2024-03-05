# Convert log file (segmented with ' ') into csv file (segmented with ',').
# Usage: python log2csv.py <file_to_convert> [<output_file_name>]

import csv
import sys

if len(sys.argv) < 2:
    print("Usage: python log2csv.py <file_to_convert> [<output_file_name>]")
    exit
input_file = sys.argv[1]
if len(sys.argv) == 2:
    output_file_name = input_file+".csv"
else:
    output_file_name = sys.argv[2]
    
csvFile = open(output_file_name, 'w')
writer = csv.writer(csvFile)
csvRow = []
f = open(input_file, 'r')
for line in f:
    csvRow = line.split()
    writer.writerow(csvRow)
f.close()
