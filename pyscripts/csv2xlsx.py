# Convert csv file into xlsx file.
# Usage: python csv2xlsx.py <csv_file> [<xlsx_file>]

from pandas.io.excel import ExcelWriter
import pandas as pd
import sys

if len(sys.argv) < 2:
    print("Usage: python csv2xlsx.py <csv_file> [<xlsx_file>]")
    exit
if len(sys.argv) == 2:
    output_file_name = sys.argv[1]+".xlsx"
else:
    output_file_name = sys.argv[2]

with ExcelWriter(output_file_name) as ew:
	# 将csv文件转换为excel文件
	pd.read_csv(sys.argv[1]).to_excel(ew, sheet_name="sheet1", index=False)
