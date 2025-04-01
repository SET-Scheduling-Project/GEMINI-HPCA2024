# Find the best arch within DSE log.
# Usage: python best_arch.py <dse_log_csv> [<output_file>]

import pandas as pd
import sys

def format_number(num):
    # Check if the number is an integer
    if num % 1 == 0:
        return int(num)
    else:
        return num

def find_best_arch(csv_file_name, output_file=None):
    # Load the CSV file
    df = pd.read_csv(csv_file_name)
    
    # Convert the 'cost' column to numeric, errors will be converted to NaN
    df['cost'] = pd.to_numeric(df['cost'], errors='coerce')
    
    # Find the row with the minimum value in the 'cost' column
    min_cost_row = df.loc[df['cost'].idxmin()]
    
    # Get the relevant data from the row
    tech = min_cost_row['tech']
    mm = format_number(min_cost_row['mm'])
    xx = format_number(min_cost_row['xx'])
    yy = format_number(min_cost_row['yy'])
    ss = format_number(min_cost_row['ss'])
    ff = format_number(min_cost_row['ff'])
    xcut = format_number(min_cost_row['xcut'])
    ycut = format_number(min_cost_row['ycut'])
    package_type = min_cost_row['package_type']
    IO_type = min_cost_row['IO_type']
    nop = format_number(min_cost_row['nop_bw'])
    ddr_type = min_cost_row['ddr_type']
    ddr = format_number(min_cost_row['ddr_bw'])
    noc = format_number(min_cost_row['noc'])
    mac = format_number(min_cost_row['mac'])
    ul3 = format_number(min_cost_row['ul3'])
    tops = format_number(min_cost_row['tops'])

    numbers = (tech, mm, xx, yy, ss, ff, xcut, ycut, package_type, IO_type, nop, ddr_type, ddr, noc, mac, ul3, tops)
    if (output_file == None):
        output_file = "best_arch.txt"
    with open(output_file, 'w') as file:
        numbers_str = ' '.join(str(num) for num in numbers)
        file.write(numbers_str)
        print(f"Arch parameters have been writed into {output_file}")

    # Format and output the result
    print(f'Best Arch is: tech = {tech},')
    print(f'core number of X line= {xx},')
    print(f'core number of Y line = {yy}.')
    print(f'XCut= {xcut}, YCut= {ycut}.')
    print(f'package type = {package_type},')
    print(f'IO type = {IO_type},')
    print(f'DDR type = {ddr_type}')
    print(f'NoP bandwidth = {nop} GB/s.')
    print(f'NoC bandwidth = {noc} GB/s.')
    print(f'DDR bandwidth = {int(ddr / 1024)} GB/s.')
    print(f'mac per core = {mac}, UBUF size = {ul3} KB.')

if len(sys.argv) < 2:
    print("Usage: python best_arch.py <dse_log_csv> [<output_file>]")
    exit

if len(sys.argv) < 3:
    find_best_arch(sys.argv[1])
else:
    find_best_arch(sys.argv[1], sys.argv[2])
