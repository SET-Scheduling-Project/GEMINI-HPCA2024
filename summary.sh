# Summarize an experiment's result.
# Usage: ./summary.sh <target_dir> [max_dir_depth]
# `max_dir_depth`: maximum recursion depth of directories.

# Number of ERRORs.
num_err=0
summary() {
    if [ $2 -lt 0 ]; then
        echo "[ERROR] Directory is too deep to summarize recursively."
        exit
    fi
    pushd `pwd` > /dev/null
    cd $1
    if [ -f $search_log ];  then 
        cat $search_log >> $search_summary; 
    fi
    if [ -f $output_point_file ]; then 
        cat $output_point_file >> $result_summary; 
    fi
    if [ -f $err_log ]; then
        if [ -s $err_log ]; then
            (( num_err++ ))
            echo -e "[ERROR RECORD] No.$num_err" >> $error_summary
            echo -e "[ERROR INPUT] " >> $error_summary
            cat $search_log >> $error_summary
            echo -e "[ERROR INFO]" >> $error_summary
            cat $err_log >> $error_summary
            echo "" >> $error_summary
        fi
    fi
    for sub_dir in `ls`
    do  
        if [ -d $sub_dir ]; then
            summary $sub_dir `expr $2 - 1`
        fi
    done
    popd > /dev/null
}

output_path=`readlink -f $1`
output_point_file="result.log"
output_struct_file="out.log"
err_log="err.log"
search_log="search.log"

# Summarize the experiment result.
search_summary=$output_path/search.log
error_summary=$output_path/error.log
result_summary=$output_path/result.log
result_summary_xlsx=$output_path/result.xlsx
if [ -f $search_summary ]; then rm $search_summary; fi
if [ -f $error_summary ]; then rm $error_summary; fi
if [ -f $result_summary ]; then rm $result_summary; fi
touch $search_summary
touch $error_summary
touch $result_summary

if [ $# -lt 2 ]; then
    summary $1 13
else
    summary $1 $2
fi

headers="tech,mm,nn,xx,yy,ss,bb,rr,ff,xcut,ycut,package_type,IO_type,nop_bw,ddr_type,ddr_bw,noc,mac,ul3,tops,cost_overall,energy,cycle,edp,cost,idx,ubuf,buf,bus,mac,NoC,NoP,DRAM,compute_die_area,IO_die_area,total_die_area,cosy_chip,cost_package,cost_system_package,cost_soc"

python3 pyscripts/log2csv.py $result_summary $output_path/result.csv
sed "1i\\$headers" -i $output_path/result.csv
python3 pyscripts/csv2xlsx.py $output_path/result.csv $result_summary_xlsx
