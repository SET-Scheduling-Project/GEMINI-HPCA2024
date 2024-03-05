# Design Space Experiment of 128TOPS
# Script Usage: ./128tops_dse.sh [output_folder]

# Execute `stschedule` once.
# Usage: search $mm $nn $xx $yy $ss $bb $rr $ff $xcut $ycut $_serdes_lane $_DRAM_bw $_NoC_bw $_mac_num $_ul3 $TOPS result.log out.log err.log search.log
search(){
    start=$(date +%s)
    echo "$1 $2 $3 $4 $5 $6 $7 $8 $9 ${10} ${11} ${12} ${13} ${14} ${15} ${16}" | ./build/stschedule ${17} > ${18} 2> ${19}
    end=$(date +%s)
    iter_time=$(( end - start ))
    printf "INPUT(%1d %2d %3d %3d %3d %3d %3d %2d %3d %3d %3d %8d %3d %4d %4d %8d) ITERATION_TIME(%5d)seconds\n" $1 $2 $3 $4 $5 $6 $7 $8 $9 ${10} ${11} ${12} ${13} ${14} ${15} ${16} $iter_time >> ${20}
    return $iter_time
}

experiment() {
    echo '************************* Start Experiment *************************'
    echo "[PROMPT] Check the output in dir $output_path."
    network_range=(11)
    DRAM_bw_range=(`expr $TOPS / 2` $TOPS `expr $TOPS \* 2` `expr $TOPS \* 4`)
    NoC_bw_range=(8 16 32 64)
    ul3_range=(256 512 1024 2048 4096)
    mac_num_range=(512 1024 2048 4096)

    start_time=$(date +%s)

    for nn in ${network_range[*]}
    do  
        temp_path0=$output_path/${network[$nn]}
        mkdir $temp_path0
        for _DRAM_bw in ${DRAM_bw_range[*]}
        do
            temp_path1=$temp_path0/DRAM_bw_${_DRAM_bw}
            mkdir $temp_path1          
            for _NoC_bw in ${NoC_bw_range[*]}
            do
                temp_path2=$temp_path1/NoC_bw_${_NoC_bw}
                mkdir $temp_path2
                for _ul3 in ${ul3_range[*]}
                do
                {
                    temp_path3=$temp_path2/ul3_${_ul3}
                    mkdir $temp_path3
                    for _serdes_lane in `expr $_NoC_bw / 2` $_NoC_bw
                    do
                        temp_path4=$temp_path3/NoP_bw_${_serdes_lane}
                        mkdir $temp_path4
                        for _mac_num in ${mac_num_range[*]}
                        do
                            temp_path5=$temp_path4/mac_num_${_mac_num}                                
                            mkdir $temp_path5
                            x2=`expr $TOPS / 2 / $_mac_num`
                            # xx >= yy
                            xx=$(echo "sqrt($x2)" | bc)
                            if [ `expr $xx \* $xx` -eq $x2 ]; then
                            # x2 is an even power of 2.
                                yy=$xx
                            else
                            # x2 is an odd power of 2.
                                x2=`expr $x2 / 2`
                                yy=$(echo "sqrt($x2)" | bc)
                                xx=`expr 2 \* $yy`
                            fi
                            ss=`expr $xx / 2`
                            for cut_idx in `seq 0 1 $(( ${#cut_long[*]} - 1))`
                            do
                                if [ $xx -gt $yy ]; then
                                    xcut=${cut_long[$cut_idx]}
                                    ycut=${cut_short[$cut_idx]}
                                else
                                    ycut=${cut_long[$cut_idx]}
                                    xcut=${cut_short[$cut_idx]}
                                fi
                                # Abandon invalid chiplet partition schemes.
                                if [ `expr $xx % $xcut` -ne 0 -o `expr $yy % $ycut` -ne 0 ]; then continue; fi
                                if [ `expr $xcut \* $ycut` -eq 1 -a $_serdes_lane -ne $_NoC_bw ]; then continue; fi
                                temp_path_innerest=$temp_path5/xcut_${xcut}_ycut_${ycut}
                                mkdir $temp_path_innerest
                                touch $temp_path_innerest/$search_log
                                touch $temp_path_innerest/$err_log
                                touch $temp_path_innerest/$output_point_file
                                touch $temp_path_innerest/$output_struct_file
                                search $mm $nn $xx $yy $ss $bb $rr $ff $xcut $ycut $_serdes_lane $_DRAM_bw $_NoC_bw $_mac_num $_ul3 $TOPS "$temp_path_innerest/$output_point_file" "$temp_path_innerest/$output_struct_file" "$temp_path_innerest/$err_log" "$temp_path_innerest/$search_log"
                            done
                        done
                    done
                } &
                done
            done
        done
    done

    wait

    # Print experiment time.
    end_time=$(date +%s)
    search_time=$(( end_time - start_time ))
    _second=`expr $search_time % 60`
    _minute=`expr $search_time / 60`
    _hour=`expr $_minute / 60`
    _minute=`expr $_minute % 60`
    echo "[PROMPT] Experiment time: $_hour h $_minute m $_second s ($search_time)seconds."
    echo '************************ Finish Experiment *************************'
}

# *************************** Output File ****************************
output_point_file="result.log"
output_struct_file="out.log"
err_log="err.log"
search_log="search.log"

# Make output dir and files.
temp_date=$(date "+%Y_%m_%d_%H_%M_%S")
if [ $# -lt 1 ]; then
    if [ ! -d "dse_log" ]; then
        mkdir "dse_log"
    fi
    output_path=`pwd`/dse_log/${temp_date}
else
    output_path=`readlink -f $1`
fi
mkdir $output_path

# The max depth of output dir.
max_dir_depth=16

# ************************* Other Parameters *************************
network=(darknet19 vgg resnet50 googlenet resnet101 densenet ires gnmt lstm zfnet transformer transformer_cell pnasnet resnext50 resnet152)
# chiplet number : 1 2 4 8 16 32 64
cut_long=(1 2 2 4 4 8 8)
cut_short=(1 1 2 2 4 4 8)
mm=0
TOPS=`expr 128 \* 1024`
rr=150
bb=64
ff=1

# ************************ Execute Experiment ************************
echo "******************** 128TOPS DSE Start ********************"
echo "[INFO] Check output in dir $output_path."
echo "[INFO] DSE running..."
experiment > $output_path/dse.log 2> $output_path/dse.err.log
# Summarize the experiment result.
echo "[INFO] Summarizing experiment results..."
./summary.sh $output_path $max_dir_depth
echo "[INFO] Summary at $output_path/result.xlsx"
echo "[INFO] Summary accomplished."
echo "[INFO] Best Arch found."
python3 pyscripts/best_arch.py $output_path/result.csv $output_path/best_arch.txt
echo "******************** 128TOPS DSE Finish *******************"
