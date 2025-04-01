# Design Space Experiment of 512TOPS
# Script Usage: ./dse.sh [output_folder]

# Execute `stschedule` once.
# Usage: search $tech $mm $nn $xx $yy $ss $bb $rr $ff $xcut $ycut $package_type $IO_type $_NoP_bw $DDR_type $_DRAM_bw $_NoC_bw $_mac_num $_ul3 $TOPS result.log out.log err.log search.log
search(){
    start=$(date +%s)
    echo "$1 $2 $3 $4 $5 $6 $7 $8 $9 ${10} ${11} ${12} ${13} ${14} ${15} ${16} ${17} ${18} ${19} ${20}" | ./build/stschedule ${21} > ${22} 2> ${23}
    end=$(date +%s)
    iter_time=$(( end - start ))
    printf "INPUT(%s %1d %2d %3d %3d %3d %3d %3d %2d %3d %3d %s %s %3d %s %8d %3d %4d %4d %8d) ITERATION_TIME(%5d)seconds\n" $1 $2 $3 $4 $5 $6 $7 $8 $9 ${10} ${11} ${12} ${13} ${14} ${15} ${16} ${17} ${18} ${19} ${20} $iter_time >> ${24}
    return $iter_time
}

experiment() {
    echo '************************* Start Experiment *************************'
    echo "[PROMPT] Check the output in dir $output_path."
    network_range=(17)
    #network_range=(16)
    DRAM_bw_range=(`expr $TOPS / 2` $TOPS `expr $TOPS \* 2` `expr $TOPS \* 4`)
    # NoC_bw_range=(8 16 32 64)
    # ul3_range=(512 1024 2048 4096)
    # mac_num_range=(512 1024 2048 4096)
    NoC_bw_range=(16 32 64)
    ul3_range=(1024 2048 4096)
    mac_num_range=(2048 4096 8192)
    tech_range=("7" "12")
    package_type_range=("OS" "FO" "SI")
    # package_type_range=("OS" "SI")
    IO_type_range=("XSR" "USR" "UCIe")
    DDR_type_range=("GDDR6X")

    start_time=$(date +%s)
    
    MAX_PROCESSES=75
    CURRENT_PROCESSES=0

    # tech
    for tech in ${tech_range[*]}
    do
        temp_path_tech=$output_path/tech_${tech}
        mkdir $temp_path_tech

        # package_type
        for package_type in ${package_type_range[*]}
        do
            temp_path_package=$temp_path_tech/package_${package_type}
            mkdir $temp_path_package

            # IO_type
            for IO_type in ${IO_type_range[*]}
            do
                temp_path_io=$temp_path_package/IO_${IO_type}
                mkdir $temp_path_io

                # DDR_type
                for DDR_type in ${DDR_type_range[*]}
                do
                    temp_path_ddr=$temp_path_io/DDR_${DDR_type}
                    mkdir $temp_path_ddr

                    # network_range
                    for nn in ${network_range[*]}
                    do  
                        temp_path0=$temp_path_ddr/${network[$nn]}
                        mkdir $temp_path0

                        # DRAM_bw_range
                        for _DRAM_bw in ${DRAM_bw_range[*]}
                        do
                            temp_path1=$temp_path0/DRAM_bw_${_DRAM_bw}
                            mkdir $temp_path1          

                            # NoC_bw_range
                            for _NoC_bw in ${NoC_bw_range[*]}
                            do
                                temp_path2=$temp_path1/NoC_bw_${_NoC_bw}
                                mkdir $temp_path2

                                # ul3_range
                                for _ul3 in ${ul3_range[*]}
                                do
                                {
                                    temp_path3=$temp_path2/ul3_${_ul3}
                                    mkdir $temp_path3

                                    # _NoP_bw
                                    for _NoP_bw in `expr $_NoC_bw / 2` $_NoC_bw
                                    do
                                        temp_path4=$temp_path3/NoP_bw_${_NoP_bw}
                                        mkdir $temp_path4

                                        # mac_num_range
                                        for _mac_num in ${mac_num_range[*]}
                                        do
                                            temp_path5=$temp_path4/mac_num_${_mac_num}                                
                                            mkdir $temp_path5

                                            # xx 和 yy
                                            x2=`expr $TOPS / 2 / $_mac_num`
                                            xx=$(echo "sqrt($x2)" | bc)
                                            if [ `expr $xx \* $xx` -eq $x2 ]; then
                                                yy=$xx
                                            else
                                                x2=`expr $x2 / 2`
                                                yy=$(echo "sqrt($x2)" | bc)
                                                xx=`expr 2 \* $yy`
                                            fi
                                            ss=`expr $xx / 2`

                                            # cut_idx
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
                                                if [ `expr $xcut \* $ycut` -eq 1 -a $_NoP_bw -ne $_NoC_bw ]; then continue; fi

                                                temp_path_innerest=$temp_path5/xcut_${xcut}_ycut_${ycut}
                                                mkdir $temp_path_innerest

                                                touch $temp_path_innerest/$search_log
                                                touch $temp_path_innerest/$err_log
                                                touch $temp_path_innerest/$output_point_file
                                                touch $temp_path_innerest/$output_struct_file

                                                if [ $CURRENT_PROCESSES -ge $MAX_PROCESSES ]; then
                                                    wait -n  # 等待一个进程完成
                                                    CURRENT_PROCESSES=$((CURRENT_PROCESSES - 1))
                                                fi

                                                search $tech $mm $nn $xx $yy $ss $bb $rr $ff $xcut $ycut $package_type $IO_type $_NoP_bw $DDR_type $_DRAM_bw $_NoC_bw $_mac_num $_ul3 $TOPS "$temp_path_innerest/$output_point_file" "$temp_path_innerest/$output_struct_file" "$temp_path_innerest/$err_log" "$temp_path_innerest/$search_log" &
                                                CURRENT_PROCESSES=$((CURRENT_PROCESSES + 1))
                                            done
                                        done
                                    done
                                } 
                                done
                            done
                        done
                    done
                done
            done
        done
    done

    wait

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
network=(darknet19 vgg resnet50 googlenet resnet101 densenet ires gnmt lstm zfnet transformer transformer_cell pnasnet resnext50 resnet152 bert_block GPT2_prefill_block GPT2_decode_clock)
# chiplet number : 1 2 4 8 16 32 64
cut_long=(1 2 2 4 4 8 8)
cut_short=(1 1 2 2 4 4 8)
mm=0
TOPS=`expr 512 \* 1024`
rr=50
bb=1
ff=1

# ************************ Execute Experiment ************************
echo "******************** 512TOPS DSE Start ********************"
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
echo "******************** 512TOPS DSE Finish *******************"
