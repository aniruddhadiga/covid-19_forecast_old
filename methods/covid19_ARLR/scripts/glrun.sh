#!/bin/bash
gl_list="../input/gl_list.txt"
#input_cnty="../input/va_cnty_list.txt"
input_hrzn="../input/hrzn_list_curr.txt"
while IFS= read -r st
do
        echo $st
    sbatch gljob.sbatch $st $cnty $input_hrzn
            #qreg
done < $gl_list

#while IFS= read -r st
#do
#    while IFS= read -r cnty
#    do
#        echo $st $cnty
#        sbatch job.sbatch $st $cnty $input_hrzn
#    done < "$st"
#done < "$st_list"
