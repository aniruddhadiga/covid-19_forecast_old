#!/bin/bash
dt_list="input/date_file_curr.txt"
#input_cnty="../input/va_cnty_list.txt"
mtd_list="input/mtd_file.txt"
while IFS= read -r hrzn
do
    echo $hrzn
    sbatch wtsjob.sbatch $hrzn
done < $dt_list


