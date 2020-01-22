#!/usr/bin/env bash

export SUBJECTS_DIR="PATH/freesurfer_all"

subjects_file='PATH/LIFE16_preprocessed_subjects_list_n2557.txt'
subjects_list=`cat ${subjects_file}`
out_base="PATH/metrics"

echo "" > parcstats_err.txt

for s in $subjects_list
    do
    echo $s
    subject_out_dir=${out_base}/${s}/parcstats
    cmd="mkdir ${subject_out_dir}"
    echo $cmd
    $cmd
    for h in lh rh
        do
        for m in thickness volume area
            do
            cmd="aparcstats2table --hemi ${h} --subjects ${s} --parc aparc.a2009s --meas ${m} --tablefile ${subject_out_dir}/aparc.${h}.a2009s.${m} 2>> parcstats_err.txt"
            echo $cmd
            $cmd
        done
    done

    cmd="asegstats2table --subjects ${s} --meas volume --tablefile ${subject_out_dir}/aseg 2>> parcstats_err.txt"
    echo $cmd
    $cmd

    echo
    echo

done
