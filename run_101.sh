#!/usr/bin/env bash

for i in {0..9};
do
python run_101_learning_split_2samp_R1_randomization.py $i &
done

wait
