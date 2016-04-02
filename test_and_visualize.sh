#!/bin/bash

set -e

rm -rf plots

mkdir plots

# len eps alpha gamma
./ai2048 TST_SARSA $1 $2 $3 $4 | tee game_log.txt

gnuplot -p  visualize.gnuplot &

( echo "set terminal x11 size 1200,600"; for ((i=1;i< $1; i++)) do echo "set multiplot layout 1,2; plot 'plots/q_${i}.mat' matrix with image; plot 'plots/p_${i}.mat' matrix with image; unset multiplot"; done) | gnuplot -p  -
