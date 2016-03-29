#!/usr/bin/env bash
set -e
./ai2048 RL $1 $2 > game_log.txt
gnuplot -p visualize.gnuplot
