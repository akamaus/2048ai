set key right center
set multiplot layout 2,1
plot 'game_log.txt' using 2:10 with lines title "avg score", 'game_log.txt' using 2:14 with lines title "min score", 'game_log.txt' using 2:16 with lines title "max score"
plot 'game_log.txt' using 2:4 with lines title 'Q size', 'game_log.txt' using 2:6 with lines title 'policy size'
unset multiplot
