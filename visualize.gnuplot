if (!exists("datafile")) datafile = 'game_log.txt'

set key right center
set multiplot layout 2,1
plot datafile using 2:10 with lines title "avg score", datafile using 2:14 with lines title "min score", datafile using 2:16 with lines title "max score"
plot datafile using 2:4 with lines title 'Q size', datafile using 2:6 with lines title 'policy size'
unset multiplot
