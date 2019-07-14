#!/usr/bin/env bash
DIR=`ls cuda`
LOG=cuda.log
CSV=cuda.csv


rm -f $LOG $CSV
for dir in $DIR
    do
        printf "\n\n$dir\n\n" >> $LOG
        BENCH="$BENCH$dir,"
        n=`grep --color=auto -Irne  "<<<.*>>>" cuda/$dir | tee -a $LOG | wc -l`
        COUNT="$COUNT$n,"
    done
echo $BENCH >> $CSV
echo $COUNT >> $CSV


