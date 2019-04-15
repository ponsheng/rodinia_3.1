#!/usr/bin/env bash
PROJECT=openmp
DIR=`ls $PROJECT`
LOG=$PROJECT.log
CSV=$PROJECT.csv

rm -f $LOG $CSV
for dir in $DIR
    do
        printf "\n\n$dir\n\n" >> $LOG
        BENCH="$BENCH$dir,"
        n=`grep --color=auto -Irne  "parallel for"  $PROJECT/$dir | tee -a $LOG | wc -l`
        COUNT="$COUNT$n,"
        n2=`grep --color=auto -Irne  "omp target"  $PROJECT/$dir | tee -a $LOG | wc -l`
        n3=`grep --color=auto -Irne  "omp target data"  $PROJECT/$dir | tee -a $LOG | wc -l`
        n4=`grep --color=auto -Irne  "omp for"  $PROJECT/$dir | tee -a $LOG | wc -l`
        COUNT2="$COUNT2$n2,"
        COUNT3="$COUNT3$n3,"
        COUNT4="$COUNT4$n4,"
    done
echo "bench,"$BENCH >> $CSV
echo "parallel_for,"$COUNT >> $CSV
echo "omp_for,"$COUNT4 >> $CSV
echo "omp_target,"$COUNT2 >> $CSV
echo "omp_target_data"$COUNT3 >> $CSV


