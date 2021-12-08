#!/bin/bash
#PBS -N cs.n0.t7.data
#PBS -o csout.n0.t7.log
#PBS -e cserr.n0.t7.log
#PBS -l select=1:ncpus=4:mem=8gb
#PBS -l walltime=01:30:00
#PBS -q large
echo "Starting at "`date`
cd $PBS_O_WORKDIR
cd ..
export PATH=$PATH:/home/apps/GnuParallel/bin
cd $PBS_WORKDIR
parallel --jobs 4 < $PBS_O_WORKDIR/ipjobs_cs_00_07.sh
echo "Finished at "`date`