#!/bin/bash
#PBS -N cs.n01.data
#PBS -o csout.n01.log
#PBS -e cserr.n01.log
#PBS -l select=1:ncpus=32:mem=64gb
#PBS -l walltime=01:30:00
#PBS -q small
echo "Starting at "`date`
cd $PBS_O_WORKDIR
export PATH=$PATH:/home/apps/GnuParallel/bin
export TERM=xterm
cd $PBS_WORKDIR
echo $PBS_JOBID
module load anaconda3
conda activate helio
echo `which python`
parallel --jobs 32 < $PBS_O_WORKDIR/ipjobs_cs_01.sh
echo "Finished at "`date`
