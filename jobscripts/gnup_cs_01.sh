#!/bin/bash
#PBS -N cs.n1.t0.data
#PBS -o csout.n1.t0.log
#PBS -e cserr.n1.t0.log
#PBS -l select=1:ncpus=2:mem=4gb
#PBS -l walltime=02:30:00
#PBS -q small
echo "Starting at "`date`
cd $PBS_O_WORKDIR
cd ..
export PATH=$PATH:/home/apps/GnuParallel/bin
cd $PBS_WORKDIR
parallel --jobs 2 < $PBS_O_WORKDIR/ipjobs_cs_01_00.sh
echo "Finished at "`date`