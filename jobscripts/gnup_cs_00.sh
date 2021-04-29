#!/bin/bash
#PBS -N cs.n00.data
#PBS -o csout.n00.log
#PBS -e cserr.n00.log
#PBS -l select=1:ncpus=32:mem=64gb
#PBS -l walltime=01:30:00
#PBS -q large
echo "Starting at "`date`
cd /home/g.samarth/Woodard2013/
export PATH=$PATH:/home/apps/GnuParallel/bin
export TERM=xterm
cd $PBS_WORKDIR
echo $PBS_JOBID
parallel --jobs 32 < /home/g.samarth/Woodard2013/job_scripts/ipjobs_cs_00.sh
echo "Finished at "`date`
