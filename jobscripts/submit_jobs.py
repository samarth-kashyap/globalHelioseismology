import numpy as np
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--n", help='radial order', type=int)
args = parser.parse_args()

fname = f"gnup_cs_{args.n:02}.sh"
with open(fname, "w") as f:
    f.write(f"""#!/bin/bash
#PBS -N cs.n{args.n:02}.data
#PBS -o csout.n{args.n:02}.log
#PBS -e cserr.n{args.n:02}.log
#PBS -l select=1:ncpus=4:mem=8gb
#PBS -l walltime=01:30:00
#PBS -q long
echo \"Starting at \"`date`
cd $PBS_O_WORKDIR
export PATH=$PATH:/home/apps/GnuParallel/bin
export TERM=xterm
cd $PBS_WORKDIR
echo $PBS_JOBID
module load anaconda3
conda activate helio
echo `which python`
parallel --jobs 32 < $PBS_O_WORKDIR/ipjobs_cs_{args.n:02d}.sh
echo \"Finished at \"`date`""")
os.system(f"qsub {fname}")
