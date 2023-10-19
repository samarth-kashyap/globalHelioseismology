import numpy as np
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--nmin", help='radial order', type=int, default=0)
parser.add_argument("--nmax", help='radial order', type=int, default=25)
parser.add_argument("--t", help='radial order', type=int, default=0)
args = parser.parse_args()

fname = f"gnup_cs_n{args.nmin}-{args.nmax}.t{args.t:02d}.sh"
with open(fname, "w") as f:
    f.write(f"""#!/bin/bash
#PBS -N cs.n{args.nmin}.{args.nmax}.t{args.t}.data
#PBS -o csout.n{args.nmin}.{args.nmax}.t{args.t}.log
#PBS -e cserr.n{args.nmin}.{args.nmax}.t{args.t}.log
#PBS -l select=1:ncpus=16:mem=32gb
#PBS -l walltime=12:30:00
#PBS -q large
echo \"Starting at \"`date`
cd $PBS_O_WORKDIR
cd ..
export PATH=$PATH:/home/apps/GnuParallel/bin
cd $PBS_WORKDIR
parallel --jobs 16 < $PBS_O_WORKDIR/ipjobs_cs_n{args.nmin}-{args.nmax}.t{args.t:02d}.sh
echo \"Finished at \"`date`""")
os.system(f"qsub {fname}")
