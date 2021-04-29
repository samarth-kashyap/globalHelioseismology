import numpy as np
import argparse
import os, sys

parser = argparse.ArgumentParser()
parser.add_argument("--n", help='radial order',
                    type=int)
parser.add_argument("--ps", help='power spectrum',
                    action="store_true")
args = parser.parse_args()

pythonpath = "/home/g.samarth/anaconda3/bin/python"
execpath = "/home/g.samarth/Woodard2013/WoodardPy/cs_data.py"
n = args.n
l1_arr = np.load(f"/scratch/g.samarth/csfit/l1_{n:02d}.npy")
l2_arr = np.load(f"/scratch/g.samarth/csfit/l2_{n:02d}.npy")

if args.ps:
    l_arr = np.append(l1_arr, l2_arr)
    l_arr = np.unique(l_arr)
    l_arr = np.sort(l_arr)
    l1_arr = l_arr
    l2_arr = l_arr

len_ell = len(l1_arr)
if args.ps:
    filename = f"/home/g.samarth/Woodard2013/job_scripts/ipjobs_ps_{n:02d}.sh"
else:
    filename = f"/home/g.samarth/Woodard2013/job_scripts/ipjobs_cs_{n:02d}.sh"

with open(filename, "w") as f:
    for i in range(len_ell):
        l1, l2 = int(l1_arr[i]), int(l2_arr[i])
        exec_cmd = (f"{pythonpath} {execpath} --l {l1} --lp {l2} --n {n} --np {n} " +
                    f"--find_baselines\n")  #--plot
        f.write(exec_cmd)
