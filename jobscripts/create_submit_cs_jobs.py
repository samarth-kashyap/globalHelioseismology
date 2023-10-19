import numpy as np
import argparse
import os, sys

parser = argparse.ArgumentParser()
parser.add_argument("--nmin", help='radial order', type=int, default=0)
parser.add_argument("--nmax", help='radial order', type=int, default=25)
parser.add_argument("--t", help='radial order', type=int, default=0)
args = parser.parse_args()

pythonpath = "/home/seismogroup/anaconda3/envs/helio-sgk/bin/python"
execpath = "/scratch/seismogroup/samarth/home-sgk/globalHelioseismology/raw_data.py"

ell_list = np.load('/scratch/seismogroup/samarth/home-sgk/globalHelioseismology/data/ell_list.npy')
enn_list = np.load('/scratch/seismogroup/samarth/home-sgk/globalHelioseismology/data/enn_list.npy')

l1_arr = np.array(ell_list)
l2_arr = np.array(ell_list)
n1_arr = np.array(enn_list)
n2_arr = np.array(enn_list)

len_ell = len(l1_arr)

filename = f"/scratch/seismogroup/samarth/home-sgk/globalHelioseismology/jobscripts/ipjobs_cs_n{args.nmin}-{args.nmax}.t{args.t:02}.sh"

with open(filename, "w") as f:
    for i in range(len_ell):
        l1, l2 = int(l1_arr[i]), int(l2_arr[i])
        n1, n2 = int(n1_arr[i]), int(n2_arr[i])
        exec_cmd = (f"{pythonpath} {execpath} --l1 {l1} --l2 {l2} --n1 {n1} " +
                    f"--n2 {n2} --t {args.t}\n")
        f.write(exec_cmd)

os.system(f"python submit_jobs.py --nmin {args.nmin} --nmax {args.nmax} --t {args.t}")
