import numpy as np
import argparse
import os, sys

parser = argparse.ArgumentParser()
parser.add_argument("--nmin", help='radial order', type=int)
parser.add_argument("--nmax", help='radial order', type=int)
parser.add_argument("--t", help='radial order', type=int, default=0)
args = parser.parse_args()

pythonpath = "/home/g.samarth/.conda/envs/helio/bin/python"
execpath = "/home/g.samarth/globalHelioseismology/cs_data.py"

nmin = args.nmin
nmax = args.nmax
t = args.t
dell = 2

data = np.loadtxt("/home/g.samarth/globalHelioseismology/mode-params/hmi.6328.36")
ln_arr = data[:, :2]
ln_list = ln_arr.astype('int').tolist()
l1arr = []
l2arr = []
n1arr = []
n2arr = []

for n in range(nmin, nmax+1):
    mask_radial = ln_arr[:, 1] == n
    larr = ln_arr[mask_radial, 0]

    for ell1 in larr:
        for fac in np.array([0, 1, 2]):
            ell2 = ell1 + fac*dell
            mode_found = True
            try:
                ln_list.index([ell2, n])
            except ValueError:
                print(f" Mode not found ell = {ell2}; n = {n}")
                mode_found = False

            if mode_found:
                l1arr.append(ell1)
                l2arr.append(ell2)
                n1arr.append(n)
                n2arr.append(n)

l1_arr = np.array(l1arr)
l2_arr = np.array(l2arr)
n1_arr = np.array(n1arr)
n2_arr = np.array(n2arr)

len_ell = len(l1_arr)

filename = f"/home/g.samarth/globalHelioseismology/jobscripts/ipjobs_cs_n{args.nmin}-{args.nmax}.t{args.t:02}.sh"

with open(filename, "w") as f:
    for i in range(len_ell):
        l1, l2 = int(l1_arr[i]), int(l2_arr[i])
        n1, n2 = int(n1_arr[i]), int(n2_arr[i])
        exec_cmd = (f"{pythonpath} {execpath} --l1 {l1} --l2 {l2} --n1 {n1} " +
                    f"--n2 {n2} --t {args.t}\n")  #--plot
        f.write(exec_cmd)

os.system(f"python submit_jobs.py --nmin {args.nmin} --nmax {args.nmax} --t {args.t}")
