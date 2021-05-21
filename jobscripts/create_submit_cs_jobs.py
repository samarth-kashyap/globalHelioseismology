import numpy as np
import argparse
import os, sys

parser = argparse.ArgumentParser()
parser.add_argument("--n", help='radial order', type=int)
parser.add_argument("--dell", help='delta ell', type=int)
parser.add_argument("--t", help='radial order', type=int)
args = parser.parse_args()

pythonpath = "python"
execpath = "/home/g.samarth/globalHelioseismology/cs_data.py"

n = args.n
dell = args.dell
t = args.t

data = np.loadtxt("/home/g.samarth/globalHelioseismology/mode-params/hmi.6328.36")
ln_arr = data[:, :2]
ln_list = ln_arr.astype('int').tolist()

mask_radial = ln_arr[:, 1] == args.n
larr = ln_arr[mask_radial, 0]
l1arr = []
l2arr = []

for ell1 in larr:
    ell2 = ell1 + dell
    mode_found = True
    try:
        ln_list.index([ell2, args.n])
    except ValueError:
        print(f" Mode not found ell = {ell2}; n = {args.n}")
        mode_found = False

    if mode_found:
        l1arr.append(ell1)
        l2arr.append(ell2)

l1_arr = np.array(l1arr)
l2_arr = np.array(l2arr)
len_ell = len(l1_arr)

filename = f"/home/g.samarth/globalHelioseismology/jobscripts/ipjobs_cs_{n:02d}.sh"

with open(filename, "w") as f:
    for i in range(len_ell):
        l1, l2 = int(l1_arr[i]), int(l2_arr[i])
        exec_cmd = (f"{pythonpath} {execpath} --l {l1} --lp {l2} --n {n} " +
                    f"--np {n} --t {args.t} --find_baselines\n")  #--plot
        f.write(exec_cmd)

os.system(f"python submit_jobs.py --n {args.n}")
