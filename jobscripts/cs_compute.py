import os, sys
from multiprocessing import Pool

def get_cs(ecmd):
    os.system(ecmd)
    return None

pythonpath = "/home/g.samarth/anaconda3/bin/python"
execpath = "/home/g.samarth/Woodard2013/WoodardPy/cs_data.py"
lmin, lmax, n = 50, 100, 5
with Pool(32) as pool:
    for ell in range(lmin, lmax+1):
        for i in range(3):
            l1, l2 = ell, ell + i*1
            exec_cmd = (f"{pythonpath} {execpath} --l {l1} --lp {l2} --n {n} --np {n} ")
            print(exec_cmd)

            
    

