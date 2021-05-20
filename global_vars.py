"""Handles global variables"""
import getpass
import numpy as np


class globalVars():
    def __init__(self):
        local_dir = "/home/g.samarth/globalHelioseismology"
        self.data = np.loadtxt(f'{local_dir}/mode-params/hmi.6328.36')
        self.writedir = "/scratch/g.samarth/globalHelioseismology"
        self.outdir = "/scratch/g.samarth/globalHelioseismology"
        self.plotdir = f"{self.outdir}/figures"
        self.HMI_datadir = f"/scratch/seismogroup/data/HMI/data"
