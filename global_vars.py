"""Handles global variables"""
import getpass
import numpy as np


class globalVars():
    def __init__(self):
        username = getpass.getuser()
        if username == "g.samarth":
            local_dir = "/home/g.samarth/Woodard2013"
            self.data = np.loadtxt(f'{local_dir}/WoodardPy/HMI/hmi.6328.36')
            self.writedir = "/scratch/g.samarth/csfit/"
            self.outdir = "/scratch/g.samarth/csfit/"
            self.plotdir = f"{self.outdir}figures/"
            self.leakdir = "/scratch/g.samarth/HMIDATA/leakmat/"
            self.HMI_datadir = f"/scratch/seismogroup/data/HMI/data/"
            self.username = username
        elif username == "sbdas":
            scratch_helio = "/scratch/gpfs/sbdas/Helioseismology/"
            self.data = np.loadtxt(f'{scratch_helio}HMI/hmi.6328.36')
            self.outdir = f"{scratch_helio}output_files/"
            self.plotdir = f"{self.outdir}figures/"
            self.writedir = f"{scratch_helio}HMI/Samarth_Data/"
            self.leakdir = f"{scratch_helio}HMI/"
            self.HMI_datadir = f"{scratch_helio}HMI/"
            self.username = username
        else:
            local_dir = "/Users/srijanbharatidas/Documents/Research/" +\
                "Codes/Helioseismology/DR_Bayesian/Woodard2013"
            self.data = np.loadtxt(f'{local_dir}/WoodardPy/HMI/hmi.6328.36')
            self.outdir = f"{local_dir}/output_files/"
            self.plotdir = f"{self.outdir}figures/"
            self.writedir = f"{local_dir}/WoodardPy/HMI/Samarth_Data/"
            self.leakdir = f"{local_dir}/WoodardPy/HMI/"
            self.HMI_datadir = f"{local_dir}/WoodardPy/HMI/"
            self.username = username
