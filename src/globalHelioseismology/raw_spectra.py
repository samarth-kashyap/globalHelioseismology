from numpy.polynomial.legendre import legval
from astropy.io import fits
import matplotlib.pyplot as plt
from math import sqrt, pi
import healpy as hp
import numpy as np
import time
import os

from .spectra import frequencyBins
from .spectra import observedData
from ..globalvars import dirConfig

DIRS = dirConfig()

current_dir = os.path.dirname(os.path.realpath(__file__))
package_dir = DIRS.package_dir

__all__ = ["crossSpectra"]


class crossSpectra():
    """Class to compute helioseismic cross-spectra.

    """
    def __init__(self,
                 n1=0,
                 l1=200,
                 n2=0,
                 l2=200,
                 t=0,
                 instrument="hmi",
                 daynum=6328):
        # swapping values of ell if l2 < l1
        if l2 < l1:
            ltemp, ntemp = l2, n2
            l2, n2 = l1, n1
            l1, n1 = ltemp, ntemp

        self.delta_ell = abs(l2 - l1)
        self.n1, self.n2 = int(n1), int(n2)
        self.l1, self.l2 = int(l1), int(l2)
        self.t = int(t)
        self.mode_data = np.loadtxt(f"{DIRS.mode_dir}/hmi.6328.36")
        self.dirname = DIRS.output_dir
        self.fname_suffix = f"{daynum:04d}-{n1:02d}.{l1:03d}-{n2:02d}.{l2:03d}-{self.t:03d}"

        # observed data relevant to the class instance
        self.od = observedData(instrument)

        # loading frequency bins relevant to the class instance
        self.fb = frequencyBins(n1, l1, n2, l2, instrument=instrument)
        freq, idx_np, idx_diff_np, idx_derot_diff_np = self.fb.window_freq()

        self.freq, (self.idx_n, self.idx_p) = freq, idx_np
        self.idx_diff_n, self.idx_diff_p = idx_diff_np
        self.idx_derot_diff_n, self.idx_derot_diff_p = idx_derot_diff_np


    # {{{ def get_raw_cs(self, plot=False):
    def get_raw_cs(self, daynum=6328):
        afft1p, afft1n = self.od.load_time_series(self.l1, day=daynum)
        (afft2p, afft2n) = (self.od.load_time_series(self.l2, day=daynum)) if \
            self.delta_ell != 0 else (afft1p*1.0, afft1n*1.0)

        # windowing in frequency and restricting to m-values to be
        # min(l1, l2)
        afft1p = afft1p[:, self.idx_n:self.idx_p]
        afft1n = afft1n[:, self.idx_n:self.idx_p]
        afft2p = afft2p[:(self.l1+1), self.idx_n:self.idx_p]
        afft2n = afft2n[:(self.l1+1), self.idx_n:self.idx_p]

        # shifting the \phi2 by t
        if self.t != 0:
            afft2p = np.roll(afft2p, self.t, axis=0)
            afft2n = np.roll(afft2n, self.t, axis=0)

        # computing the cross-spectrum
        _csp = afft1p.conjugate()*afft2p
        _csn = afft1n.conjugate()*afft2n

        return _csp, _csn
    # }}} get_raw_cs(self, plot=False):
