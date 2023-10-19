# {{{ Library imports
import os
from tqdm import tqdm
import numpy as np
import argparse
from src import globalHelioseismology as GH

# {{{ ArgumentParser
parser = argparse.ArgumentParser()
parser.add_argument("--l1",
                    help="spherical harmonic degree for mode 1",
                    type=int, default=50)
parser.add_argument("--l2",
                    help="spherical harmonic degree for mode 2",
                    type=int, default=52)
parser.add_argument("--n1", help="radial order for mode 1", type=int, default=5)
parser.add_argument("--n2", help="radial order for mode 2", type=int, default=5)
parser.add_argument("--t", help="m - mp", type=int, default=0)
parser.add_argument("--plot",
                    help="plot results after computation",
                    action="store_true", default=True)
ARGS = parser.parse_args()
# }}} parser

daynum_start = 6328
day_diff = 72
num_blocks = 10
daylist = np.arange(daynum_start, daynum_start+num_blocks*day_diff, day_diff)
daylist = daylist.astype('int')

for day in tqdm(daylist, desc='Raw spectra for HMI blocks'):
    cs = GH.raw_spectra.crossSpectra(ARGS.n1, ARGS.l1, ARGS.n2, ARGS.l2, ARGS.t, daynum=day, instrument='HMI')
    csp, csn = cs.get_raw_cs(daynum=day)
    store_dir = f"{cs.dirname}/{day:04d}"
    if not os.path.exists(store_dir): os.makedirs(store_dir)
    fname_p = f"{store_dir}/{cs.fname_suffix}-csp.npy"
    fname_n = f"{store_dir}/{cs.fname_suffix}-csn.npy"
    np.save(fname_p, csp)
    np.save(fname_n, csn)
