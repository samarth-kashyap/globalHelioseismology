import numpy as np
import argparse
from scipy.signal import savgol_filter as savgol
from numpy.polynomial.legendre import legval
import matplotlib.pyplot as plt
import sys
from math import sqrt

sys.path.append("/home/g.samarth/Woodard2013")
from WoodardPy.helioPy import datafuncs as cdata

WINLEN = 25
POLYORD = 5

parser = argparse.ArgumentParser()
parser.add_argument("--l",
                    help="spherical harmonic degree for mode 1",
                    type=int)
parser.add_argument("--lp",
                    help="spherical harmonic degree for mode 2",
                    type=int)
parser.add_argument("--n", help="radial order for mode 1", type=int)
parser.add_argument("--np", help="radial order for mode 2", type=int)
parser.add_argument("--t", help="m - mp", type=int)
parser.add_argument("--load_smooth", help="use smooth spectra", action="store_true")
args = parser.parse_args()
# }}} parser


# {{{ def finda1(data, l, n, m):
def finda1(data, l, n, m):
    """
    Find a coefficients for given l, n, m
    """
    L = sqrt(l*(l+1))
    try:
        modeindex = np.where((data[:, 0] == l) * (data[:, 1] == n))[0][0]
    except IndexError:
        print(f"MODE NOT FOUND : l = {l}, n = {n}")
        return None, None, None
    splits = np.append([0.0], data[modeindex, 12:48])
    totsplit = legval(1.0*m/L, splits)*L
    return totsplit - 31.7*m
# }}} finda1(data, l, n, m)



# {{{ def get_freqwin(phi, l, n, freq, winhalflen, sgn):
def get_freqwin(l, n, freq, winhalflen, sgn):
    """Derotate the given cross-spectra
    Inputs:
    -------
    phi - np.ndarray(ndim=2)
        frequency series.
    l - int
        spherical harmonic degree
    n - int
        radial order
    freq - np.ndarray(ndim=1)
        array containing frequency bins
    winhalflen - int
        half the number of frequency bins for windowing
    sgn - int
        sgn = +1 for m >= 0
        sgn = -1 for m  < 0

    Outputs:
    --------
    phinew - np.ndarray(ndim=2)
        derotated cross spectra
    freq_win - np.ndarray(ndim=2)
        derotated frequency array for every m
    """
    data = np.loadtxt("/home/g.samarth/Woodard2013/WoodardPy/HMI/hmi.6328.36")
    freq_win = np.zeros(2*winhalflen + 1)

    cenfreq, __, __ = cdata.findfreq(data, l, n, 0)
    omeganl0 = cenfreq + finda1(data, l, n, 0) * 1e-3
    omeganl0_ind = np.argmin(np.abs(freq-omeganl0))
    _indmin = omeganl0_ind - winhalflen
    _indmax = omeganl0_ind + winhalflen + 1
    freq_win = freq[_indmin:_indmax]
    return freq_win
# }}} get_freqwin(phi, l, n, freq, winhalflen, sgn)


tsLen = 138240  # array length of the time series
t = np.linspace(0, 72*24*3600, tsLen)
dt = t[1] - t[0]
freq = np.fft.fftfreq(t.shape[0], dt)*1e6
df = freq[1] - freq[0]  # frequencies in microHz


writedir = f"/scratch/g.samarth/globalHelioseismology/csdata_{args.n:02d}"
plotdir = f"/scratch/g.samarth/globalHelioseismology/plots/smooth-spectrum"
fname_suffix = f"{args.n:02d}_{args.l:03d}_{args.lp:03d}"
print(f"Generating plots {fname_suffix}")
if args.t != 0:
    fname_suffix += f"_{args.t:03d}"
if args.load_smooth:
    fname_suffix += "_smooth"
fname_suffix += ".npy"

# loading data
csp = np.load(f"{writedir}/csp_data_{fname_suffix}")
csm = np.load(f"{writedir}/csm_data_{fname_suffix}")
varp = np.load(f"{writedir}/variance_p_{fname_suffix}")
varn = np.load(f"{writedir}/variance_n_{fname_suffix}")

cspsum = (csp.imag).sum(axis=0)
csmsum = (csm.imag).sum(axis=0)
stdp = np.sqrt(varp.imag)
stdn = np.sqrt(varn.imag)

if not args.load_smooth:
    cspsum_smooth = savgol(cspsum, WINLEN, POLYORD)
    csmsum_smooth = savgol(csmsum, WINLEN, POLYORD)
else:
    cspsum_smooth = cspsum
    csmsum_smooth = csmsum

winhalflen = int((len(cspsum) - 1)//2)

freq_win_p = get_freqwin(args.l, args.n, freq, winhalflen, 1.0)
freq_win_n = get_freqwin(args.l, args.n, freq, winhalflen, -1.0)

fig, axs = plt.subplots(2, figsize=(5, 6))
axs.flatten()[0].fill_between(freq_win_p, cspsum-stdp, cspsum+stdp,
                              color='gray', alpha=0.4, label="1$\sigma$")
if not args.load_smooth:
    axs.flatten()[0].plot(freq_win_p, cspsum, 'k', label="Raw spectrum")
axs.flatten()[0].plot(freq_win_p, cspsum_smooth, 'k', label="Smoothened spectrum")
axs.flatten()[0].set_title(f" n = {args.n}; $\ell_1-\ell_2$ = {args.l}$-${args.lp}; m>0")
axs.flatten()[0].set_xlabel(f"Frequency in $\mu$Hz")
axs.flatten()[0].legend()

axs.flatten()[1].fill_between(freq_win_n, csmsum-stdn, csmsum+stdn,
                              color='gray', alpha=0.4, label="1$\sigma$")
if not args.load_smooth:
    axs.flatten()[1].plot(freq_win_n, csmsum, 'k', label="Raw spectrum")
axs.flatten()[1].plot(freq_win_n, csmsum_smooth, 'k', label="Smoothened spectrum")
axs.flatten()[1].set_title(f" n = {args.n}; $\ell_1-\ell_2$ = {args.l}$-${args.lp}; m<0")
axs.flatten()[1].set_xlabel(f"Frequency in $\mu$Hz")
axs.flatten()[1].legend()

fig.tight_layout()
fig.savefig(f"{plotdir}/cspm_{args.n}_{args.l}_{args.lp}.pdf")
