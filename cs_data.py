# {{{ Library imports
from numpy.polynomial.legendre import legval
import matplotlib.pyplot as plt
import global_vars as GV
from math import sqrt
import numpy as np
import argparse
import time
import os
import sys
# }}} imports


# {{{ custom functions
sys.path.append('/home/g.samarth/Woodard2013/')
from WoodardPy.helioPy import datafuncs as cdata
# }}} custom functions

gvar = GV.globalVars()

# {{{ ArgumentParser
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
parser.add_argument("--find_baselines",
                    help="use extended window for baseline estimation",
                    action="store_true")
parser.add_argument("--plot",
                    help="plot results after computation",
                    action="store_true")
args = parser.parse_args()
# }}} parser

# number of extra ell frequency we shall consider to fit baselines
extd_win_ell = 6
extd_pixels = 0 #100
if args.n==0:
    extd_pixels = 100
noisewin = 50  #extd_pixels

# order for baseline
order = 2

# {{{ def find_freq_mask4bsl(data, lmin, lmax, n):
def find_freq_mask4bsl(data, freq, lmin, lmax, n): 
    # creating the lmax and lmins for the edge of window
    # for baseline estimation
    lmax = lmax + 6 
    lmin = lmin - 6  

    mask_lmax = (data[:, 0] == lmax) * (data[:, 1] == n)
    mask_lmin = (data[:, 0] == lmin) * (data[:, 1] == n)

    # edges of main signal
    freq_max = data[mask_lmax, 2] + 10 * data[mask_lmax, 4]
    freq_min = data[mask_lmin, 2] - 10 * data[mask_lmin, 4]

    # computing mask for frequency axis for bsl_estimation
    mask_freq = (freq < freq_min) + (freq > freq_max)

    return mask_freq
# }}} find_freq_mask4bsl(data, lmin, lmax, n)


# {{{ def find_maskpeaks4bsl(data, freq, lmin, lmax, n):
def find_maskpeaks4bsl(data, freq, lmin, lmax, n):
    mask_freq = np.ones(len(freq), dtype=bool)
    nlw = 5
    if n==0:
        for ell in range(lmin-6, lmin+10):
            f1, fwhm1, a1 = cdata.findfreq(data, ell, n, 0)
            mask = (freq < f1 + nlw*fwhm1) * (freq > f1 - nlw*fwhm1)
            mask_freq[mask] = False
        return mask_freq
    else:
        for ell in range(lmin-6, lmin+10):
            for enn in range(n-1, n+2):
                f1, fwhm1, a1 = cdata.findfreq(data, ell, enn, 0)
                if f1 == None:
                    continue
                else:
                    mask = (freq < f1 + nlw*fwhm1) * (freq > f1 - nlw*fwhm1)
                    mask_freq[mask] = False
        return mask_freq
# }}} find_maskpeaks4bsl(data, freq, lmin, lmax, n)


# {{{ def find_winhalflen(freq, freq_min, freq_max):
def find_winhalflen(freq, data, lmin, lmax, n, fit_bsl=False):
    # computing freq_min, freq_max of window
    try: # checking if L for \delta_ell = 4 is available
        lmax1 = lmax + 6
        assert ((data[:, 0] == lmax1) * (data[:, 1] == n)).any(), f"not found"
    except AssertionError:
        lmax1 = lmax + 4

    lmax = lmax1

    assert lmin > 6, "l should be greater than 6"
    lmin = lmin - 6

    mask_lmax = (data[:, 0] == lmax) * (data[:, 1] == n)
    mask_lmin = (data[:, 0] == lmin) * (data[:, 1] == n)
    assert mask_lmin.any(), f"Mode not found: lmin = {lmin}, n = {n}"

    freq_max = data[mask_lmax, 2] + 5*data[mask_lmax, 4]
    freq_min = data[mask_lmin, 2] - 5*data[mask_lmin, 4]

    # computing indices of freq_min and freq_max
    idx_min = np.argmin(np.abs(freq - freq_min))
    idx_max = np.argmin(np.abs(freq - freq_max))

    if fit_bsl:
        idx_min -= extd_pixels
        idx_max += extd_pixels
    winhalflen = (idx_max - idx_min)//2
    return winhalflen
# }}} find_winhalflen(freq, freq_min, freq_max)


# {{{ def plot_cs_contour(freq, cs, cenfreq, l1, pm):
def plot_cs_contour(freq, cs, cenfreq, l1, pm):
    ar = (freq[-1]-freq[0])/l1
    l1 = l1 if pm > 0 else -l1

    fig = plt.figure()
    im = plt.imshow(cs.real, cmap="gray",
                    vmax=(cs.real).max()/3, aspect=ar,
                    extent=[freq[0]-cenfreq, freq[-1]-cenfreq, l1, 0])
    plt.xlabel("$\omega - \omega_{nl}$ in $\mu$Hz")
    plt.ylabel("m")
    cbar_min = cs.min()
    cbar_max = cs.max()/3
    cbar_step = (cbar_max - cbar_min)/8
    cbar = plt.colorbar(im)
    cbar.ax.set_yticklabels(['{:.1e}'.format(x)
                             for x in np.arange(cbar_min,
                                                cbar_max+cbar_step,
                                                cbar_step)],
                            fontsize=12, weight='bold')
    return fig
# }}} plot_cs_contour(freq, cs, cenfreq, l1, pm)


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


# {{{ def derotate(phi, l, n, freq, winhalflen, sgn):
def derotate(phi, l, n, freq, winhalflen, sgn):
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
    data = gvar.data
    phinew = np.zeros((l+1, 2*winhalflen+1), dtype=complex)
    freq_win = np.zeros(phinew.shape)

    cenfreq, __, __ = cdata.findfreq(data, l, n, 0)
    m_arr = np.arange(0, sgn*(l+1), sgn)
    for m in m_arr:
        omeganl0 = cenfreq + finda1(data, l, n, m) * 1e-3
        omeganl0_ind = np.argmin(np.abs(freq-omeganl0))
        _indmin = omeganl0_ind - winhalflen
        _indmax = omeganl0_ind + winhalflen + 1
        try:
            freq_win[abs(m)] = freq[_indmin:_indmax]
        except ValueError:
            print(f"l = {l}, m = {m}, omeganl0_ind = {omeganl0_ind}; _indmin = {_indmin}; _indmax = {_indmax}")
        _real = phi[abs(m), _indmin-1:_indmax-1].real
        _imag = phi[abs(m), _indmin-1:_indmax-1].imag
        phinew[abs(m), :] = _real + 1j*_imag
    return phinew, freq_win
# }}} derotate(phi, l, n, freq, winhalflen, sgn)


# {{{ def fit_baseline(x, d, n):
def fit_baseline(x, d, n):
    # fitting an nth order quadratic
    G = np.zeros((len(d), n))
    for i in range(n):
        G[:,i] = x**i
    
    GTG = G.T @ G
    fit_coeffs = np.linalg.inv(GTG) @ (G.T @ d)

    return fit_coeffs
# }}} fit_baseline(x, d, n)


# {{{ def plot_scatter(fig, cs, pm):
def plot_scatter(fig, cs, pm):
    cs1, freqwin = derotate(cs, l1, n1, freq_all, winhalflen, pm)
    cs1sum = cs1.sum(axis=0)
    fig.plot(freqwin[0, :], cs1sum.real, '.b', markersize=0.8,
             linewidth=0.8, alpha=0.8)
    return fig
# }}} plot_scatter(fig, cs, pm)


def plot_cs_sigma(axs, freqpm, cspm, varpn, bslpm, l1l2n1):
    axp, axn = axs
    freqp, freqm = freqpm
    varp, varn = varpn
    bslp, bslm = bslpm
    csp, csm = cspm
    sig1p = np.sqrt(varp)
    sig1n = np.sqrt(varn)
    l1, l2, n1 = l1l2n1
    axp.fill_between(freqp, csp - sig1p, csp + sig1p, color='green', alpha=0.8)
    axp.plot(freqp, csp, 'r', linewidth=0.9)
    if extd_pixels > 0:
        axp.plot(freqp, bslp[extd_pixels:-extd_pixels], '--b', linewidth=0.9)
    else:
        axp.plot(freqp, bslp, '--b', linewidth=0.9)
    axp.set_title(f"Cross spectrum: {l1}, {l2} (n={n1}, t={args.t}), m+")
    axp.set_xlabel("Frequency in microHz")

    axn.fill_between(freqm, csm - sig1n, csm + sig1n, color='green', alpha=0.8)
    axn.plot(freqm, csm, 'r', linewidth=0.9)
    if extd_pixels > 0:
        axn.plot(freqm, bslm[extd_pixels:-extd_pixels], '--b', linewidth=0.9)
    else:
        axn.plot(freqm, bslm, '--b', linewidth=0.9)
    axn.set_title(f"Cross spectrum: {l1}, {l2} (n={n1}, t={args.t}), m-")
    axn.set_xlabel("Frequency in microHz")
    return axp, axn


# {{{ def compute_d2(cs, pm):
def compute_d2(cs, pm):
    csp1, freqp_win = derotate(cs, l1, n1, freq_all, winhalflen, pm)
    csp = csp1.sum(axis=0)
    return csp**2
# }}} compute_d2(cs, pm)

if __name__ == "__main__":
    # directories
    writedir = gvar.writedir

    
    # loading and computation times
    delt_load = 0.0
    delt_compute = 0.0

    # calculation parameters
    daynum = 1      # length of time series
    dayavgnum = 5   # number of time series to be averaged
    tsLen = 138240  # array length of the time series

    # time domain and frequency domain
    t = np.linspace(0, 72*24*3600*daynum, tsLen*daynum)
    dt = t[1] - t[0]
    freq = np.fft.fftfreq(t.shape[0], dt)*1e6
    df = freq[1] - freq[0]  # frequencies in microHz

    # loading mode parameters data
    data = gvar.data

    l1, n1 = args.l, args.n
    l2, n2 = args.lp, args.np
    print(f"l1-l2 = {l1}-{l2}, n1-n2 = {n1}-{n2}")

    # swapping n, l values if l2>l1
    if l2 < l1:
        ltemp, ntemp = l2, n2
        l2, n2 = l1, n1
        l1, n1 = ltemp, ntemp

    delta_ell = abs(l2 - l1)

    # baseline estiamtion flag
    fit_bsl = args.find_baselines

    # creating separate folder for each radial order
    dirname = f"{writedir}/csdata_{n1:02d}"
    if os.path.isdir(dirname):
        pass
    else:
        os.mkdir(dirname)
    writedir = dirname


    # considering only frequencies in a window
    # where cross spectrum is significant
    # choosing the window to be wide enough to accommodate
    # the entire cross-spectrum irrespective of ell
    print(f"n1 = {n1}")
    winhalflen = find_winhalflen(freq, data, l1, l1+4, n1, False)
    print(f"winhalflen = {winhalflen}")
    cenfreq, cenfwhm, __ = cdata.findfreq(data, l1, n1, 0)
    cenfreq2, cenfwhm2, __ = cdata.findfreq(data, l2+6, n1, 0)
    cenfreq0, cenfwhm0, __ = cdata.findfreq(data, l1-6, n1, 0)
    pmfreq_p = cenfreq2 - cenfreq + (l2+6)*0.7
    pmfreq_n = cenfreq - cenfreq0 + (l2+6)*0.7
    indm = cdata.locatefreq(freq, cenfreq - pmfreq_n)
    indp = cdata.locatefreq(freq, cenfreq + pmfreq_p)

    # if fit_bsl:
    print(f"n1 = {n1}")
    whlwindow = find_winhalflen(freq, data, l1, l1+4, n1, True)
    indm -= extd_pixels + whlwindow
    indp += extd_pixels + whlwindow

    print(f"freqmin = {freq[indm]}; freqmax = {freq[indp]}")
    freq = freq[indm:indp].copy()
    freq_all = freq.copy()

    fig_var_r, (axp_r, axn_r) = plt.subplots(2, 1, sharex=True, figsize=(5, 10))
    fig_var_i, (axp_i, axn_i) = plt.subplots(2, 1, sharex=True, figsize=(5, 10))

    for days in range(dayavgnum):
        day = 6328 + 72*days
        t1 = time.time()

        afftplus1, afftminus1 = cdata.separatefreq(
            cdata.loadHMIdata_avg(l1, day=day))

        if delta_ell == 0:
            afftplus2, afftminus2 = afftplus1, afftminus1
        else:
            afftplus2, afftminus2 = cdata.separatefreq(
                    cdata.loadHMIdata_avg(l2, day=day))

        afft1 = afftplus1[:, indm:indp]
        afft2 = afftplus2[:, indm:indp]
        afft1m = afftminus1[:, indm:indp]
        afft2m = afftminus2[:, indm:indp]

        # shifting the \phi2 by t
        afft2 = np.roll(afft2[:(l1+1), :], args.t, axis=0)
        afft2m = np.roll(afft2m[:(l1+1), :], args.t, axis=0)
        t2 = time.time()
        delt_load += t2 - t1

        # Calculating the cross-spectrum
        if days == 0:
            t1 = time.time()
            csp = afft1.conjugate()*afft2[:(l1+1), :]
            csm = afft1m.conjugate()*afft2m[:(l1+1), :]
            axp_r = plot_scatter(axp_r, csp.real, 1)
            axn_r = plot_scatter(axn_r, csm.real, -1)
            axp_i = plot_scatter(axp_i, csp.imag, 1)
            axn_i = plot_scatter(axn_i, csm.imag, -1)
            csp2r = compute_d2(csp.real, 1)
            csp2i = compute_d2(csp.imag, 1)
            csn2r = compute_d2(csm.real, -1)
            csn2i = compute_d2(csm.imag, -1)
            t2 = time.time()
        else:
            t1 = time.time()
            csp_temp = afft1.conjugate()*afft2[:(l1+1), :]
            csm_temp = afft1m.conjugate()*afft2m[:(l1+1), :]
            axp_r = plot_scatter(axp_r, csp.real, 1)
            axn_r = plot_scatter(axn_r, csm.real, -1)
            axp_i = plot_scatter(axp_i, csp.imag, 1)
            axn_i = plot_scatter(axn_i, csm.imag, -1)
            csp2r += compute_d2(csp_temp.real, 1)
            csp2i += compute_d2(csp_temp.imag, 1)
            csn2r += compute_d2(csm_temp.real, -1)
            csn2i += compute_d2(csm_temp.imag, -1)
            csp += csp_temp
            csm += csm_temp
            t2 = time.time()
        delt_compute += t2 - t1
    print(csp.real.max())
    
    csp /= dayavgnum
    csm /= dayavgnum

    csp2r /= dayavgnum
    csp2i /= dayavgnum
    csn2r /= dayavgnum
    csn2i /= dayavgnum

    print(f"Time series load time = {delt_load:7.3f} seconds")
    print(f"Cross spectra computation time = {delt_compute:7.3f} seconds")

    # finding winhalflen and frequency window
#    fig = plot_cs_contour(freq, cs, cenfreq, l1, 1)
#    plt.show()
#    fig = plot_cs_contour(freq, csm, cenfreq, l1, -1)
#    plt.show()

    if fit_bsl:
        # if delta_ell == 0:
            # winhalflen = find_winhalflen(freq, data, l1, l1+4, n1, fit_bsl)
        # else:
        print(f"n1 = {n1}")
        winhalflen = find_winhalflen(freq, data, l1, l1+4, n1, fit_bsl)
        print(f"winhalflen = {winhalflen}")

        # derotating the cross spectra using the grid without interpolation
        print(f"n1 = {n1}")
        csp1, freqp_win = derotate(csp, l1, n1, freq_all, winhalflen, 1)
        csm1, freqm_win = derotate(csm, l1, n1, freq_all, winhalflen, -1)

        # computing mask to extract the noise 
        # mask_freqp = find_freq_mask4bsl(data, freqp_win[0], l1, l1+4, n1)
        # mask_freqm = find_freq_mask4bsl(data, freqm_win[0], l1, l1+4, n1)
        mask_freqp = find_maskpeaks4bsl(data, freqp_win[0], l1, l1+4, n1)
        mask_freqm = find_maskpeaks4bsl(data, freqm_win[0], l1, l1+4, n1)
        # mask_freqp = np.ones(len(freqp_win[0, :]), dtype=bool)
        # mask_freqm = np.ones(len(freqm_win[0, :]), dtype=bool)
        # mask_freqp[noisewin:-noisewin] = False
        # mask_freqm[noisewin:-noisewin] = False

        # finding the corresponding noise arrays
        freqp_noise = freqp_win[0, mask_freqp]
        freqm_noise = freqm_win[0, mask_freqm]
        csp_noise = np.sum(csp1, axis=0)[mask_freqm]
        csm_noise = np.sum(csm1, axis=0)[mask_freqp]

        # creating or updating file storing: [l1,l2,c0,c1,c2,....]
        fname_p = f'{writedir}/bsl_p_{n1:02d}_{l1:03d}_{l2:03d}.npy'
        fname_n = f'{writedir}/bsl_n_{n1:02d}_{l1:03d}_{l2:03d}.npy'

        bsl_p_arr = np.array([])
        bsl_p_arr = np.array([])

        # plotting average over positive and negative m

        freqm_m0 = freqm_win[0]
        freqp_m0 = freqp_win[0]

        # calculating the coefficients for the polynomial
        fit_coeffs_m1 = fit_baseline(freqm_noise, csm_noise.real, order)
        fit_coeffs_m2 = fit_baseline(freqm_noise, csm_noise.imag, order)
        fit_coeffs_m = fit_coeffs_m1 + 1j*fit_coeffs_m2
        print(f"bsl_coeffs_m = {fit_coeffs_m}")
        # calculating the baseline
        bsl_m1 = np.polynomial.polynomial.polyval(freqm_m0, fit_coeffs_m1)
        bsl_m2 = np.polynomial.polynomial.polyval(freqm_m0, fit_coeffs_m2)
        bsl_m = bsl_m1 + 1j*bsl_m2

        # calculating the coefficients for the polynomial
        fit_coeffs_p1 = fit_baseline(freqp_noise, csp_noise.real, order)
        fit_coeffs_p2 = fit_baseline(freqp_noise, csp_noise.imag, order)
        fit_coeffs_p = fit_coeffs_p1 + 1j*fit_coeffs_p2
        print(f"bsl_coeffs_p = {fit_coeffs_p}")
        # calculating the baseline
        bsl_p1 = np.polynomial.polynomial.polyval(freqp_m0, fit_coeffs_p1)
        bsl_p2 = np.polynomial.polynomial.polyval(freqp_m0, fit_coeffs_p2)
        bsl_p = bsl_p1 + 1j*bsl_p2

        l1_l2 = np.array([l1,l2])

        bslp_spec = np.zeros((1, 2+order), dtype=np.complex128)
        bsln_spec = np.zeros((1, 2+order), dtype=np.complex128)

        bslp_spec[0,:2] = l1_l2
        bsln_spec[0,:2] = l1_l2

        bslp_spec[0,2:] = fit_coeffs_p 
        bsln_spec[0,2:] = fit_coeffs_m
        print(f"bsl_p = {bslp_spec}")
        print(f"bsl_n = {bsln_spec}")
            
        np.save(fname_p, bslp_spec)
        np.save(fname_n, bsln_spec)

    if args.plot:
        fig2r, axs2r = plt.subplots(2, figsize=(6, 12))
        axs2r[0].imshow(csm1.real)
        axs2r[1].imshow(csp1.real)

        fig2i, axs2i = plt.subplots(2, figsize=(6, 12))
        axs2i[0].imshow(csm1.imag)
        axs2i[1].imshow(csp1.imag)

        figr = plt.figure(figsize=(10, 5))
        plt.subplot(121)
        plt.plot(freqm_m0-cenfreq, np.sum(csm1[args.t:-args.t, :].real, axis=0),
                 color='k', label='-m')
        plt.plot(freqm_noise-cenfreq, csm_noise.real, '.g')
        plt.plot(freqm_m0-cenfreq, bsl_m.real, 'r')


        plt.subplot(122)
        plt.plot(freqp_m0-cenfreq, np.sum(csp1[args.t:-args.t, :].real, axis=0),
                 color='k', label='+m')
        plt.plot(freqp_noise-cenfreq, csp_noise.real, '.g')
        plt.plot(freqp_m0-cenfreq, bsl_p.real, 'r')
        plt.legend()
        plt.tight_layout()

        figi = plt.figure(figsize=(10, 5))
        plt.subplot(121)
        plt.plot(freqm_m0-cenfreq, np.sum(csm1[args.t:-args.t, :].imag, axis=0),
                 color='k', label='-m')
        plt.plot(freqm_noise-cenfreq, csm_noise.imag, '.g')
        plt.plot(freqm_m0-cenfreq, bsl_m.imag, 'r')


        plt.subplot(122)
        plt.plot(freqp_m0-cenfreq, np.sum(csp1[args.t:-args.t, :].imag, axis=0),
                 color='k', label='+m')
        plt.plot(freqp_noise-cenfreq, csp_noise.imag, '.g')
        plt.plot(freqp_m0-cenfreq, bsl_p.imag, 'r')
        plt.legend()
        plt.tight_layout()

        dirname = f"{gvar.outdir}/plots/{n1:02d}"
        if os.path.isdir(dirname):
            figr.savefig(f"{dirname}/{l1}_{l2}_{args.t}_real.png")
            figi.savefig(f"{dirname}/{l1}_{l2}_{args.t}_imag.png")
            fig2r.savefig(f"{dirname}/fullspec_{l1}_{l2}_{args.t}_real.png")
            fig2i.savefig(f"{dirname}/fullspec_{l1}_{l2}_{args.t}_imag.png")
        else:
            os.mkdir(dirname)
            figr.savefig(f"{dirname}/{l1}_{l2}_{args.t}_real.png")
            figi.savefig(f"{dirname}/{l1}_{l2}_{args.t}_imag.png")
            fig2r.savefig(f"{dirname}/fullspec_{l1}_{l2}_{args.t}_real.png")
            fig2i.savefig(f"{dirname}/fullspec_{l1}_{l2}_{args.t}_imag.png")
        plt.show(figi)

    # computing winhalflen for fit_bsl=False (without extra pixels)
    winhalflen = find_winhalflen(freq, data, l1, l1+4, n1, False)
    # derotating the cross spectra using the grid without interpolation
    csp, freqp_win = derotate(csp, l1, n1, freq_all, winhalflen, 1)
    csm, freqm_win = derotate(csm, l1, n1, freq_all, winhalflen, -1)

    unbias_corr = dayavgnum/(dayavgnum-1)
    if args.t == 0:
        csp_summ = csp.sum(axis=0)
        csm_summ = csm.sum(axis=0)
    else:
        csp_summ = csp[args.t:-args.t, :].sum(axis=0)
        csm_summ = csm[args.t:-args.t, :].sum(axis=0)
    variance_p = ((csp2r - (csp_summ.real)**2) +
                    1j*(csp2i - (csp_summ.imag)**2))
    variance_n = ((csn2r - (csm_summ.real)**2) +
                    1j*(csn2i - (csm_summ.imag)**2))
    variance_p *= unbias_corr
    variance_n *= unbias_corr

    freqpm = (freqp_win[0, :], freqm_win[0, :])
    l1l2n1 = (l1, l2, n1)

    axesr = (axp_r, axn_r)
    cspm = (csp_summ.real, csm_summ.real)
    varpn = (variance_p.real, variance_n.real)
    bslpm = (bsl_p.real, bsl_m.real)
    axp_r, axn_r = plot_cs_sigma(axesr, freqpm, cspm, varpn, bslpm, l1l2n1)

    axesr = (axp_i, axn_i)
    cspm = (csp_summ.imag, csm_summ.imag)
    varpn = (variance_p.imag, variance_n.imag)
    bslpm = (bsl_p.imag, bsl_m.imag)
    axp_i, axn_i = plot_cs_sigma(axesr, freqpm, cspm, varpn, bslpm, l1l2n1)

    dirname = f"{gvar.outdir}/plots/{n1:02d}"
    if os.path.isdir(dirname):
        fig_var_r.savefig(f"{dirname}/csdata_{l1}_{l2}_{args.t}_real.png")
        fig_var_i.savefig(f"{dirname}/csdata_{l1}_{l2}_{args.t}_imag.png")
    else:
        os.mkdir(dirname)
        fig_var_r.savefig(f"{dirname}/csdata_{l1}_{l2}_{args.t}_real.png")
        fig_var_i.savefig(f"{dirname}/csdata_{l1}_{l2}_{args.t}_imag.png")

    # plt.show(fig_var)

    # if args.plot:
    #     plt.figure(figsize=(10, 5))
    #     plt.subplot(121)
    #     plt.plot(freqm_win[0]-cenfreq, np.sum(csm.real, axis=0),
    #              color='k', label='-m')

    #     plt.subplot(122)
    #     plt.plot(freqp_win[0]-cenfreq, np.sum(csp.real, axis=0),
    #              color='k', label='+m')
    #     plt.legend()
    #     plt.tight_layout()
    #     plt.show()

    # storing the spectra
    fp_name = f"{writedir}/csp_data_{n1:02d}_{l1:03d}_{l2:03d}_{args.t:03d}.npy"
    fm_name = f"{writedir}/csm_data_{n1:02d}_{l1:03d}_{l2:03d}_{args.t:03d}.npy"
    np.save(fp_name, csp)
    np.save(fm_name, csm)

    # storing the variance
    fp_name = f"{writedir}/variance_p_{n1:02d}_{l1:03d}_{l2:03d}_{args.t:03d}.npy"
    fm_name = f"{writedir}/variance_n_{n1:02d}_{l1:03d}_{l2:03d}_{args.t:03d}.npy"
    np.save(fp_name, variance_p)
    np.save(fm_name, variance_n)
