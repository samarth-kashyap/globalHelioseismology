from numpy.polynomial.legendre import legval
from astropy.io import fits
import matplotlib.pyplot as plt
from math import sqrt, pi
import healpy as hp
import numpy as np
import time
import os

current_dir = os.path.dirname(os.path.realpath(__file__))
package_dir = os.path.dirname(os.path.dirname(current_dir))
with open(f"{package_dir}/.config", "r") as f:
    data_dir = f.readlines()[0].split("\n")[0]

__all__ = ["observedData",
           "frequencyBins",
           "crossSpectra"]


class observedData():
    __all__ = ["find_freq",
               "load_time_series"]

    __methods__ = ["find_freq",
                   "load_time_series"]

    __attributes__ = ["data",
                      "ts_datadir",
                      "WINLEN",
                      "POLYORD"]

    #---- Savitzky-Golay filter for smoothening data ----
    from scipy.signal import savgol_filter as savgol
    WINLEN = 25
    POLYORD = 5
    # --------------------------------------------------

    def __init__(self, instrument="HMI"):
        self.instrument = instrument

        if instrument == "HMI":
            self.data = np.loadtxt(f"{package_dir}/mode-params/hmi.6328.36")
            self.ts_datadir = data_dir

    # {{{ def find_freq(self, l, n, m):
    def find_freq(self, l, n, m):
        '''Find the eigenfrequency for a given (l, n, m)
        using the splitting coefficients

        Inputs: 
        -------
        (l, n, m)
        l - int
            spherical harmonic degree
        n - int
            radial order
        m - int
            azimuthal order

        Returns: 
        --------
        (nu, fwhm, amp)
        nu  - np.float64
            eigenfrequency nu_{nlm} in microHz
        fwhm - np.float64
            FWHM of the mode fwhm_{nl} in microHz
        amp  - np.float64
            Mode amplitude (A_{nl})
        '''
        try:
            mode_idx = np.where((self.data[:, 0] == l) *
                                (self.data[:, 1] == n))[0][0]
        except IndexError:
            print(f"Run-time error: Mode (l, n) = ({l:3d}, {n:2d}) not found.")
            return None, None, None

        (nu, amp, fwhm) = self.data[mode_idx, 2:5]
        if m==0:
            return nu, fwhm, amp
        else:
            # splitting coefficient a0 is not stored in the data file
            # splitting coefficients are in nHz where are nu is in microHz
            L = sqrt(l*(l+1))
            splits = np.append([0.0], self.data[modeindex, 12:48])
            splits[1] -= 31.7
            totsplit = legval(1.0*m/L, splits)*L*0.001
            return nu + totsplit, fwhm, amp
    # }}} find_freq(self, l, n, m)

    # {{{ def load_time_series(self, l, day=6328, num=1, smooth=False):
    def load_time_series(self, l, day=6328, smooth=False):
        """Loads time series data and averages over a given number.
        Returns the IFFT of the averaged time series.

        Inputs:
        ------
        l - int
            spherical harmonic degree
        day - int (default=6328)
            starting day of time series
            day number corresponds to day from MDI epoch

        Returns:
        --------
        (phi_omega_p, phi_omega_n)

        phi_omega_p - np.ndarray, ndim = 2)[m, omega]
            frequency series for m >= 0
        phi_omega_n - np.ndarray, ndim = 2)[m, omega]
            frequency series for m <= 0
        """
        ts_filename = f"{self.ts_datadir}/{self.instrument}_{l:03d}_{day:04d}.fits"
        print(f"Reading {ts_filename}")
        with fits.open(ts_filename) as f:
            phi_time = f[1].data

        if smooth: phi_time = savgol(phi_time, WINLEN, POLYORD)

        phi_omega = np.fft.ifft(phi_time[:, 0::2] - 1j*phi_time[:, 1::2], axis=1)

        freq_shape = phi_omega.shape[1]
        phi_omega_p = phi_omega[:, :freq_shape]
        phi_omega_n = phi_omega[:, freq_shape:]
        phi_omega_n = phi_omega[:, ::-1]
        return phi_omega_p, phi_omega_n
    # }}} load_time_series(self, l, day=6328, num=1, smooth=False)


class frequencyBins():
    __attributes__ = ["n1", "l1", "n2", "l2",
                      "instrument",
                      "freq_bins_global",
                      "mode_data",
                      "od"]

    __methods__ = ["get_freq_bins",
                   "window_freq",
                   "find_dnu_nlm"]

    def __init__(self, n1, l1, n2, l2,
                 instrument="HMI"):
        self.n1, self.n2 = n1, n2
        self.l1, self.l2 = l1, l2
        self.instrument = instrument
        self.freq_bins_global = None
        self.get_freq_bins(1)
        if instrument == "HMI":
            self.mode_data = np.loadtxt(f"{package_dir}/mode-params/hmi.6328.36")
        self.od = observedData(self.instrument)

    # {{{ def get_freq_bins(self, num_ts_blocks=1)
    def get_freq_bins(self, num_ts_blocks=1):
        """Defines the frequency array for the time series. 
        For the HMI instrument - cadence = 45 seconds.
        For the MDI instrument - ...

        Inputs:
        -------
        num_ts_blocks - int
            number of time-series blocks that are concatenated to form the 
            full time-series

        Returns:
        -------
        freq - np.ndarray(ndim=1, dtype=np.float64)
            frequency bins corresponding to the time-series
        """
        ts_len = 138240  # array length of the time series
        t = np.linspace(0, 72*24*3600*num_ts_blocks, ts_len*num_ts_blocks)
        dt = t[1] - t[0]
        freq = np.fft.fftfreq(t.shape[0], dt)*1e6
        df = freq[1] - freq[0]  # frequencies in microHz
        self.freq_bins_global = freq
        return freq
    # }}} get_freq_bins

    # {{{ def window_freq(self)
    def window_freq(self):
        """Window the frequency bins such that only region around expected
        signal is retained.

        Inputs:
        -------
        None

        Returns:
        --------
        freq, (idx_n, idx_p)
        
        freq - np.ndarray(ndim=1, dtype=np.float64)
            frequency bins corresponding to the chosen window
        idx_n - int 
            index corresponding to beginning of frequency window
        idx_p - int 
            index corresponding to the end of frequency window
        """
        data = self.mode_data
        num_lw = 5
        mode1freq, __, __ = self.od.find_freq(self.l1, self.n1, 0)
        mode2freq, __, __ = self.od.find_freq(self.l2, self.n2, 0)

        # obtaining the central, left and right frequencies
        cen_freq, cen_fwhm, __ = self.od.find_freq(self.l1, self.n1, 0)

        if mode2freq > mode1freq:
            right_freq, right_fwhm, __ = self.od.find_freq(self.l2+6, self.n2, 0)
            left_freq, left_fwhm, __ = self.od.find_freq(self.l1-6, self.n1, 0)
        else:
            right_freq, right_fwhm, __ = self.od.find_freq(self.l1+6, self.n1, 0)
            left_freq, left_fwhm, __ = self.od.find_freq(self.l2-6, self.n2, 0)

        pmfreq_p = right_freq - cen_freq + (self.l2+6)*0.7 + right_fwhm*num_lw
        pmfreq_n = cen_freq - left_freq + (self.l2+6)*0.7 + left_fwhm*num_lw

        idx_n = np.argmin(abs(self.freq_bins_global - (cen_freq - pmfreq_n)))
        idx_p = np.argmin(abs(self.freq_bins_global - (cen_freq + pmfreq_p)))
        idx_0 = np.argmin(abs(self.freq_bins_global - cen_freq))

        idx_diff_p = idx_p - idx_0
        idx_diff_n = idx_0 - idx_n

        print(f"freqmin = {self.freq_bins_global[idx_n]}; " +
              f"freqmax = {self.freq_bins_global[idx_p]}")

        # frequency bins for the un-derotated signal
        freq = self.freq_bins_global[idx_n:idx_p]*1.0
        self.idx_n, self.idx_p = int(idx_n), int(idx_p)
        self.idx_diff_n, self.idx_diff_p = int(idx_diff_n), int(idx_diff_p)

        # frequency bins for the derotated signal
        pmfreq_p = right_freq - cen_freq + right_fwhm*num_lw
        pmfreq_n = cen_freq - left_freq + left_fwhm*num_lw
        idx_n2 = np.argmin(abs(self.freq_bins_global - (cen_freq - pmfreq_n)))
        idx_p2 = np.argmin(abs(self.freq_bins_global - (cen_freq + pmfreq_p)))
        idx_0 = np.argmin(abs(self.freq_bins_global - cen_freq))
        idx_derot_diff_p = idx_p2 - idx_0
        idx_derot_diff_n = idx_0 - idx_n2
        self.idx_derot_np = (idx_derot_diff_n, idx_derot_diff_p)
        return freq, (idx_n, idx_p), (idx_diff_n, idx_diff_p), \
            (idx_derot_diff_n, idx_derot_diff_p)
    # }}} window_freq

    # {{{ def find_dnu_nlm(data, n, l, m)
    def find_dnu_nlm(self, n, l, m):
        """Find a coefficients for given l, n, m (in microHz)

        Inputs:
        -------
        (n, l, m)
        n - int
            Radial order 
        l - int
            Sperical Harmonic degree
        m - int
            Azimuthal order 

        Returns:
        --------
        diff_nu - float
            nu_0 - nu_{nlm} in microHz
        """
        L = sqrt(l*(l+1))
        data = self.mode_data
        try:
            mode_idx = np.where((data[:, 0] == l) *
                                (data[:, 1] == n))[0][0]
        except IndexError:
            print(f"MODE NOT FOUND : l = {l}, n = {n}")
            return None, None, None
        splits = np.append([0.0], data[mode_idx, 12:48])
        totsplit = legval(1.0*m/L, splits)*L
        diff_nu = totsplit - 31.7*m
        return diff_nu * 1e-3
    # }}} find_dnu_nlm(data, n, l, m)


class crossSpectra():
    """Class to deal with cross-spectral computation from 
    observed data. 
    Instruments supported: HMI 
    Instrument support for future: MDI, GONG
    """
    from astropy.io import fits

    __attributes__ = ["n1", "l1",
                      "n2", "l2", "t",
                      "instrument",
                      "dayavgnum", "od"]

    __methods__ = ["store_cross_spectra",
                   "save_data",
                   "compute_freq_series",
                   "compute_d2",
                   "derotate",
                   "finda1",
                   "find_maskpeaks4bsl",
                   "fit_polynomial",
                   "find_baseline_coeffs",
                   "get_baseline_from_coeffs",
                   "plot_scatter"]

    def __init__(self, n1, l1, n2, l2, t=0, instrument="HMI", smooth=False,
                 daynum=1, dayavgnum=5, fit_bsl=False, store_spectra=True,
                 plot_data=True, plot_snr=True):
        # swapping values of ell if l2 < l1
        if l2 < l1:
            ltemp, ntemp = l2, n2
            l2, n2 = l1, n1
            l1, n1 = ltemp, ntemp

        self.plot_data = plot_data
        self.plot_snr = plot_snr

        self.delta_ell = abs(l2 - l1)
        # assert n1 == n2, f"n1 != n2, forcing n2 = {n1}"
        self.n1, self.n2 = int(n1), int(n2)
        self.l1, self.l2 = int(l1), int(l2)
        self.t, self.dayavgnum = int(t), int(dayavgnum)
        self.mode_data = np.loadtxt("/home/g.samarth/globalHelioseismology/" +
                                    f"mode-params/hmi.6328.36")
        self.dirname = f"/scratch/g.samarth/globalHelioseismology"
        self.fname_suffix = f"{n1:02d}.{l1:03d}-{n2:02d}.{l2:03d}-{self.t:03d}"

        if plot_data:
            if not os.path.isdir(f"{self.dirname}/plots/csplot_{self.n1:02d}"):
                os.mkdir(f"{self.dirname}/plots/csplot_{self.n1:02d}")
            fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
            self.fig = fig
            self.axs = axs.flatten()
            self.fig.suptitle(f"Spectra for $(n_1, \ell_1)$ = ({self.n1}, {self.l1})" +
                              f" and $(n_2, \ell_2)$ = ({self.n2}, {self.l2})" +
                              f" - t = {self.t}")
            self.axs[0].set_title("Real spectra, $m >= 0$")
            self.axs[1].set_title("Real spectra, $m <= 0$")
            self.axs[2].set_title("Imag spectra, $m >= 0$")
            self.axs[3].set_title("Imag spectra, $m <= 0$")
            for i in range(len(self.axs)):
                self.axs[i].set_xlabel("Frequency in $\mu$Hz")
                self.axs[i].ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
            self.fig.tight_layout()

        if plot_snr:
            fig_snr, axs_snr = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
            self.fig_snr = fig_snr
            self.axs_snr = axs_snr.flatten()
            self.fig_snr.suptitle(f"S/N ratio $(n_1, \ell_1)$ = ({self.n1}, {self.l1})" +
                                  f" and $(n_2, \ell_2)$ = ({self.n2}, {self.l2})" +
                                  f" - t = {self.t}")
            self.axs_snr[0].set_title("Real spectra, $m >= 0$")
            self.axs_snr[1].set_title("Real spectra, $m <= 0$")
            self.axs_snr[2].set_title("Imag spectra, $m >= 0$")
            self.axs_snr[3].set_title("Imag spectra, $m <= 0$")
            for i in range(len(self.axs)):
                self.axs_snr[i].set_xlabel("Frequency in $\mu$Hz")
                self.axs_snr[i].ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
            self.fig_snr.tight_layout()

        # observed data relevant to the class instance
        self.od = observedData(instrument)

        # loading frequency bins relevant to the class instance
        self.fb = frequencyBins(n1, l1, n2, l2, instrument=instrument)
        freq, idx_np, idx_diff_np, idx_derot_diff_np = self.fb.window_freq()
        self.freq, (self.idx_n, self.idx_p) = freq, idx_np
        self.idx_diff_n, self.idx_diff_p = idx_diff_np
        self.idx_derot_diff_n, self.idx_derot_diff_p = idx_derot_diff_np
        if store_spectra:
            self.store_cross_spectra()
            self.fig.savefig(f"{self.dirname}/plots/csplot_{self.n1:02d}/csplot_" +
                             f"{self.fname_suffix}.png")
            self.fig_snr.savefig(f"{self.dirname}/plots/csplot_{self.n1:02d}/snrplot_" +
                                 f"{self.fname_suffix}.png")

    # {{{ def store_cross_spectra(self):
    def store_cross_spectra(self):
        csp, csn, csp2, csn2 = self.compute_cross_spectra()
        if self.t == 0:
            csp_summ = csp.sum(axis=0)
            csn_summ = csn.sum(axis=0)
        else:
            # csp_summ = csp[self.t:-self.t, :].sum(axis=0)
            # csn_summ = csn[self.t:-self.t, :].sum(axis=0)
            csp_summ = csp[:-self.t, :].sum(axis=0)
            csn_summ = csn[:-self.t, :].sum(axis=0)

        variance_p = ((csp2.real - (csp_summ.real)**2) +
                        1j*(csp2.imag - (csp_summ.imag)**2))
        variance_n = ((csn2.real - (csn_summ.real)**2) +
                        1j*(csn2.imag - (csn_summ.imag)**2))

        unbias_corr = self.dayavgnum/(self.dayavgnum-1)
        variance_p *= unbias_corr
        variance_n *= unbias_corr

        l1, l2 = self.l1, self.l2
        n1, n2 = self.n1, self.n2

        bslp_coeff, bsln_coeff = self.find_baseline_coeffs(csp, csn)
        bslp_spec = np.zeros((1, 4), dtype=np.complex128)
        bsln_spec = np.zeros((1, 4), dtype=np.complex128)
        bslp_spec[0, :2] = np.array([self.l1, self.l2])
        bsln_spec[0, :2] = np.array([self.l1, self.l2])
        bslp_spec[0, 2:] = bslp_coeff
        bsln_spec[0, 2:] = bsln_coeff
        _bslp, _bsln = self.get_baseline_from_coeffs(bslp_coeff, bsln_coeff)

        snr_p, snr_r = self.compute_snr((csp_summ - _bslp, csn_summ - _bsln),
                                        (variance_p, variance_n))

        if self.plot_data:
            self.axs[0].fill_between(self.freq_p[0],
                                     csp_summ.real - np.sqrt(variance_p.real),
                                     csp_summ.real + np.sqrt(variance_p.real),
                                     color='g', alpha=0.9)
            self.axs[0].plot(self.freq_p[0], csp_summ.real, 'r', linewidth=0.7)

            self.axs[1].fill_between(self.freq_n[0],
                                     csn_summ.real - np.sqrt(variance_n.real),
                                     csn_summ.real + np.sqrt(variance_n.real),
                                     color='g', alpha=0.9)
            self.axs[1].plot(self.freq_n[0], csn_summ.real, 'r', linewidth=0.7)

            self.axs[2].fill_between(self.freq_p[0],
                                     csp_summ.imag - np.sqrt(variance_p.imag),
                                     csp_summ.imag + np.sqrt(variance_p.imag),
                                     color='g', alpha=0.9)
            self.axs[2].plot(self.freq_p[0], csp_summ.imag, 'r', linewidth=0.7)

            self.axs[3].fill_between(self.freq_n[0],
                                     csn_summ.imag - np.sqrt(variance_n.imag),
                                     csn_summ.imag + np.sqrt(variance_n.imag),
                                     color='g', alpha=0.9)
            self.axs[3].plot(self.freq_n[0], csn_summ.imag, 'r', linewidth=0.7)

            self.axs[0].plot(self.freq_p[0], _bslp.real, '--y')
            self.axs[1].plot(self.freq_n[0], _bsln.real, '--y')
            self.axs[2].plot(self.freq_p[0], _bslp.imag, '--y')
            self.axs[3].plot(self.freq_n[0], _bsln.imag, '--y')

        if not os.path.isdir(f"{self.dirname}/csdata_{self.n1:02d}"):
            os.mkdir(f"{self.dirname}/csdata_{self.n1:02d}")

        self.save_data(f"{self.dirname}/csdata_{self.n1:02d}/" +
                       f"csp_data_{self.fname_suffix}.npy", csp_summ)
        self.save_data(f"{self.dirname}/csdata_{self.n1:02d}/" +
                       f"csm_data_{self.fname_suffix}.npy", csn_summ)
        self.save_data(f"{self.dirname}/csdata_{self.n1:02d}/" +
                       f"variance_p_{self.fname_suffix}.npy", variance_p)
        self.save_data(f"{self.dirname}/csdata_{self.n1:02d}/" +
                       f"variance_n_{self.fname_suffix}.npy", variance_n)
        self.save_data(f"{self.dirname}/csdata_{self.n1:02d}/" +
                       f"bsl_p_{self.fname_suffix}.npy", _bslp) #bslp_spec)
        self.save_data(f"{self.dirname}/csdata_{self.n1:02d}/" +
                       f"bsl_n_{self.fname_suffix}.npy", _bsln) #bsln_spec)
        self.save_data(f"{self.dirname}/csdata_{self.n1:02d}/" +
                       f"snrp_{self.fname_suffix}.npy", snr_p)
        self.save_data(f"{self.dirname}/csdata_{self.n1:02d}/" +
                       f"snrn_{self.fname_suffix}.npy", snr_r)
        return csp_summ, csn_summ, variance_p, variance_n, bslp_coeff, bsln_coeff
    # }}} store_cross_spectra(self)

    def compute_snr(self, cspn, varpn, snr_thresh=0.5):
        csp, csn = cspn
        varp, varn = varpn

        def mask_small(yval, threshold=5e-2):
            maxval = abs(yval).max()
            limval = threshold*maxval
            mask = abs(yval) > limval
            return mask

        snr_p1 = abs(csp.real)**2/varp.real
        snr_p2 = abs(csp.imag)**2/varp.imag
        snr_n1 = abs(csn.real)**2/varn.real
        snr_n2 = abs(csn.imag)**2/varn.imag

        mask_p1 = mask_small(csp.real,
                             threshold=np.mean(np.sqrt(varp.real))/abs(csp.real).max())
        mask_p2 = mask_small(csp.imag,
                             threshold=np.mean(np.sqrt(varp.imag))/abs(csp.imag).max())
        mask_n1 = mask_small(csn.real,
                             threshold=np.mean(np.sqrt(varn.real))/abs(csn.real).max())
        mask_n2 = mask_small(csn.imag,
                             threshold=np.mean(np.sqrt(varn.imag))/abs(csn.imag).max())

        # snr_p1[~mask_p1] = 0
        # snr_p2[~mask_p2] = 0
        # snr_n1[~mask_n1] = 0
        # snr_n2[~mask_n2] = 0

        # snr_p1[snr_p1 < snr_thresh] = np.nan
        # snr_n1[snr_n1 < snr_thresh] = np.nan
        # snr_p2[snr_p2 < snr_thresh] = np.nan
        # snr_n2[snr_n2 < snr_thresh] = np.nan

        mask_p1 = mask_p1 * snr_p1 > snr_thresh
        mask_p2 = mask_p2 * snr_p2 > snr_thresh
        mask_n1 = mask_n1 * snr_n1 > snr_thresh
        mask_n2 = mask_n2 * snr_n2 > snr_thresh

        snr_p = snr_p1 + 1j*snr_p2
        snr_n = snr_n1 + 1j*snr_n2
        snr_lim = 3

        if self.plot_snr:
            count = (snr_p.real > snr_lim).sum()
            self.axs_snr[0].plot(self.freq_p[0], snr_p.real, '.c', markersize=3)
            self.axs_snr[0].semilogy(self.freq_p[0][mask_p1],
                                     snr_p.real[mask_p1], '+k', markersize=5)
            self.axs_snr[0].set_xlim([self.freq_p[0][0], self.freq_p[0][-1]])
            self.axs_snr[0].set_title(f'Real spectra, m > 0:' +
                                      f'SNR>{snr_lim} count = {count}')

            count = (snr_n.real > snr_lim).sum()
            self.axs_snr[1].semilogy(self.freq_n[0], snr_n.real, '.c', markersize=3)
            self.axs_snr[1].semilogy(self.freq_n[0][mask_n1],
                                     snr_n.real[mask_n1], '+k', markersize=5)
            self.axs_snr[1].set_xlim([self.freq_n[0][0], self.freq_n[0][-1]])
            self.axs_snr[1].set_title(f'Real spectra, m < 0:' +
                                      f'SNR>{snr_lim} count = {count}')

            count = (snr_p.imag > snr_lim).sum()
            self.axs_snr[2].semilogy(self.freq_p[0], snr_p.imag, '.c', markersize=3)
            self.axs_snr[2].semilogy(self.freq_p[0][mask_p2],
                                     snr_p.imag[mask_p2], '+k', markersize=5)
            self.axs_snr[2].set_xlim([self.freq_p[0][0], self.freq_p[0][-1]])
            self.axs_snr[2].set_title(f'Imag spectra, m > 0:' +
                                      f'SNR>{snr_lim} count = {count}')

            count = (snr_n.imag > snr_lim).sum()
            self.axs_snr[3].semilogy(self.freq_n[0], snr_n.imag, '.c', markersize=3)
            self.axs_snr[3].semilogy(self.freq_n[0][mask_n2],
                                     snr_n.imag[mask_n2], '+k', markersize=5)
            self.axs_snr[3].set_xlim([self.freq_n[0][0], self.freq_n[0][-1]])
            self.axs_snr[3].set_title(f'Imag spectra, m < 0:' +
                                      f'SNR>{snr_lim} count = {count}')

        return snr_p, snr_n


    def save_data(self, fname, data, info=True):
        np.save(fname, data)
        print(f"Saving {fname}")

    # {{{ def compute_freq_series(self, ell=70, plot=False):
    def compute_freq_series(self, ell=70, plot=False, rotated=False,
                            euler_angle=np.array([0, -np.pi/2.0, 0.0])):
        """Computest the frequency series corresponding to the spherical
        harmonic time-series obtained from the observatory pipeline.
        p-angle correction can also be performed using the "rotated" flag

        Inputs:
        -------
        ell - int
            spherical harmonic degree
        plot - bool
            boolean switch for plotting
        rotated - bool
            boolean switch for rotation of spherical harmonics
            set rotated=True for performing rotation for a given 
            euler_angle. Useful for p-angle correction.
        euler_angle - np.ndarray(ndim=1, dtype=np.float64)
            the euler angle for rotation of spherical harmonics.

        Returns:
        --------
        (afft1p, afft1n), (freq_p, freq_n)
        afft1p - np.ndarray(ndim=2, dtype=np.complex128)
            derotated fourier transform of \phi_{lm}(t) for +ve m 
        afft1n - np.ndarray(ndim=2, dtype=np.complex128)
            derotated fourier transform of \phi_{lm}(t) for -ve m 
        freq_p - np.ndarray(ndim=1, dtype=np.float64)
            frequency series corresponding to the derotated spectra for +ve m
        freq_n - np.ndarray(ndim=2, dtype=np.complex128)
            frequency series corresponding to the derotated spectra for -ve m
        """
        afft1p, afft1n = 0, 0
        for day_idx in range(self.dayavgnum):
            day = 6328 + 72*day_idx

            _afft1p, _afft1n = self.od.load_time_series(ell, day=day)

            # windowing in frequency
            _afft1p = _afft1p[:, self.idx_n:self.idx_p]
            _afft1n = _afft1n[:, self.idx_n:self.idx_p]

            # adding the cross-spectrum (for expectation value computation)
            afft1p += _afft1p
            afft1n += _afft1n

        afft1p /= self.dayavgnum
        afft1n /= self.dayavgnum

        if rotated:
            _nsidearr = np.array([2, 4, 8, 16, 32, 64, 128])
            _lmaxarr = 3*_nsidearr - 1
            _elldiff = _lmaxarr - ell
            _nsidearr[_elldiff < 0] = 100000000
            _nside_idx = np.argmin(_nsidearr)
            NSIDE = _nsidearr[_nside_idx]
            LMAX = _lmaxarr[_nside_idx]
            ellArr, emmArr = hp.sphtfunc.Alm.getlm(LMAX)
            almp = np.zeros(len(ellArr), dtype=np.complex128)
            almn = np.zeros(len(ellArr), dtype=np.complex128)
            mask_ell = ellArr == ell
            num_bins = afft1p.shape[1]
            r = hp.rotator.Rotator(euler_angle, deg=False, eulertype='zyz')
            for idx in range(num_bins):
                almp[mask_ell] = afft1p[:, idx]
                almn[mask_ell] = afft1n[:, idx]
                almp = r.rotate_alm(almp)
                almn = r.rotate_alm(almn)
                afft1p[:, idx] = almp[mask_ell]
                afft1n[:, idx] = almn[mask_ell]

        afft1p, freq_p = self.derotate(afft1p, 1)
        afft1n, freq_n = self.derotate(afft1n, -1)
        return (afft1p, afft1n), (freq_p, freq_n)
    # }}} compute_freq_series(self, ell=70, plot=False):

    # {{{ def compute_cross_spectra(self, plot=False):
    def compute_cross_spectra(self, plot=False):
        csp, csn = 0.0, 0.0
        csp2r, csp2i = 0.0, 0.0
        csn2r, csn2i = 0.0, 0.0

        for day_idx in range(self.dayavgnum):
            day = 6328 + 72*day_idx

            afft1p, afft1n = self.od.load_time_series(self.l1, day=day)
            (afft2p, afft2n) = (self.od.load_time_series(self.l2, day=day)) if \
                self.delta_ell != 0 else (afft1p*1.0, afft1n*1.0)

            # windowing in frequency
            afft1p = afft1p[:, self.idx_n:self.idx_p]
            afft2p = afft2p[:, self.idx_n:self.idx_p]
            afft1n = afft1n[:, self.idx_n:self.idx_p]
            afft2n = afft2n[:, self.idx_n:self.idx_p]

            # shifting the \phi2 by t
            if self.t != 0:
                afft2p = np.roll(afft2p[:(self.l1+1), :], self.t, axis=0)
                afft2n = np.roll(afft2n[:(self.l1+1), :], self.t, axis=0)

            # computing the cross-spectrum
            _csp = afft1p.conjugate()*afft2p[:(self.l1+1), :]
            _csn = afft1n.conjugate()*afft2n[:(self.l1+1), :]
            self.plot_scatter(_csp, _csn)

            # adding the cross-spectrum (for expectation value computation)
            csp += _csp
            csn += _csn

            csp2r += self.compute_d2(_csp.real, 1)
            csp2i += self.compute_d2(_csp.imag, 1)
            csn2r += self.compute_d2(_csn.real, -1)
            csn2i += self.compute_d2(_csn.imag, -1)

        csp /= self.dayavgnum
        csn /= self.dayavgnum

        csp2r /= self.dayavgnum
        csp2i /= self.dayavgnum
        csn2r /= self.dayavgnum
        csn2i /= self.dayavgnum

        csp, freq_p = self.derotate(csp, 1)
        csn, freq_n = self.derotate(csn, -1)

        self.freq_p = freq_p
        self.freq_n = freq_n

        return csp, csn, csp2r+1j*csp2i, csn2r+1j*csn2i
    # }}} compute_cross_spectra(self, plot=False):

    # {{{ def compute_d2(cs, pm):
    def compute_d2(self, cs, pm):
        csp1, freqp_win = self.derotate(cs, pm)
        if self.t == 0:
            csp = csp1.sum(axis=0)
        else:
            csp = csp1[:-self.t, :].sum(axis=0)
        return csp**2
    # }}} compute_d2(cs, pm)

    # {{{ def derotate(phi, sgn):
    def derotate(self, phi, sgn):
        """Derotate the given cross-spectra
        Inputs:
        -------
        phi - np.ndarray(ndim=2)
            frequency series.
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
        # renaming parameters
        l, n = self.l1, self.n1
        freq = self.freq
        data = self.mode_data
        freq_len = self.idx_derot_diff_p + self.idx_derot_diff_n + 1

        phi_derotated = np.zeros((l+1, freq_len), dtype=np.complex128)
        freq_win = np.zeros((l+1, freq_len))

        cen_freq, __, __ = self.od.find_freq(l, n, 0)
        m_arr = np.arange(0, sgn*(l+1), sgn)
        for m in m_arr:
            _nu_nlm = cen_freq + self.finda1(n, l, m) * 1e-3
            _nu_nlm_idx = np.argmin(np.abs(freq - _nu_nlm))
            _idx_min = _nu_nlm_idx - self.idx_derot_diff_n
            _idx_max = _nu_nlm_idx + self.idx_derot_diff_p + 1
            try:
                freq_win[abs(m), :] = freq[_idx_min:_idx_max]
            except ValueError:
                print(f"l = {l}, m = {m}, _nu_nlm_idx = {_nu_nlm_idx}; " +
                      f"_idx_min = {_idx_min}; _idx_max = {_idx_max}")
            _real = phi[abs(m), _idx_min-1:_idx_max-1].real
            _imag = phi[abs(m), _idx_min-1:_idx_max-1].imag
            phi_derotated[abs(m), :] = _real + 1j*_imag
        return phi_derotated, freq_win
    # }}} derotate(phi, l, n, freq, winhalflen, sgn)

    # {{{ def finda1(data, n, l, m):
    def finda1(self, n, l, m):
        """Find a coefficients for given l, n, m
        """
        L = sqrt(l*(l+1))
        data = self.mode_data
        try:
            mode_idx = np.where((data[:, 0] == l) * (data[:, 1] == n))[0][0]
        except IndexError:
            print(f"MODE NOT FOUND : l = {l}, n = {n}")
            return None, None, None
        splits = np.append([0.0], data[mode_idx, 12:48])
        totsplit = legval(1.0*m/L, splits)*L
        return totsplit - 31.7*m
    # }}} finda1(data, n, l, m)

    # {{{ def find_maskpeaks4bsl(data, freq, lmin, lmax, n):
    def find_maskpeaks4bsl(self, freq, lmin, lmax, n1, n2):
        mask_freq = np.ones(len(freq), dtype=bool)
        halflen = len(freq)//2
        n_arr = np.array([n1, n2])
        nlw = 5
        if n1 == 0:
            for ell in range(lmin-6, lmax+6):
                f1, fwhm1, a1 = self.od.find_freq(ell, n1, 0)
                mask = (freq < f1 + nlw*fwhm1) * (freq > f1 - nlw*fwhm1)
                mask_freq[mask] = False
            # return mask_freq
        else:
            for ell in range(lmin-6, lmax+6):
                for n in n_arr:
                    for enn in range(n-1, n+2):
                        f1, fwhm1, a1 = self.od.find_freq(ell, enn, 0)
                        if f1 == None:
                            continue
                        else:
                            mask = (freq < f1 + nlw*fwhm1) * (freq > f1 - nlw*fwhm1)
                            mask_freq[mask] = False
            # return mask_freq
        left_sum = mask_freq[:halflen].sum()
        right_sum = mask_freq[halflen:].sum()
        if left_sum == 0:
            mask_freq[5] = True
        if right_sum == 0:
            mask_freq[-5] = True
        return mask_freq
    # }}} find_maskpeaks4bsl(data, freq, lmin, lmax, n)

    # {{{ def fit_baseline(x, data, order):
    def fit_polynomial(self, x, data, order):
        # fitting an nth order polynomial
        G = np.zeros((len(data), order))
        for i in range(order):
            G[:, i] = x**i

        GTG = G.T @ G
        fit_coeffs = np.linalg.inv(GTG) @ (G.T @ data)
        return fit_coeffs
    # }}} fit_baseline(x, d, n)

    # {{{ def find_baselines(self, csp, csn):
    def find_baseline_coeffs(self, csp, csn):
        mask_freq_p = self.find_maskpeaks4bsl(self.freq_p[0], self.l1, self.l2,
                                              self.n1, self.n2)
        mask_freq_n = self.find_maskpeaks4bsl(self.freq_n[0], self.l1, self.l2,
                                              self.n1, self.n2)
        csp_noise = np.sum(csp, axis=0)[mask_freq_p]
        csn_noise = np.sum(csn, axis=0)[mask_freq_n]
        freq_p_noise = self.freq_p[0][mask_freq_p]
        freq_n_noise = self.freq_n[0][mask_freq_n]

        fit_coeffs_p1 = self.fit_polynomial(freq_p_noise, csp_noise.real, 2)
        fit_coeffs_p2 = self.fit_polynomial(freq_p_noise, csp_noise.imag, 2)
        fit_coeffs_p = fit_coeffs_p1 + 1j*fit_coeffs_p2

        fit_coeffs_n1 = self.fit_polynomial(freq_n_noise, csn_noise.real, 2)
        fit_coeffs_n2 = self.fit_polynomial(freq_n_noise, csn_noise.imag, 2)
        fit_coeffs_n = fit_coeffs_n1 + 1j*fit_coeffs_n2
        return fit_coeffs_p, fit_coeffs_n
    # }}} find_baseline_coeffs(self, csp, csn)

    # {{{ def get_baseline_from_coeffs(self, bsl_coeffs_p, bsl_coeffs_n):
    def get_baseline_from_coeffs(self, bsl_coeffs_p, bsl_coeffs_n):
        bsl_pr = np.polynomial.polynomial.polyval(self.freq_p[0], bsl_coeffs_p.real)
        bsl_pi = np.polynomial.polynomial.polyval(self.freq_p[0], bsl_coeffs_p.imag)
        bsl_p = bsl_pr + 1j*bsl_pi

        bsl_nr = np.polynomial.polynomial.polyval(self.freq_n[0], bsl_coeffs_n.real)
        bsl_ni = np.polynomial.polynomial.polyval(self.freq_n[0], bsl_coeffs_n.imag)
        bsl_n = bsl_nr + 1j*bsl_ni
        return bsl_p, bsl_n
    # }}} get_baseline_from_coeffs(self, bsl_coeffs_p, bsl_coeffs_n)

    # {{{ def plot_scatter(self, csp, csn):
    def plot_scatter(self, csp, csn):
        cs1, freqwin1 = self.derotate(csp, 1)
        cs2, freqwin2 = self.derotate(csn, -1)

        cs1sum = cs1.sum(axis=0)
        cs2sum = cs2.sum(axis=0)

        self.axs[0].plot(freqwin1[0, :], cs1sum.real, '+b',
                         markersize=0.8, linewidth=0.8, alpha=0.6)
        self.axs[1].plot(freqwin2[0, :], cs2sum.real, '+b',
                         markersize=0.8, linewidth=0.8, alpha=0.6)

        self.axs[2].plot(freqwin1[0, :], cs1sum.imag, '+b',
                         markersize=0.8, linewidth=0.8, alpha=0.6)
        self.axs[3].plot(freqwin2[0, :], cs2sum.imag, '+b',
                         markersize=0.8, linewidth=0.8, alpha=0.6)
        return None
    # }}} plot_scatter(self, cs, pm)
