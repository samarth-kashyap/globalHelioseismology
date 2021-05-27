import numpy as np
import time


__all__ = ["crossSpectra"]


class crossSpectra():
    """Class to deal with cross-spectral computation from 
    observed data. 
    Instruments supported: HMI 
    Instrument support for future: MDI, GONG
    """

    def __init__(self, n1, l1, n2, l2, t, instrument='HMI', smooth=False,
                 daynum=1, dayavgnum=5, fit_bsl=False):
        # swapping values of ell if l2 < l1
        if l2 < l1:
            ltemp, ntemp = l2, n2
            l2, n2 = l1, n1
            l1, n1 = ltemp, ntemp

        self.delta_ell = abs(l2 - l1)
        assert n1 == n2, f"n1 != n2, forcing n2 = {n1}"
        self.n1, self.n2 = int(n1), int(n2)
        self.l1, self.l2 = int(l1), int(l2)
        self.t = t
        self.freq_global = self.get_freq(daynum, instrument)
        self.mode_data = np.loadtxt("/home/g.samarth/globalHelioseismology/mode-params/" +
                                    f"hmi.6328.36")
        self.winhalflen = self.find_winhalflen(False)
        self.freq, __ = self.slice_freq()

    def get_freq(self, daynum, instrument):
        ts_len = 138240  # array length of the time series
        t = np.linspace(0, 72*24*3600*daynum, ts_len*daynum)
        dt = t[1] - t[0]
        freq = np.fft.fftfreq(t.shape[0], dt)*1e6
        df = freq[1] - freq[0]  # frequencies in microHz
        return freq

    def slice_freq(self):
        data = self.mode_data
        print(f"winhalflen = {winhalflen}")
        cenfreq, cenfwhm, __ = cdata.findfreq(data, self.l1, self.n1, 0)
        cenfreq2, cenfwhm2, __ = cdata.findfreq(data, self.l2+6, self.n1, 0)
        cenfreq0, cenfwhm0, __ = cdata.findfreq(data, self.l1-6, self.n1, 0)
        pmfreq_p = cenfreq2 - cenfreq + (self.l2+6)*0.7
        pmfreq_n = cenfreq - cenfreq0 + (self.l2+6)*0.7
        indm = cdata.locatefreq(self.freq_global, cenfreq - pmfreq_n)
        indp = cdata.locatefreq(self.freq_global, cenfreq + pmfreq_p)

        # if fit_bsl:
        if self.fit_bsl:
            whlwindow = self.winhalflen
            indm -= extd_pixels + whlwindow
            indp += extd_pixels + whlwindow

        print(f"freqmin = {freq[indm]}; freqmax = {freq[indp]}")
        freq = freq[indm:indp]*1.0

        self.indm, self.indp = int(indm), int(indp)
        return freq, (indm, indp)


    # {{{ def find_winhalflen(freq, freq_min, freq_max):
    def find_winhalflen(self, fit_bsl=False):
        # lmax = self.l1 + 4 # old convention
        lmin, lmax, n = self.l1, self.l2, self.n1
        data, freq = self.mode_data, self.freq_global
        num_linewidths = 5

        try:
            lmax1 = lmax + 6
            assert ((data[:, 0] == lmax1) * (data[:, 1] == n)).any(), f"not found"
        except AssertionError:
            lmax1 = lmax + 4

        lmax = lmax1

        assert lmin > 6, "l should be greater than 6"
        lmin = lmin - 6

        lmax_idx = (data[:, 0] == lmax) * (data[:, 1] == n)
        lmin_idx = (data[:, 0] == lmin) * (data[:, 1] == n)
        assert lmin_idx.any(), f"Mode not found: lmin = {lmin}, n = {n}"
        assert lmax_idx.any(), f"Mode not found: lmax = {lmax}, n = {n}"

        freq_max = data[lmax_idx, 2] + num_linewidths*data[lmax_idx, 4]
        freq_min = data[lmin_idx, 2] - num_linewidths*data[lmin_idx, 4]

        # computing indices of freq_min and freq_max
        idx_min = np.argmin(np.abs(freq - freq_min))
        idx_max = np.argmin(np.abs(freq - freq_max))

        if fit_bsl:
            idx_min -= extd_pixels
            idx_max += extd_pixels
        winhalflen = (idx_max - idx_min)//2
        return winhalflen
    # }}} find_winhalflen(freq, freq_min, freq_max)

    def compute_cross_spectra(self):
        csp, csn = 0.0, 0.0
        csp2r, csp2i = 0.0, 0.0
        csn2r, csn2i = 0.0, 0.0

        for days in range(self.dayavgnum):
            day = 6328 + 72*days

            afft1p, afft1n = cdata.separatefreq(cdata.loadHMIdata_avg(self.l1, day=day))

            if self.delta_ell == 0:
                afft2p, afft2n = afft1p*1.0, afft1n*1.0
            else:
                afft2p, afft2n= cdata.separatefreq(cdata.loadHMIdata_avg(self.l2, day=day))

            afft1p = afft1p[:, self.indm:self.indp]
            afft2p = afft2p[:, self.indm:self.indp]
            afft1n = afft1n[:, self.indm:self.indp]
            afft2n = afft2n[:, self.indm:self.indp]

            # shifting the \phi2 by t
            if args.t != 0:
                afft2p = np.roll(afft2p[:(self.l1+1), :], self.t, axis=0)
                afft2n = np.roll(afft2n[:(self.l1+1), :], self.t, axis=0)

            # computing the cross-spectrum
            _csp = afft1p.conjugate()*afft2p[:(self.l1+1), :]
            _csn = afft1n.conjugate()*afft2n[:(self.l1+1), :]

            # adding the cross-spectrum (for expectation value computation)
            csp += _csp
            csn += _csn

            axp_r = plot_scatter(axp_r, _csp.real, 1)
            axn_r = plot_scatter(axn_r, _csn.real, -1)
            axp_i = plot_scatter(axp_i, _csp.imag, 1)
            axn_i = plot_scatter(axn_i, _csn.imag, -1)

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

    # {{{ def compute_d2(cs, pm):
    def compute_d2(self, cs, pm):
        csp1, freqp_win = self.derotate(cs, pm)
        csp = csp1.sum(axis=0)
        return csp**2
    # }}} compute_d2(cs, pm)

    # {{{ def derotate(phi, l, n, freq, winhalflen, sgn):
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
        freq_all, winhalflen = self.freq_global, self.winhalflen
        data = self.mode_data

        phinew = np.zeros((l+1, 2*winhalflen+1), dtype=np.complex128)
        freq_win = np.zeros(phinew.shape)

        cenfreq, __, __ = cdata.findfreq(data, l, n, 0)
        m_arr = np.arange(0, sgn*(l+1), sgn)
        for m in m_arr:
            omeganl0 = cenfreq + self.finda1(l, m) * 1e-3
            omeganl0_ind = np.argmin(np.abs(freq-omeganl0))
            _indmin = omeganl0_ind - self.winhalflen
            _indmax = omeganl0_ind +self. winhalflen + 1
            try:
                freq_win[abs(m)] = freq[_indmin:_indmax]
            except ValueError:
                print(f"l = {l}, m = {m}, omeganl0_ind = {omeganl0_ind}; " +
                      f"_indmin = {_indmin}; _indmax = {_indmax}")
            _real = phi[abs(m), _indmin-1:_indmax-1].real
            _imag = phi[abs(m), _indmin-1:_indmax-1].imag
            phinew[abs(m), :] = _real + 1j*_imag
        return phinew, freq_win
    # }}} derotate(phi, l, n, freq, winhalflen, sgn)

    # {{{ def finda1(data, l, n, m):
    def finda1(self, l, m):
        """
        Find a coefficients for given l, n, m
        """
        L = sqrt(l*(l+1))
        data = self.mode_data
        n = self.n1
        try:
            mode_idx = np.where((data[:, 0] == l) * (data[:, 1] == n))[0][0]
        except IndexError:
            print(f"MODE NOT FOUND : l = {l}, n = {n}")
            return None, None, None
        splits = np.append([0.0], data[modeindex, 12:48])
        totsplit = legval(1.0*m/L, splits)*L
        return totsplit - 31.7*m
    # }}} finda1(data, l, n, m)

