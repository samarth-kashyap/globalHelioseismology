import numpy as np
import healpy as hp
from pyshtools import legendre as pleg
from scipy.integrate import simps
import healpy as hp

__all__ = ["gen_leg_x", "CtoL"]


def gen_leg_x(lmax, x):
    max_index = int(lmax+1)
    ell = np.arange(max_index)
    norm = np.sqrt(ell*(ell+1)).reshape(max_index, 1)
    norm[norm == 0] = 1.0
    
    leg = np.zeros((max_index, x.size))
    leg_d1 = np.zeros((max_index, x.size))
    
    count = 0
    for z in x:
        leg[:, count], leg_d1[:, count] = pleg.PlBar_d1(lmax, z)
        count += 1
    return leg/np.sqrt(2)/norm, leg_d1/np.sqrt(2)/norm



def rotate_map_old(hmap, rot_theta, rot_phi):
    """
    Take hmap (a healpix map array) and return another healpix map array 
    which is ordered such that it has been rotated in (theta, phi) by the 
    amounts given.
    """
    nside = hp.npix2nside(len(hmap))

    # Get theta, phi for non-rotated map
    t, p = hp.pix2ang(nside, np.arange(hp.nside2npix(nside))) #theta, phi

    # Define a rotator
    r = hp.Rotator(deg=False, rot=[rot_phi,rot_theta])

    # Get theta, phi under rotated co-ordinates
    trot, prot = r(t,p)

    # Interpolate map onto these co-ordinates
    rot_map = hp.get_interp_val(hmap, trot, prot)

    return rot_map


def rotate_map(hmap, euler_angle=np.array([0, -np.pi/2.0, 0.0])):
    """
    Take hmap (a healpix map array) and return another healpix map array 
    which is ordered such that it has been rotated in (theta, phi) by the 
    amounts given.
    """
    nside = hp.npix2nside(len(hmap))

    # Get theta, phi for non-rotated map
    t, p = hp.pix2ang(nside, np.arange(hp.nside2npix(nside))) #theta, phi

    # Define a rotator
    r = hp.rotator.Rotator(euler_angle, deg=False, eulertype='zxz')

    # Get theta, phi under rotated co-ordinates
    trot, prot = r(t,p)

    # Interpolate map onto these co-ordinates
    rot_map = hp.get_interp_val(hmap, trot, prot)

    return rot_map




class CtoL():
    __all__ = ["get_ctol", "get_sph_coeffs"]
    __attributes__ = ["b_angle", "th", "ctol"]

    def __init__(self, b_angle=0.0, NSIDE=64):
        self.b_angle = b_angle
        self.NSIDE = NSIDE
        self.theta, self.ctol = self.get_ctol()
        self.hmap, self.alm = self.get_sph_coeffs(self.ctol, self.theta, NSIDE=NSIDE)

    def get_ctol(self, ci=np.array([0.0, -627.06, 335.41, 154.07, -372.02, 363.59])):
        """Get the Center-to-Limb profile using data from 
        dopplergrams. 

        Inputs:
        -------
        ci - np.ndarray(ndim=1, dtype=np.float64)
            coefficients for shifted-Legendre polynomials (Hathaway 2015) in m/s

        Returns:
        --------
        (th, ctol)
        th - np.ndarray(ndim=1, dtype=np.float64)
            array containing latitude coordinates (pole at disk center)
        ctol - np.ndarray(ndim=1, dtype=np.float64) 
            the center-to-limb effect as a function of latitude
            - Unit: m/s
        """
        th = np.linspace(1e-4, np.pi/2.0-1e-4, 5*self.NSIDE)
        costh, sinth = np.cos(th), np.sin(th)
        x = 1 - sinth

        plx, dt_plx = gen_leg_x(5, x)
        ctol = ci.dot(plx)
        return th, ctol

    def get_sph_coeffs(self, ctol, theta, NSIDE=64):
        """Get the spherical harmonic coefficients for the given 
        CtoL profile.

        Inputs:
        -------
        ctol - np.ndarray(ndim=1, dtype=np.float64)
            Center-to-Limb effect in m/s
        theta - np.ndarray(ndim=1, dtype=np.float64)
            latitude in radians (pole at disk center)
        NSIDE - int
            NSIDE for the healPix map

        Returns:
        --------
        alm - np.ndarray(ndim=1, dtype=np.complex128)
            Spherical harmonic coefficients of the CtoL profile (pole at disk center)
        
        Notes:
        ------
        The Center-to-Limb effect is an axisymmetric when the pole is 
        located at the disk center. Hence the spherical harmonic coefficients
        are nonzero for m=0 only.
        """
        lmax = int(3*NSIDE - 1)
        NPIX = hp.nside2npix(NSIDE)
        ellArr, emmArr = hp.sphtfunc.Alm.getlm(lmax)
        len_alm = len(ellArr)

        costh, sinth = np.cos(theta), np.sin(theta)
        pl, dt_pl = gen_leg_x(lmax, costh)
        alm = np.zeros(len_alm, dtype=np.complex128)

        for ell in range(lmax):
            mask_ell = (ellArr == ell) * (emmArr == 0)
            alm[mask_ell] = simps(ctol*pl[ell]*sinth, x=theta)
        hmap = hp.sphtfunc.alm2map(alm, NSIDE)
        t, p = hp.pix2ang(NSIDE, np.arange(NPIX))
        maskt = t > np.pi/2.0 - 1e-2
        # hmap[maskt] = hp.UNSEEN
        hmap[maskt] = 0.0
        alm = hp.sphtfunc.map2alm(hmap)

        return hmap, alm
