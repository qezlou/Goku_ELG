# -*- coding: utf-8 -*-
import logging
import sys
import numpy as np
import mcfit
from gal_goku import emu_cosmo
from scipy.interpolate import make_interp_spline, RectBivariateSpline
from scipy import special
from scipy.integrate import quad
class GalBase:
    """
    Convert halo summary statistics to galaxy observables such as xi(r) and w_p(r).

    This is the main class to convert halo summary statistics to galaxy observables,
    e.g., xi(r) and w_p(r), the 3D and projected correlation functions, respectively.
    """

    def __init__(self, logging_level='INFO', logger_name='BaseGal'):
        self.logger = self.configure_logging(logging_level, logger_name)





    def configure_logging(self, logging_level='INFO', logger_name='BaseGal'):
        """
        Set up logging based on the provided logging level in an MPI environment.

        Parameters
        ----------
        logging_level : str, optional
            The logging level (default is 'INFO').
        logger_name : str, optional
            The name of the logger (default is 'BaseGal').

        Returns
        -------
        logger : logging.Logger
            Configured logger instance.
        """
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging_level)

        # Create a console handler with flushing
        console_handler = logging.StreamHandler(sys.stdout)

        # Include Rank, Logger Name, Timestamp, and Message in format
        formatter = logging.Formatter(
            f'%(name)s | %(asctime)s | %(levelname)s  |  %(message)s',
            datefmt='%m/%d/%Y %I:%M:%S %p'
        )
        console_handler.setFormatter(formatter)

        # Ensure logs flush immediately
        console_handler.flush = sys.stdout.flush  

        # Add handler to logger
        logger.addHandler(console_handler)
        
        return logger
    
    def set_cosmology(self, cosmo):
        """
        Set the cosmology parameters for calculations.

        Parameters
        ----------
        cosmo : dict
            Dictionary containing cosmology parameters with keys:
            'omega0', 'omegab', 'hubble', 'scalar_amp', 'ns',
            'w0_fld', 'wa_fld', 'N_ur', 'alpha_s', 'm_nu'.
        """
        self.cosmo = cosmo
    

class Gal(GalBase):
    """
    Compute galaxy correlation functions for a given cosmology, redshift, and HOD model.

    This class contains methods to calculate the 3D correlation function xi(r) and the
    projected correlation function w_p(r) from halo summary statistics.
    """

    def __init__(self, logging_level='INFO'):
        super().__init__(logging_level=logging_level, logger_name='gal.XiGal')
        # Load the trained emulator for xi_direct
        self.xi_emu = emu_cosmo.Xi()
        self.rbins = self.xi_emu.rbins
        # Load the emulator for HMF
        self.hmf_emu = emu_cosmo.Hmf()

        # Set the resolution parameters
        # The halo mass bins for integrating the HOD model with P_hh, defining the halo mass resolution
        self.logMh = np.linspace(self.hmf_emu.mbins[0], self.hmf_emu.mbins[-1], 100)

    def reset_cosmo(self, cosmo_pars=None):
        """
        Reset core statistics that depend on cosmology.

        Recompute the halo mass function (HMF) spline interpolator across the halo mass range supported by the emulator,
        and the 3D correlation function xi(r; M_thresh1, M_thresh2) spline interpolator.

        Parameters
        ----------
        cosmo_pars : np.ndarray, shape (10,), optional
            Cosmology parameters in the order:
            'omega0', 'omegab', 'hubble', 'scalar_amp', 'ns',
            'w0_fld', 'wa_fld', 'N_ur', 'alpha_s', 'm_nu'.
        """
        # Recompute the HMF for the given cosmology as a cubic spline interpolator taking logMh as input
        self.hmf, self.hmf_err = self.hmf_emu.predict(cosmo_pars)
        
        self.hmf = self.hmf.squeeze()
        self.dndm = make_interp_spline(self.hmf_emu.mbins, self.hmf, k=3, axis=0)

        # Recompute the P_hh(M, M') for the given cosmology
        self.xi_grid, _ = self.xi_emu.predict(cosmo_pars)
        # TODO: Implement the tree-level correlation function and stitching between the two
        self.xi_grid = self.xi_grid.squeeze()
        self.xi_hh_mth_bspline = None

    def reset_hod(self, hod_params=None):
        """
        Reset all quantities computed that depend on the HOD parameters.

        Parameters
        ----------
        hod_params : dict, optional
            Dictionary of HOD parameters. If None, default parameters are used.
        """
        # Choose the default HOD parameters
        if hod_params is None:
            self.hod_params = {
                # Central galaxy parameters
                'logMmin': 12.0,
                'sig_logM': 0.7,
                'alpha_inc': 0.5,
                'Minc': 12.0,
                # Satellite galaxy parameters
                'M1': 13.0,
                'kappa': 0.5,
                'alpha': 1.0
                }
        else:
            self.hod_params = hod_params

    def get_Ncen_Nsat(self, logMh):
        """
        Compute the mean number of central and satellite galaxies for given halo masses.

        Parameters
        ----------
        logMh : array_like
            Logarithm (base 10) of halo masses.

        Returns
        -------
        N_cen : array_like
            Mean number of central galaxies.
        N_sat : array_like
            Mean number of satellite galaxies.
        """
        # Compute central galaxy number density
        Mnin = 10**self.hod_params['logMmin']
        f_inc = np.maximum(0, np.minimum(1, 1 + self.hod_params['alpha_inc'] * (logMh - self.hod_params['logMmin'])))
        N_cen = f_inc * 0.5 * (1 + special.erf((logMh - self.hod_params['logMmin']) / self.hod_params['sig_logM']))

        # Compute satellite galaxy number density
        M1 = 10**self.hod_params['M1']
        Mmin = 10**self.hod_params['logMmin']
        Mh = 10**logMh
        N_sat = N_cen * ((Mh - self.hod_params['kappa'] * Mmin) / M1) ** self.hod_params['alpha']
        self.computed_hod = True
        return N_cen, N_sat

    def get_ng(self):
        """
        Compute the number density of galaxies for the current HOD parameters and cosmology.

        Returns
        -------
        ng : float
            Number density of galaxies in (Mpc/h)^-3.
        """
        pass

    def mth_to_dens(self, log_mth):
        """
        Convert a halo mass threshold to the cumulative number density above that mass.

        Parameters
        ----------
        log_mth : float
            Logarithm (base 10) of the halo mass threshold in Msun/h.

        Returns
        -------
        dens : float
            Number density in (Mpc/h)^-3 above the given mass threshold.
        """
        # Integrate the mass function spline from mass threshold to maximum mass
        return quad(self.dndm, log_mth, self.hmf_emu.mbins[-1])[0]  # The second arg is the error on the integral

    def _interp_xi_hh_mth(self):
        """
        Compute the spline interpolators for the 3D correlation function xi(r; M_thresh1, M_thresh2).

        Returns
        -------
        None
        """
        if self.xi_hh_mth_bspline is None:
            self.xi_hh_mth_bspline = []
            for i in range(self.xi_grid.shape[-1]):
                self.xi_hh_mth_bspline.append(RectBivariateSpline(self.xi_emu.mass_bins, 
                                                        self.xi_emu.mass_bins, 
                                                        self.xi_grid[:, :, i], kx=3, ky=3))

    def xi_hh_mth(self, logm1, logm2):
        """
        Calculate the 3D correlation function xi(r; M_thresh1, M_thresh2) for arbitrary mass thresholds.

        Parameters
        ----------
        logm1 : float or np.ndarray
            Logarithm (base 10) of halo mass for the first sample.
        logm2 : float or np.ndarray
            Logarithm (base 10) of halo mass for the second sample.

        Returns
        -------
        xi_hh_mth : np.ndarray
            3D correlation function xi(r; M_thresh1, M_thresh2) for the given mass pairs.
            Shape is (m, m, r) where m is the number of mass inputs and r is the number of radial bins.
        """
        # Convert single mass pairs to arrays if necessary
        if not isinstance(logm1, np.ndarray):
            logm1 = np.array([logm1])[:, np.newaxis]
        if not isinstance(logm2, np.ndarray):
            logm2 = np.array([logm2])[:, np.newaxis]

        # Compute the spline interpolators for the 3D correlation function
        self._interp_xi_hh_mth()
        # Initialize output array with NaNs
        xi_hh_mth = np.full((logm1.shape[0], logm1.shape[1], len(self.xi_hh_mth_bspline)), np.nan)
        # Evaluate the interpolators over the input mass pairs for each radial bin
        for i in range(len(self.xi_hh_mth_bspline)):
            xi_hh_mth[:, :, i] = self.xi_hh_mth_bspline[i].ev(logm1, logm2)
        return xi_hh_mth.squeeze()

    def xi_hh_m1m2(self, masses):
        """
        Compute the 3D correlation function xi(r) at exact halo masses using finite difference.

        Parameters
        ----------
        masses : tuple of float
            Tuple containing (logm1, logm2), the logarithms (base 10) of halo masses for the two samples.

        Returns
        -------
        xi_exact_mass : np.ndarray
            3D correlation function xi(r) at the exact halo masses.
        """
        # Use finite difference to compute correlation function at exact masses
        delta = 0.01  # log10 mass step size
        logm1, logm2 = masses[0], masses[1]
        logm1p, logm1m = logm1 + delta, logm1 - delta
        logm2p, logm2m = logm2 + delta, logm2 - delta

        d1p = self.mth_to_dens(logm1p)
        d1m = self.mth_to_dens(logm1m)
        d2p = self.mth_to_dens(logm2p)
        d2m = self.mth_to_dens(logm2m)
        
        xi_mm = self.xi_hh_mth(logm1m, logm2m)
        xi_mp = self.xi_hh_mth(logm1m, logm2p)
        xi_pm = self.xi_hh_mth(logm1p, logm2m)
        xi_pp = self.xi_hh_mth(logm1p, logm2p)
        numer = xi_mm * d1m * d2m - xi_mp * d1m * d2p - xi_pm * d1p * d2m + xi_pp * d1p * d2p
        denom = d1m * d2m - d1m * d2p - d1p * d2m + d1p * d2p

        xi_exact_mass = numer / denom

        return xi_exact_mass


    def p_hh_m1m2(self, masses):
        """
        Compute the 3D power spectrum P(k) at exact halo masses using finite difference.

        Parameters
        ----------
        masses : tuple of float
            Tuple containing (logm1, logm2), the logarithms (base 10) of halo masses for the two samples.

        Returns
        -------
        k : np.ndarray
            Wave numbers.
        phh : np.ndarray
            3D power spectrum P(k) at the given halo masses.
        """
        xi_exact_mass = self.xi_hh_m1m2(masses)
        # Use mcfit to convert xi(r) to P(k)
        k, phh = mcfit.xi2P(self.rbins, l=0, lowring=True)(xi_exact_mass, extrap=True)
        return k, phh.squeeze()


    def get_xi_gg(self, cosmo, hod):
        """
        Compute the galaxy correlation function xi(r) for a given cosmology and HOD parameters.

        Parameters
        ----------
        cosmo : list or np.ndarray
            Cosmology parameters.
        hod : list or np.ndarray
            HOD parameters.

        Returns
        -------
        xi_gg : np.ndarray
            Galaxy correlation function xi(r).
        """
        pass
    
    def get_pk_gg(self):
        """
        Compute the galaxy power spectrum P_gg(k) for the current cosmology and HOD parameters.

        This applies the HOD to the emulated P_hh(M, M').

        The computation flow is:
        1. Get dndm for self.Mh.
        2. Get P_hh(M, M') on the (self.Mh, self.Mh) grid.
        3. Compute <N_cen>(M) and <N_sat>(M) for self.Mh.
        4. Compute the displacement kernels H_off(k,M) and u_s(k,M).
        5. Integrate the terms P_1c(k), P_2c(k), P_1s(k), and P_2s(k).

        Returns
        -------
        P_gg : np.ndarray
            Galaxy power spectrum P_gg(k).
        """
        pass

    
        
        


        