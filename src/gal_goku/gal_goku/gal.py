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
    The main class to convert halo summary statistics to galaxy observables, e.g. xi(r) and
    w_p(r), the 3d and projected correlation functions, respectively.
    """

    def __init__(self, logging_level='INFO', logger_name='BaseGal'):
        self.logger = self.configure_logging(logging_level, logger_name)





    def configure_logging(self, logging_level='INFO', logger_name='BaseGal'):
        """Sets up logging based on the provided logging level in an MPI environment."""
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
        Set the cosmology to be used in the calculations.
        Parameters
        ----------
        cosmo : dict
            A dictionary containing the cosmology parameters.
            The keys should be:
            'omega0', 'omegab', 'hubble', 'scalar_amp', 'ns',
            'w0_fld', 'wa_fld', 'N_ur',  'alpha_s', 'm_nu'
        """
        self.cosmo = cosmo
    

class Gal(GalBase):
    """
    Galaxy correlation functions for a cosmology, redshift, and HOD model.
    This class contains methods to calculate the 3D correlation function xi(r) and the
    projected correlation function w_p(r) from the halo summary statistics.
    """

    def __init__(self, logging_level='INFO'):
        super().__init__(logging_level=logging_level, logger_name='gal.XiGal')
        # Load the trained emulator for xi_direct
        self.xi_emu = emu_cosmo.Xi()
        self.rbins = self.xi_emu.rbins
        # Laod the meualtor for HMF
        self.hmf_emu = emu_cosmo.Hmf()

        # Set the resolution paramters
        ## The halo mass bins for intergating the HOD model with P_hh, this defines the halo mass resolution
        self.logMh = np.linspace(self.hmf_emu.mbins[0], self.hmf_emu.mbins[-1], 100)

    def reset_cosmo(self, cosmo_pars=None):
        """"
        Reset all the core statistics that are cosmology dependent.
        
        The stats recomputed are:
        1. self.hmf: The halo mass function (HMF) spline interpolator across
        the halo mass range supported by the emulato, i.e. ~ 1e11-1e13
        2. self.xi_hh_mth: The 3D correlation function xi(r;M_thresh1, M_thresh2) 
        spline interpolator across the halo mass range supported by the emulator.
        
        Parameters
        ----------
        cosmo_pars : np.ndarray. shpe (10,)
            The cosmology parameters to use for the calculation.
            The order of the parameters should be:
            'omega0', 'omegab', 'hubble', 'scalar_amp', 'ns',
            'w0_fld', 'wa_fld', 'N_ur',  'alpha_s', 'm_nu'
        """
        # Re-compute the HMF for the given cosmology, it is a 
        # cubic spline interpolator talking logMh as input
        self.hmf, self.hmf_err = self.hmf_emu.predict(cosmo_pars)
        
        self.hmf = self.hmf.squeeze()
        self.dndm = make_interp_spline(self.hmf_emu.mbins, self.hmf, k=3, axis=0)

        # Recompute the P_hh(M, M') for the given cosmology
        self.xi_grid, _ = self.xi_emu.predict(cosmo_pars)
        # TODO:  the tree-level correlation function
        # and the stitching between the two
        self.xi_grid = self.xi_grid.squeeze()
        self.xi_hh_mth_bspline = None

    def reset_hod(self, hod_params=None):
        """
        Reset all the quantities computd that are HOD dependent.
        """
        # Choose the default HOD parameters
        if hod_params is None:
            self.hod_params = {
                # Center galaxy parameters
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
        Compute the HOD for the given HOD parameters, `hod_params`.
        It is set to `self.computed_hod=-True`, after it's done.
        """
        # Compute Central galaxy Number density
        Mnin = 10**self.hod_params['logMmin']
        f_inc = np.maximum(0, np.minimum(1, 1+self.hod_params['alpha_inc']*(logMh-self.hod_params['logMmin'])))
        N_cen = f_inc * 0.5 * ( 1 + special.erf((logMh - self.hod_params['logMmin']) / self.hod_params['sig_logM']))

        # Compute Satellite galaxy Number density
        M1 = 10**self.hod_params['M1']
        Mmin= 10**self.hod_params['logMmin']
        Mh = 10**logMh
        N_sat = N_cen * ( (Mh - self.hod_params['kappa']*Mmin) / M1)**self.hod_params['alpha'] 
        self.computed_hod = True
        return N_cen, N_sat

    def get_ng(self):
        """
        Get the number density of galaxies for the given HOD parameters and cosmology.
        Returns
        -------
        ng : float
            The number density of galaxies in (Mpc/h)^-3.
        """
        pass

    def mth_to_dens(self, log_mth):
        """
        Convert the threshold mass to density.
        Parameters
        ----------
        mth : float
            The halo mass in Msun/h.
        Returns
        -------
        dens : float
            The density in Msun/h/Mpc^3.
        """
        # Get the density for the given mass threshold
        return quad(self.dndm, log_mth, self.hmf_emu.mbins[-1])[0] # The second arg would be the error on the integral

    def _interp_xi_hh_mth(self):
        """
        Calculate the 3D correlation function xi(r;M_thresh1, M_thresh2) for arbitrary 
        mass thresholds interpolated form the emulator's prediction on the grid of mass pairs.
        Parameters
        ----------
        cosmo_pars : list or np.ndarrays
            The desired cosmology to predict the xi_hh for
        Returns
        -------
        xi : scipy.interpolate.Bspline
            A RectBivariateSpline object representing the 3D correlation function xi(r,M_th1, M_th2).
        """
        if self.xi_hh_mth_bspline is None:
            self.xi_hh_mth_bspline = []
            for i in range(self.xi_grid.shape[-1]):
                self.xi_hh_mth_bspline.append(RectBivariateSpline(self.xi_emu.mass_bins, 
                                                        self.xi_emu.mass_bins, 
                                                        self.xi_grid[:,:,i], kx=3, ky=3))
    def xi_hh_mth(self, logm1, logm2):
        """
        Calculate the 3D correlation function xi(r;M_thresh1, M_thresh2) for arbitrary
        mass thresholds interpolated form the emulator's prediction on the grid of mass pairs.
        Parameters
        ----------
        logm1, logm2 : flaot or np.array, shape (m, m)
            The halo masses for the first and second halo samples.
        Returns
        -------
        xi_hh_mth : np.array, shape (m, m, r)
            The 3D correlation function xi(r;M_thresh1, M_thresh2) for the given mass pairs.
        """
        # If single mass pairs are given, convert them to arrays
        if type(logm1) is not np.ndarray:
            logm1 = np.array([logm1])[:, np.newaxis]
        if type(logm2) is not np.ndarray:
            logm2 = np.array([logm2])[:, np.newaxis]

        # Compute the spline interpolator for the 3D correlation function
        self._interp_xi_hh_mth()
        # this will have the shape (m, m, r)
        xi_hh_mth = np.full((logm1.shape[0], logm1.shape[1], len(self.xi_hh_mth_bspline)), np.nan)
        # iterate over the r-bins to evaluate the interpolators
        for i in range(len(self.xi_hh_mth_bspline)):
            xi_hh_mth[:,:,i] = self.xi_hh_mth_bspline[i].ev(logm1, logm2)
        return xi_hh_mth.squeeze()

    def xi_hh_m1m2(self, masses):
        """
        Get the 3D correlation function xi(r) at exact halo masses using finite difference.
        Parameters
        ----------
        masses : tuple
            The halo masses for the first and second halo sample.
        cosmo : list or np.ndarrays
            The desired cosmology to predict the xi_hh for
        Returns
        -------
        xi_exact_mass : array_like
            The 3D correlation function xi(r) at exact halo masses.
        """
        # Use the finite difference trick to compute the correlation function at exact masses
        delta = 0.01  # log10 mass step size
        logm1, logm2 = masses[0], masses[1]
        logm1p, logm1m = logm1 + delta, logm1 - delta
        logm2p, logm2m = logm2 + delta, logm2 - delta

        d1p = self.mth_to_dens(logm1p)
        d1m = self.mth_to_dens(logm1m)
        d2p = self.mth_to_dens(logm2p)
        d2m = self.mth_to_dens(logm2m)
        
        xi_mm= self.xi_hh_mth(logm1m, logm2m)
        xi_mp= self.xi_hh_mth(logm1m, logm2p)
        xi_pm= self.xi_hh_mth(logm1p, logm2m)
        xi_pp= self.xi_hh_mth(logm1p, logm2p)
        numer = xi_mm * d1m * d2m - xi_mp * d1m * d2p - xi_pm * d1p * d2m + xi_pp * d1p * d2p
        denom = d1m * d2m - d1m * d2p - d1p * d2m + d1p * d2p

        xi_exact_mass = numer / denom

        return xi_exact_mass


    def p_hh_m1m2(self, masses):
        """
        Get the 3D power spectrum P(k) at exact halo masses using finite difference.
        Parameters
        ----------
        masses : tuple
            The halo masses for the first and second halo sample.
        cosmo : list or np.ndarrays
            The desired cosmology to predict the xi_hh for
        """
        xi_exact_mass = self.xi_hh_m1m2( masses)
        # Use mcfit to convert xi to Pk
        k, phh = mcfit.xi2P(self.rbins, l=0, lowring=True)(xi_exact_mass, extrap=True)
        return k, phh.squeeze()


    def get_xi_gg(self, cosmo, hod):
        """
        Get xi(r) for a cosmology. It applies the HOD to emualted p_hh(M, M').
        Parameters
        ----------
        cosmo : list or np.ndarrays
            The desired cosmology to predict the xi_hh for
        hod : list or np.ndarrays
            The HOD parameters to use for the calculation.
        """

        pass
    
    def get_pk_gg(self):
        """
        Get P_gg(k) for the chosen cosmology and HOD parameters set
        in `self.reset_cosmo()` and `self.reset_hod()`.
        It applies the HOD to emualted p_hh(M, M').
        The computation flow:
        1. Geth dndm for the self.Mh
        2. Get the P_hh(M,M') for the (self.Mh, self.Mh) grid
        3. Get <N_cen>(M) and <N_sat>(M) for the self.Mh
        4. Comopute the displcaing kernels, H_off(k,M) and u_s(k,M)
        5. Integrate the P_1c(k), P_2c(k), P_1s(k), and P_2s(k) terms
        """

        pass

    
        
        


        