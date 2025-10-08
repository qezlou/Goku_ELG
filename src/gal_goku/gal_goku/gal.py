# -*- coding: utf-8 -*-
import logging
import sys
import numpy as np
import mcfit
from gal_goku import emu_cosmo
from scipy.interpolate import make_interp_spline, RectBivariateSpline, UnivariateSpline, interp1d
from scipy import special
from scipy.integrate import quad, simpson
from gal_goku import halo_tools
from . import init_power


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
    
    def get_init_linear_power(self, cosmo_pars, nu_acc=1e-5, redshifts=[99]):
        """
        Get the initial linear power spectrum for the cosmology
        set in the constructor. Using `classylss`
        Parameters
        ----------
        z : float, optional
            Redshift at which to evaluate the power spectrum. If None, uses the redshift set in the constructor.
        
        Returns
        -------
        np.ndarray
            The linear power spectrum at the specified redshift and wavenumbers.
        """

        sim_ics = init_power.SimulationICs(redshifts=redshifts,
                                           omega0=cosmo_pars[0], omegab=cosmo_pars[1],
                                           hubble= cosmo_pars[2], scalar_amp= cosmo_pars[3],
                                           ns=cosmo_pars[4], w0_fld=cosmo_pars[5],
                                           wa_fld=cosmo_pars[6], N_ur=cosmo_pars[7],
                                           alpha_s=cosmo_pars[8], m_nu= cosmo_pars[9],
                                           nu_acc=nu_acc
                                           )
        # We set maxk=0.5 to significantly save on the computing time
        all_ks, all_pks_lin = sim_ics.cambfile(maxk=0.5)
        return all_ks.squeeze(), all_pks_lin.squeeze()

class LargeScaleGal(GalBase):
    """
    Compute large-scale galaxy correlation functions at large scales, r > 40 Mpc/h.
    
    NOTE: Maybe merge with `Gal` class in the future.
    """
    def __init__(self, z):
        """
        Parameters
        ----------
        z : float
            Redshift at which to compute the large-scale galaxy correlation functions.
        
        """
        self.z = z
        # From experience this range and spacing is good 
        # enough for BAO at z=2.5
        self.k = np.logspace(np.log10(7e-3), np.log10(0.5), num=500)
    
    def reset_cosmo(self, cosmo_pars, gk):
        """
        Reset the cosmology by re-evaluating the emulator for G(k), the propagator, and 
        the linear power spectrum at z=99. One would get P_ab = G_a * G_b * P_lin(k), where 
        a and b are two samples of the galaxies. 

        NOTE: For now, we don't have an emulator for G(k), so we pass G(k) directly.
        """
        self.gk = gk
        k_init, p_init = self.get_init_linear_power(cosmo_pars=cosmo_pars)

        # Resample p_init to the logarithmicaly space self.k
        self.p_init = interp1d(k_init, np.copy(p_init), axis=-1, 
                               bounds_error=False, 
                               fill_value="extrapolate")(self.k)
        pass

class Gal(GalBase):
    """
    Compute galaxy correlation functions for a given cosmology, redshift, and HOD model.

    This class contains methods to calculate the 3D correlation function xi(r) and the
    projected correlation function w_p(r) from halo summary statistics.
    """

    def __init__(self, hmf_model_file, hmf_composite_kernel= None, z=2.5, leave=18, config=None, tag_info=None, logging_level='INFO'):
        """
        Initialize the Gal class with cosmology, redshift, and HOD model parameters.
        Parameters
        ----------
        z : float, optional
            Redshift at which to compute the galaxy correlation functions (default is 2.5).
        leave : int, optional
            Number of simulations to leave out for cross-validation (default is, sim 522 which the 
            truth is not good either).
        """
        self.z =z
        super().__init__(logging_level=logging_level, logger_name='gal.XiGal')
        # Load the emulator for HMF
        self.hmf_emu = emu_cosmo.Hmf(model_file=hmf_model_file, hmf_composite_kernel=hmf_composite_kernel, loggin_level=logging_level)
        self._load_config(config)
        # Load the trained emulator for xi_direct
        self.xi_emu = emu_cosmo.Xi(leave=leave, loggin_level=logging_level)
        self.rbins = self.xi_emu.rbins
        # We will only use the prediciton within r_range
        self.xi_emu_r_mask = ((self.rbins > self.config['r_range'][0]) & 
                              (self.rbins < self.config['r_range'][1]))
        self.rbins = self.rbins[self.xi_emu_r_mask]

        self.logger.debug(f'Emulators mass bins: {self.xi_emu.mass_bins}')
        self.logger.debug(f'Emulators rbins.size: {self.xi_emu.rbins.size}')
        self.rbins_fine = np.linspace(self.rbins[0], self.rbins[-1], 400)
        
        
        # Set the resolution parameters
        self.tag_info = tag_info

    def _load_config(self, config=None):
        """
        Load the configuration parameters for the emulator.

        Parameters
        ----------
        config : dict
            Dictionary containing configuration parameters.
        """
        if config is None:
            self.config = {
                # The halo mass bins for integrating the HOD model with P_hh, 
                # defining the halo mass resolution
                'logMh': np.arange(self.hmf_emu.mbins[0], self.hmf_emu.mbins[-1], 0.05),
                'smooth_xihh_r': 0,
                'smooth_phh_k': 0,
                'smooth_xihh_mass': 0,
                'r_range': [0.5, 50], # We need this cut to ensure stable Hankel transform.
                                      # The small and large scales are both fluctuating
                                      # The small scales should be overridden by the 1-halo term in
                                      # future. The large scales are fluctuating since we are
                                      # in log-scale
                'dm': 0.01,  # log10 mass step size for finite difference
                }
        else:
            self.config = config

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
        self.log_dndlog_m_spl = make_interp_spline(self.hmf_emu.mbins, np.log10(self.hmf), k=3, axis=0)
        
        # Recompute the P_hh(M, M') for the given cosmology
        self.xi_grid, _ = self.xi_emu.predict(cosmo_pars)
        print(f"xi_grid shape: {self.xi_grid.shape}")
        self.xi_grid = self.xi_grid[ :, :, self.xi_emu_r_mask].squeeze()
        # TODO: Implement the tree-level correlation function and stitching between the two
        self.xi_grid = self.xi_grid.squeeze()
        # get the power on the grid of mass thresholds defined by the emulator
        self.k, self.p_grid = self.xi_grid_to_p_grid()
        self.xi_hh_mth_bspline = None
        self.p_hh_mth_bspline = None

        # Set the halot_tools for modeling the NFW profile
        self.htools = halo_tools.HaloTools(cosmo_pars, self.z)
        self.uk = self.htools.get_uk(self.k, 10**self.config['logMh'])

    def dndlog_m(self, logMh):
        """
        Compute the halo mass function (HMF) for given halo masses.

        Parameters
        ----------
        logMh : array_like
            Logarithm (base 10) of halo masses.

        Returns
        -------
        dndlog_m : array_like
            Halo mass function (HMF) for the given halo masses.
        """
        return 10**self.log_dndlog_m_spl(logMh)
    
    def dndm(self, logMh):
        """
        Compute the halo mass function (HMF) for given halo masses.
        Convert dN/dlogM to dN/dM
        This function takes the log10 of the mass and returns the dn/dM

        Parameters
        ----------
        logMh : array_like
            Logarithm (base 10) of halo masses.

        Returns
        -------
        dndlog_m : array_like
            Halo mass function (HMF) for the given halo masses.
        """
        dndlog_m = self.dndlog_m(logMh)
        return dndlog_m/(10**logMh * np.log(10))
    

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
                'logMmin': 11.3,
                'sig_logM': 0.1,
                #'alpha_inc': 0.5,
                #'Minc': 12.0,
                # Satellite galaxy parameters
                'M1': 12.5,
                'kappa': 1.1,
                'alpha': 2.0
                }
        else:
            self.hod_params = hod_params

    def _Ncen(self, logMh):
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
        #f_inc = np.maximum(0, np.minimum(1, 1 + self.hod_params['alpha_inc'] * (logMh - self.hod_params['logMinc'])))
        f_inc = 1
        N_cen = f_inc * 0.5 * (1 + special.erf((logMh - self.hod_params['logMmin']) / self.hod_params['sig_logM']))
        return N_cen
    
    def _lambda_sat(self, logMh):
        
        N_cen = self._Ncen(logMh)
        # Compute satellite galaxy number density
        M1 = 10**self.hod_params['M1']
        Mmin = 10**self.hod_params['logMmin']
        Mh = 10**logMh

        lambda_sat =  ((Mh - self.hod_params['kappa'] * Mmin) / M1) ** self.hod_params['alpha']
        return lambda_sat

    def _Nsat(self, logMh):
        """
        Compute the mean number of satellite galaxies for given halo masses.
        Parameters
        ----------
        logMh : array_like
            Logarithm (base 10) of halo masses.
        Returns
        -------
        N_sat : array_like
            Mean number of satellite galaxies.
        """
        lambda_sat = self._lambda_sat(logMh)
        # Compute the mean number of satellite galaxies
        Ncen = self._Ncen(logMh)
        N_sat = Ncen * lambda_sat
        return N_sat

    def get_ng(self, logMh=None):
        """
        Compute the mean number density of galaxies for given halo masses.

        Parameters
        ----------
        logMh : array_like, optional
            Logarithm (base 10) of halo masses.
            If None, uses the default logMh set in the class.

        Returns
        -------
        ng : array_like
            Mean number density of galaxies.
        """
        if logMh is None:
            logMh = self.config['logMh']
        integrand = lambda m: self.dndlog_m(m) * ( self._Ncen(m) + self._Nsat(m))
        return quad(integrand, logMh[0], logMh[-1])[0]
    

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
        return quad(self.dndlog_m, log_mth, self.hmf_emu.mbins[-1])[0]  # The second arg is the error on the integral

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
                                                        self.xi_grid[:, :, i], 
                                                        kx=3, ky=3, s=self.config['smooth_xihh_mass']))

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
        delta_m_deriv = self.config['dm']  # log10 mass step size for finite difference
        logm1, logm2 = masses[0], masses[1]
        logm1p, logm1m = logm1 + self.delta_m_deriv, logm1 - self.delta_m_deriv
        logm2p, logm2m = logm2 + self.delta_m_deriv, logm2 - self.delta_m_deriv

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

        #spl = make_interp_spline(self.xi_emu.rbins, xi_exact_mass, k=3)
        spl = UnivariateSpline(self.rbins, xi_exact_mass, s=self.config['smooth_xihh_r'], k=3)
        xi_exact_mass = spl(self.rbins_fine)

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
        r_mask = self.rbins_fine < self.config['r_range'][1]
        r_mask *= self.rbins_fine > self.config['r_range'][0]
        k, phh = mcfit.xi2P(self.rbins_fine[r_mask], l=0, lowring=True)(xi_exact_mass[r_mask], extrap=True)

        spl = UnivariateSpline(k, phh, s= self.config['smooth_phh_k'], k=3)
        phh = spl(k)
        return k, phh.squeeze()



    def _pgg_1h_ss(self):
        """
        Get the 1-halo term for the central-satellite correlation function
        """
        integrand = lambda m: self.dndm(m) * 2 *self._Nsat(m) * self.halo_tools.get_uk(self.k, 10**m)
        return (1/self.get_ng()**2) * quad(integrand, self.config['logMh'][0], self.config['logMh'][-1])[0]
    
    def _pgg_1h_cs(self):
        """
        Get the 1-halo term for the central-satellite correlation function
        """
        integrand = lambda m: self.dndm(m) * (self._Nsat(m)**2 / self._Ncen(m)) * self.halo_tools.get_uk(self.k, 10**m)**2
        return (1/self.get_ng()**2) * quad(integrand, self.config['logMh'][0], self.config['logMh'][-1])[0]

    def _interp_p_grid(self):
        """
        Compute the spline interpolators for the power spectrum P_hh(M_thresh1, M_thresh2).

        Returns
        -------
        None
        """
        # interpolate the power spectrum P_hh(M_thresh1, M_thresh2) on the grid of mass thresholds
        if self.p_hh_mth_bspline is None:
            self.p_hh_mth_bspline = []
            for i in range(self.k.size):
                self.p_hh_mth_bspline.append(RectBivariateSpline(self.xi_emu.mass_bins, 
                                                        self.xi_emu.mass_bins, 
                                                        self.p_grid[:, :, i], 
                                                        kx=3, ky=3, s=self.config['smooth_xihh_mass']))    
    def xi_grid_to_p_grid(self):
        """
        Get the power spectrum P_hh(m_th1, mth2) on the grid of mass thresholds defined
        by the emulator.
        """
        self.logger.debug("Converting the emulated xi_hh(mth1, mth2) to P_hh(mth1, mth2)")
        print(self.rbins.shape, self.xi_grid.shape)
        spl = UnivariateSpline(self.rbins, self.xi_grid[0, 0], k=3, s=self.config['smooth_xihh_r'])
        xi_interp = spl(self.rbins_fine) 
        k, _ = mcfit.xi2P(self.rbins_fine, l=0, lowring=True)(xi_interp, extrap=True)

        # placehoder for the full grid of power at mass thresholds
        p_grid = np.full((len(self.xi_emu.mass_bins), len(self.xi_emu.mass_bins), k.size), np.nan)
        for i in range(len(self.xi_emu.mass_bins)):
            for j in range(len(self.xi_emu.mass_bins)):
                # We don't use NaN values in the xi_grid, so we need to mask them
                ind_finite = np.where(np.isfinite(self.xi_grid[i, j]))[0]
                spl = UnivariateSpline(self.rbins[ind_finite], self.xi_grid[i, j][ind_finite], k=3, s=self.config['smooth_xihh_r'])
                xi_interp = spl(self.rbins_fine)
                _, p_grid[i,j] = mcfit.xi2P(self.rbins_fine, l=0, lowring=True)(xi_interp, extrap=True)
        return k, p_grid

    def get_p_hh_mth(self, logm1, logm2):
        
        # Convert single mass pairs to arrays if necessary
        if not isinstance(logm1, np.ndarray):
            logm1 = np.array([logm1])[:, np.newaxis]
        if not isinstance(logm2, np.ndarray):
            logm2 = np.array([logm2])[:, np.newaxis]

        # Compute the spline interpolators for the 3D correlation function
        self._interp_p_grid()
        # Initialize output array with NaNs
        p_hh_mth = np.full((logm1.shape[0], logm1.shape[1], len(self.p_hh_mth_bspline)), np.nan)
        # Evaluate the interpolators over the input mass pairs for each radial bin
        for i in range(len(self.p_hh_mth_bspline)):
            p_hh_mth[:, :, i] = self.p_hh_mth_bspline[i].ev(logm1, logm2)
        return p_hh_mth.squeeze()       

    def get_phh_m1m2_reversed(self, masses):
        """
        First take the FFTLog of the xi_hh_m1m2 on the emualted mass_threshold grid
        and calclute on the exact mass grid of logMg X logMh usign the finite difference
        """
        # Now we need to take the FFTLog of the xi_hh_m1m2 on the emulated mass_threshold grid
        
        # and calculate on the exact mass grid of logMg X logMh using the finite difference
        delta = 0.01  # log10 mass step size
        logm1, logm2 = masses[0], masses[1]
        logm1p, logm1m = logm1 + delta, logm1 - delta
        logm2p, logm2m = logm2 + delta, logm2 - delta

        d1p = self.mth_to_dens(logm1p)
        d1m = self.mth_to_dens(logm1m)
        d2p = self.mth_to_dens(logm2p)
        d2m = self.mth_to_dens(logm2m)

        p_mm = self.get_p_hh_mth(logm1m, logm2m)
        p_mp = self.get_p_hh_mth(logm1m, logm2p)
        p_pm = self.get_p_hh_mth(logm1p, logm2m)
        p_pp = self.get_p_hh_mth(logm1p, logm2p)
        numer = p_mm * d1m * d2m - p_mp * d1m * d2p - p_pm * d1p * d2m + p_pp * d1p * d2p
        denom = d1m * d2m - d1m * d2p - d1p * d2m + d1p * d2p
        p_exact_mass = numer / denom
        return p_exact_mass


    def get_phh_m1m2_matrix_reversed(self):
        
        phh_m1m2_mat = np.full((len(self.config['logMh']), len(self.config['logMh']), self.p_grid.shape[-1]), np.nan)
        for i in range(len(self.config['logMh'])):
            for j in range(len(self.config['logMh'])):
                phh = self.get_phh_m1m2_reversed((self.config['logMh'][i], self.config['logMh'][j]))
                assert not np.any(np.isnan(phh)), f"{self.tag_info}, p_hh_m1m2 returned NaN for masses {self.config['logMh'][i]} and {self.config['logMh'][j]}"
                phh_m1m2_mat[i, j] = phh
                phh_m1m2_mat[j, i] = phh_m1m2_mat[i, j]
        return self.k, phh_m1m2_mat        

    def get_phh_m1m2_matrix(self):
        """
        Get the phhh_m1m2 on a grid of logMh, logMh' values.
        This is used to compute the 2-halo and 1-ha;p term for the central-central 
        and central-sattelite correlation function.
        """
        # get the p_hh on (M,M') grid
        k, _ = self.p_hh_m1m2((self.config['logMh'][1], self.config['logMh'][1]))
        phh_m1m2_mat = np.full((len(self.config['logMh']), len(self.config['logMh']), k.size), np.nan)
        for i in range(len(self.config['logMh'])):
            for j in range(len(self.config['logMh'])):
                k, phh = self.p_hh_m1m2((self.config['logMh'][i], self.config['logMh'][j]))
                assert not np.any(np.isnan(phh)), f"p_hh_m1m2 returned NaN for masses {self.config['logMh'][i]} and {self.config['logMh'][j]}"
                phh_m1m2_mat[i, j] = phh
                phh_m1m2_mat[j, i] = phh_m1m2_mat[i, j]
        return k, phh_m1m2_mat
    
    def _pgg_2h_cc(self):
        """
        Get the 2-halo term for the central-central correlation function 
        """
        # get the p_hh on (M,M') grid
        k, phh_m1m2_mat = self.get_phh_m1m2_matrix_reversed()
        # Set up the mass-dependent arrays
        # dndm_vals: dn/dlog10M replicated along k
        dndm_vals = np.tile(self.dndlog_m(self.config['logMh']), (k.size, 1)).T  # dn/dlog10M replicated along k
        Ncen_vals = np.tile(self._Ncen(self.config['logMh']), (k.size, 1)).T

        # First integral: integrate over logMh' for each logMh
        pgg_cc = []
        for i in range(self.config['logMh'].size):
            pgg_cc.append(simpson(dndm_vals * Ncen_vals * phh_m1m2_mat[i], x=self.config['logMh'], axis=0))
        pgg_cc = np.array(pgg_cc)

        # Second integral: integrate over logMh
        pgg_cc = simpson(dndm_vals * Ncen_vals * pgg_cc, x=self.config['logMh'], axis=0)

        # Apply the galaxy-density normalisation
        pgg_cc = (1.0 / self.get_ng()**2) * pgg_cc
        #spl = UnivariateSpline(k, pgg_cc, s=self.config['smooth_phh_k'], k=3)
        #pgg_cc = spl(k)
        return k, pgg_cc

    def _xigg_2h_cc(self):
        """
        Get the 2-halo term for the correlation function
        """
        k, pgg_2h = self._pgg_2h_cc()
        # Convert the power spectrum to the correlation function
        r, xigg_2h = mcfit.P2xi(k, l=0, lowring=True)(pgg_2h, extrap=True)
        # Interpolate the correlation function
        return r, xigg_2h

    def _pgg_2h_cs(self):
        """
        Get the 2-halo term for the central-satellite correlation function 
        TODO: Implement the sattelite displacement kernel
        """
        # get the p_hh on (M,M') grid
        k, phh_m1m2_mat = self.get_phh_m1m2_matrix_reversed()
        # Set up the mass-dependent arrays
        # dndm_vals: dn/dlog10M replicated along k
        dndm_vals = np.tile(self.dndlog_m(self.config['logMh']), (k.size, 1)).T
        Ncen_vals = np.tile(self._Ncen(self.config['logMh']), (k.size, 1)).T
        Nsat_vals = np.tile(self._Nsat(self.config['logMh']), (k.size, 1)).T
        # First integral: integrate over logMh' for each logMh
        pgg_cs = []
        for i in range(self.config['logMh'].size):
            pgg_cs.append(simpson(dndm_vals * Nsat_vals * phh_m1m2_mat[i], x=self.config['logMh'], axis=0))
        pgg_cs = np.array(pgg_cs)
        # Second integral: integrate over logMh
        pgg_cs = simpson(dndm_vals * Ncen_vals * pgg_cs, x=self.config['logMh'], axis=0)
        # Apply the galaxy-density normalisation
        pgg_cs = (1.0 / self.get_ng()**2) * pgg_cs
        #spl = UnivariateSpline(k, pgg_cs, s=self.config['smooth_phh_k'], k=3)
        #pgg_cs = spl(k)
        return k, 2*pgg_cs

    def _xigg_2h_cs(self):
        """
        Get the 2-halo term for the correlation function
        """
        k, pgg_2h = self._pgg_2h_cs()
        # Convert the power spectrum to the correlation function
        r, xigg_2h = mcfit.P2xi(k, l=0, lowring=True)(pgg_2h, extrap=True)
        # Interpolate the correlation function
        return r, xigg_2h

    def _pgg_2h_ss(self):
        """
        Get the 2-halo term for the satellite-satellite correlation function
        """
        # get the p_hh on (M,M') grid
        k, phh_m1m2_mat = self.get_phh_m1m2_matrix_reversed()
        # Set up the mass-dependent arrays
        # dndm_vals: dn/dlog10M replicated along k
        dndm_vals = np.tile(self.dndlog_m(self.config['logMh']), (k.size, 1)).T
        Nsat_vals = np.tile(self._Nsat(self.config['logMh']), (k.size, 1)).T
        # First integral: integrate over logMh' for each logMh
        pgg_ss = []
        for i in range(self.config['logMh'].size):
            pgg_ss.append(simpson(dndm_vals * Nsat_vals * phh_m1m2_mat[i], x=self.config['logMh'], axis=0))
        pgg_ss = np.array(pgg_ss)
        # Second integral: integrate over logMh
        pgg_ss = simpson(dndm_vals * Nsat_vals * pgg_ss, x=self.config['logMh'], axis=0)
        # Apply the galaxy-density normalisation
        pgg_ss = (1.0 / self.get_ng()**2) * pgg_ss
        #spl = UnivariateSpline(k, pgg_ss, s=self.config['smooth_phh_k'], k=3)
        #pgg_ss = spl(k)
        return k, pgg_ss

    def _xigg_2h_ss(self):
        """
        Get the 2-halo term for the correlation function
        """
        k, pgg_2h = self._pgg_2h_ss()
        # Convert the power spectrum to the correlation function
        r, xigg_2h = mcfit.P2xi(k, l=0, lowring=True)(pgg_2h, extrap=True)
        # Interpolate the correlation function
        return r, xigg_2h

    def _pgg_1h(self):
        """
        Get the 1-halo term for the spherically averaged power spectrum
        """
        dndm_vals = np.tile(self.dndlog_m(self.config['logMh']), (self.k.size, 1)).T
        Ncen_vals = np.tile(self._Ncen(self.config['logMh']), (self.k.size, 1)).T
        lam_s_vals = np.tile(self._lambda_sat(self.config['logMh']), (self.k.size, 1)).T
        # Sptial distribution of satellite galaxies
        pgg_1h = simpson(dndm_vals * Ncen_vals * lam_s_vals * self.uk * (2 + lam_s_vals*self.uk), x=self.config['logMh'], axis=0)
        # Apply the galaxy-density normalisation
        pgg_1h = (1.0 / self.get_ng()**2) * pgg_1h
        return self.k, pgg_1h
    
    def _xigg_1h(self):
        """
        Get the 1-halo term for the correlation function
        """
        _, pgg_1h = self._pgg_1h()
        # Convert the power spectrum to the correlation function
        r, xigg_1h = mcfit.P2xi(self.k, l=0, lowring=True)(pgg_1h, extrap=True)
        # Interpolate the correlation function
        return r, xigg_1h

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
        # Get the 1-halo term
        #_, pgg_1h = self._pgg_1h()
        # Get the 2-halo term for the central-central correlation function
        k, pgg_2h_cc = self._pgg_2h_cc()
        # Get the 2-halo term for the central-satellite correlation function
        _, pgg_2h_cs = self._pgg_2h_cs()
        # Get the 2-halo term for the satellite-satellite correlation function
        _, pgg_2h_ss = self._pgg_2h_ss()

        # Combine all terms to get the total power spectrum P_gg(k)
        #pgg = pgg_2h_cc + pgg_2h_cs + pgg_2h_ss + pgg_1h
        #FIXME: Add the 1-hal oterm
        pgg = pgg_2h_cc + pgg_2h_cs + pgg_2h_ss# + pgg_1h

        return k, pgg
    
    def get_xi_gg(self):
        """
        Compute the galaxy correlation function xi(r) for the current cosmology and HOD parameters.

        This applies the HOD to the emulated xi_hh(M, M').

        The computation flow is:
        1. Get dndm for self.Mh.
        2. Get xi_hh(M, M') on the (self.Mh, self.Mh) grid.
        3. Compute <N_cen>(M) and <N_sat>(M) for self.Mh.
        4. Compute the displacement kernels H_off(k,M) and u_s(k,M).
        5. Integrate the terms P_1c(k), P_2c(k), P_1s(k), and P_2s(k).

        Returns
        -------
        xi_gg : np.ndarray
            Galaxy correlation function xi(r).
        """
        k, pgg = self.get_pk_gg()
        # Convert the power spectrum to the correlation function
        r, xigg = mcfit.P2xi(k, l=0, lowring=True)(pgg, extrap=True)

        return r, xigg

    def get_wp_gg(self, pi_max = 40):
        """
        Get the projected correlation function w_p(r) for the current cosmology and HOD parameters.
        """
        rbins, xigg = self.get_xi_gg()
        # Build an interpolator for the correlation function
        xi_spl = UnivariateSpline(rbins, xigg, s=self.config['smooth_xihh_r'], k=3)
        # Compute the projected correlation function w_p(r)
        wp = np.zeros_like(rbins)
        for i, r in enumerate(rbins):
            integrand = lambda x: xi_spl(np.sqrt(x**2 + r**2))
            wp[i] = quad(integrand, 0, pi_max)[0]
        wp = 2 * wp
        return rbins, wp




        