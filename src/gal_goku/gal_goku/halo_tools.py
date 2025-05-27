"""
Tools for standard cosmological calculations.
Mostly taking care of the halo calculations, 
like NFW profile, concentration, etc.
"""
import numpy as np
from mcfit import Hankel
from colossus.cosmology import cosmology as colossus_cosmology
from colossus.halo import concentration
from scipy import special
from colossus.halo import mass_defs
import camb
import logging, sys

class HaloTools:
    """
    Class for cosmological calculations.
    """

    def __init__(self, cosmo_pars, z):
        """
        Initialize the CosmoTools class.

        Parameters
        ----------
        cosmo_pars : np.ndarray, shape (10,), optional
            Cosmology parameters in the order:
            'omega0', 'omegab', 'hubble', 'scalar_amp', 'ns',
            'w0_fld', 'wa_fld', 'N_ur', 'alpha_s', 'm_nu'.
        """
        self.logger = self.configure_logging()
        self.cosmo_pars = cosmo_pars
        self.z = z
        self.sigma_8 = self.get_sigma8()
        self.col_cosmo = self.set_colossus_cosmo()
        
        pass

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
    
    def set_colossus_cosmo(self):
        """
        Set up the cosmology for halo calculations using Colossus.
        """
        params = {'flat': True, 
                  'H0': self.cosmo_pars[2]*100, 
                  'Om0': self.cosmo_pars[0], 
                  'Ob0': self.cosmo_pars[1], 
                  'sigma8': self.sigma_8, 
                  'ns': self.cosmo_pars[4],
                  'de_model': 'w0wa',
                  'w0': self.cosmo_pars[5],
                  'wa': self.cosmo_pars[6],
                  'relspecies': True,
                  'Neff': self.cosmo_pars[7]
                  }
        cosmo = colossus_cosmology.setCosmology('myCosmo', **params)
        return cosmo

    def get_sigma8(self):
        """"
        Get the sigma8 value from the cosmology usin CAMB.

        Returns
        -------
        float
            The sigma8 value.
        """
        pars = camb.CAMBparams()
        pars.set_cosmology(H0=self.cosmo_pars[2]*100,  # Hubble parameter in km/s/Mpc
                           ombh2=self.cosmo_pars[1] * self.cosmo_pars[2]**2,  # Baryon density
                           omch2=self.cosmo_pars[0] * self.cosmo_pars[2]**2,  # Cold dark matter density
                           omk=0,
                           #nu_mass_numbers = self.cosmo_pars[7],
                           mnu=self.cosmo_pars[9])
        
        pars.InitPower.set_params(ns=self.cosmo_pars[4],
                                  As=self.cosmo_pars[3])
        
        pars.set_matter_power(redshifts=[self.z], kmax=2.0)
        results = camb.get_results(pars)
        sigma8 = results.get_sigma8()
        return sigma8

    def analytic_uk(self, k, c, rvir):
        """
        Compute the analytic Fourier transform of the NFW profile.
        Supports broadcasting over arrays of c and rvir.
        Parameters
        ----------
        k : array_like
            Wavenumber(s) [h/Mpc]
        c : array_like
            Concentration(s)
        rvir : array_like
            Virial radius/radii [Mpc/h]
        Returns
        -------
        uk : ndarray
            Fourier transform of the NFW profile (broadcasted over c, rvir, and k)
        """
        k = np.atleast_1d(k)
        c = np.atleast_1d(c)
        rvir = np.atleast_1d(rvir)

        rs = rvir / c  # shape (n_halo,)
        eta = np.outer(rs, k)  # shape (n_halo, n_k)
        f = 1.0 / (np.log(1 + c) - c / (1 + c))[:, np.newaxis]  # shape (n_halo, 1)

        si_1pc, ci_1pc = special.sici((1 + c)[:, np.newaxis] * eta)
        si, ci = special.sici(eta)

        uk = f * (
            np.sin(eta) * (si_1pc - si) +
            np.cos(eta) * (ci_1pc - ci) -
            np.sin(c[:, np.newaxis] * eta) / ((1 + c)[:, np.newaxis] * eta)
        )
        return uk  # shape (n_halo, n_k)

    def get_halo_props_colossus(self, m_vir, z):
        """
        Get halo properties using Colossus.

        Parameters
        ----------
        mvir : float
            The virial mass of the halo
        z : float
        """
        # gEt the concentration parameter
        c_vir = concentration.concentration(M=m_vir, z=z, mdef='vir')
        # Get the virial radius
        _, r_vir, _ = mass_defs.changeMassDefinition(m_vir, c_vir, z, 'vir', 'vir')
        return c_vir, r_vir

    def get_uk(self, k, m_vir):
        """
        Get the Fourier transform of the NFW profile.
        Parameters
        ----------
        k : np.ndarray, in units of h/Mpc
        m_vir : float, in units of Msun/h
            The virial mass of the halo
        Returns
        -------
        np.ndarray
            The Fourier transform of the NFW profile.

        """
        c_vir, r_vir = self.get_halo_props_colossus(m_vir, self.z)
        r_vir /= 1000  # Convert to Mpc/h
        self.logger.debug(f"For log_m_vir: {np.log10(m_vir)} M_sol/h, c_vir: {c_vir}, r_vir: {r_vir} Mpc/h")
        uk = self.analytic_uk(k, c_vir, r_vir)
        return uk