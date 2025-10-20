"""
Tools for standard cosmological calculations,
using Colossus and CAMB.
Mostly taking care of the halo calculations, 
like NFW profile, concentration, etc.
"""
import numpy as np
from mcfit import Hankel
from colossus.cosmology import cosmology as colossus_cosmology
from colossus.halo import concentration
from scipy import special
from scipy import integrate
from scipy.constants import G
from scipy.interpolate import UnivariateSpline
#from scipy.integrate import simps
#from colossus.halo import mass_defs
from astropy import units as u
import camb
from . import init_power
import logging, sys

class HaloTools:
    """
    Class for cosmological calculations.
    """

    def __init__(self, cosmo_pars, z=2.5):
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
        #self.sigma_8 = self.get_sigma8()
        #self.col_cosmo = self.set_colossus_cosmo()

        # Pass cosmo parameters to CAMB
        self.z = z
        self.camb_pars, self.camb_res = self.set_camb_cosmo(all_zs=[self.z])
        self.rho_c, self.rho_m = self.get_rho_cm(self.z)

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

    def set_camb_cosmo(self, all_zs=[2.5]):
        """
        Set up the cosmology for halo calculations using CAMB.
        """
        pars = camb.CAMBparams()
        pars.set_cosmology(H0=self.cosmo_pars[2]*100,  # Hubble parameter in km/s/Mpc
                           ombh2=self.cosmo_pars[1] * self.cosmo_pars[2]**2,  # Baryon density
                           omch2=(self.cosmo_pars[0] - self.cosmo_pars[1]) * self.cosmo_pars[2]**2,  # Cold dark matter density
                           omk=0,
                           nnu= self.cosmo_pars[7],
                           mnu=self.cosmo_pars[9])
        pars.set_dark_energy(w=self.cosmo_pars[5], wa=self.cosmo_pars[6])
        pars.InitPower.set_params(ns=self.cosmo_pars[4],
                                  As=self.cosmo_pars[3],
                                  nrun=self.cosmo_pars[8])
        pars.set_matter_power(redshifts=all_zs, kmax=100.0)
        results = camb.get_results(pars)
        return pars, results

    def get_power_camb(self, k):
        """
        Get the matter power spectrum from CAMB for given wavenumbers.
        Parameters
        ----------
        k : array_like
            Wavenumber(s) [h/Mpc]
        Returns
        -------
        k : array_like
            Wavenumber(s) [h/Mpc]
        pk : array_like
            Power spectrum values at the given wavenumbers [Mpc/h]^3
        """
        k, z, pk = self.camb_res.get_matter_power_spectrum(minkh=min(k), maxkh=max(k), npoints=len(k))
        return k, pk
     
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
        pars.set_matter_power(redshifts=[self.z], kmax=5.0)
        results = camb.get_results(pars)
        sigma8 = results.get_sigma8()
        return sigma8
    


    def get_zeldovich_displacement_camb_power(self):
        """
        Calculate the Zeldovich displacement for the cosmology and redshift
        set in the constructor.
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
        k = np.logspace(-4, 1, 500)  # You can increase resolution
        PK = camb.get_matter_power_interpolator(pars, hubble_units=True, k_hunit=True, kmax=10.0, nonlinear=False)
        P_lin = PK.P(self.z, k)  
        sigma_d_squared = (1.0 / (6.0 * np.pi**2)) * simps(P_lin, k)
        sigma_d = np.sqrt(sigma_d_squared)
        return sigma_d
    
    def get_zeldovich_displacement(self, k, pk):
        """
        Calculate the Zeldovich displacement for a given power spectrum.
        Parameters
        ----------
        k : array_like
            Wavenumber(s) [h/Mpc]
        pk : array_like
            Power spectrum values at the given wavenumbers [Mpc/h]^3
        Returns
        -------
        sigma_d : float
            The Zeldovich displacement in Mpc/h.
        """
        k = np.atleast_1d(k)
        pk = np.atleast_1d(pk)

        # Ensure k and pk are 1D arrays
        if k.ndim > 1 or pk.ndim > 1:
            raise ValueError("k and pk must be 1D arrays.")

        # Calculate the Zeldovich displacement
        sigma_d_squared = (1.0 / (6.0 * np.pi**2)) * simps(pk, k)
        sigma_d = np.sqrt(sigma_d_squared)
        return sigma_d

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

        # Use a fixed range of k to ensure the Hankel transform is
        # well-converged on small and large scales.
        k_long_support = np.linspace(0.05, 30, 200)
        uk = self.analytic_uk(k_long_support, c_vir, r_vir)
        # Interpolate the result to the desired k values
        uk_interp = []
        for i in range(uk.shape[0]):
            uk_interp.append(UnivariateSpline(k_long_support, uk[i], k=3)(k))
        uk_interp = np.array(uk_interp)
        return uk_interp

    def get_rho_cm(self, z):
        """
        Get the critical and mean matter density at redshift z in units of Msun/(Mpc/h)^3.

        Parameters
        ----------
        z : float
            The redshift.

        Returns
        -------
        float
            The critical density at redshift z.
        """
        Hz = self.camb_res.hubble_parameter(z)  # In km/s/Mpc
        H_si = Hz * (u.km / u.s / u.Mpc).to(u.Hz)  # Convert H(z) to 1/s
        rho_c_si = 3 * H_si**2 / (8 * np.pi * G)  # In kg/m^3
        # Convert to Msun / Mpc^3
        kg_to_msun = (u.kg).to(u.Msun)
        m_to_mpc = (u.m).to(u.Mpc)

        rho_c = rho_c_si * (kg_to_msun / m_to_mpc**3)  # Msun / Mpc^3

        Omz = self.camb_res.get_Omega('cdm', z) + self.camb_res.get_Omega('baryon', z)
        rho_m = Omz * rho_c  # Msun / Mpc^3
        return rho_c, rho_m

    def get_sigma_m(self, mass, delta=200):
        """
        Get the mass variance sigma(M) by intergrating the power spectrum.

        Parameters
        ----------
        masses : float or np.ndarray
            The mass or masses in Msun/h.

        Returns
        -------
        float or np.ndarray
            The mass variance sigma(M).
        """
        k = np.logspace(-4, 2, 500)
        k, pk = self.get_power_camb(k)
        k = k.squeeze()
        pk = pk.squeeze()
        rho_ref = delta * self.rho_c  # in Msun/(Mpc/h)^3
        r = (3 * mass / (4 * np.pi * rho_ref))**(1/3)
        def w(kr):
            return (3 * (np.sin(kr) - kr * np.cos(kr))) / (kr**3 + 1e-10)
        integrand = lambda lnk: np.interp(np.exp(lnk), k, pk) * w(np.exp(lnk) * r)**2 * np.exp(3 * lnk)
        result = integrate.quad(integrand, np.log(k[0]), np.log(k[-1]), limit=200)[0]
        sigma_m = np.sqrt(result / (2 * np.pi**2))
        return sigma_m

    def tinker_hmf(self, A=0.186, a=1.47, b=2.57, c=1.19, z=2.5, delta=200, mbins=None):
        """
        Get the Tinker et al. 2008 halo mass function.

        Parameters
        ----------
        mass : float or np.ndarray
            The mass or masses in Msun/h.

        Returns
        -------
        float or np.ndarray
            The halo mass function dn/dlnM in (Mpc/h)^-3.
        """
        if mbins is None:
            mbins = np.logspace(11, 13.5, 25)
        sigma_ms = []
        for m in mbins:
            sigma_ms.append(self.get_sigma_m(m, delta=delta))
        sigma_ms = np.array(sigma_ms)
        def f_sigma(sigma):
            return A * ((sigma / b)**-a + 1) * np.exp(-c / sigma**2)
        dlnsigma_dlnM = np.gradient(np.log(sigma_ms), np.log(mbins))
        dndlog10M = (self.rho_m / mbins) * f_sigma(sigma_ms) * np.abs(dlnsigma_dlnM) * np.log(10)
        return dndlog10M