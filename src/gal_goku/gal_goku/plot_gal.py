"""
Plotting routines for the Galaxy emualtor
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from gal_goku import gal
import logging
from colossus.cosmology import cosmology as col_cosmology
from colossus.lss import mass_function as col_mass_function

class PlotGal():
    """
    Class to plot the Galaxy emulator
    """

    def __init__(self, logging_level='INFO'):
        """
        Plotting routines for the Galaxy X Galaxy clustering emulator
        Parameters
        ----------
        logging_level : str
            The logging level to use. Options are 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'.
        """

        self.logger = self.configure_logging(logging_level)
        self.latex_labels = {'omega0': r'$\Omega_m$', 'omegab': r'$\Omega_b$', 
            'hubble': r'$h$', 'scalar_amp': r'$A_s$', 'ns': r'$n_s$', 
            'w0_fld': r'$w_0$', 'wa_fld': r'$w_a$', 'N_ur': r'$N_{ur}$', 
            'alpha_s': r'$\alpha_s$', 'm_nu': r'$m_{\nu}$'}
        
        self.params=['omega0', 'omegab', 'hubble', 
                     'scalar_amp', 'ns', 'w0_fld', 
                     'wa_fld', 'N_ur', 'alpha_s', 
                     'm_nu']
        
        config = {'logMh': np.arange(11.1, 12.5, 0.05),
          'smooth_xihh_r': 0,
          'smooth_phh_k': 0,
          'smooth_xihh_mass': 0,
          'r_range': [0.1, 50]}
                
        self.g = gal.Gal(logging_level=logging_level, config=config)
        self.g.reset_hod()
        self.cosmo_mid = self.g.xi_emu.cosmo_min + (self.g.xi_emu.cosmo_max - self.g.xi_emu.cosmo_min)/2


    def configure_logging(self, logging_level):
        """Sets up logging based on the provided logging level."""
        logger = logging.getLogger('ProjCorrEmus')
        logger.setLevel(logging_level)
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        return logger

    def xi_gg_cosmo_sensitivity(self):
        """
        Plot the sensitivity of the emulator to cosmological parameters
        """
        fig, ax = plt.subplots(10, 2, figsize=(10, 40))
        for i in range(10):
            self.logger.info(f'Plotting {self.params[i]}')

            # The 25th and 75th percentiles of the cosmological prior range
            cosmo_lower = np.copy(self.cosmo_mid)
            cosmo_lower[i] = self.cosmo_mid[i] - np.abs(self.g.xi_emu.cosmo_max[i] - self.g.xi_emu.cosmo_min[i])/4
            cosmo_upper = np.copy(self.cosmo_mid)
            cosmo_upper[i] = self.cosmo_mid[i] + np.abs(self.g.xi_emu.cosmo_max[i] - self.g.xi_emu.cosmo_min[i])/4

            # Get the xi for the lower and upper bounds
            self.g.reset_cosmo(cosmo_lower)
            r, xi_lower = self.g.get_xi_gg()
            self.g.reset_cosmo(cosmo_upper)
            r, xi_upper = self.g.get_xi_gg()
            
            # Plot the original xi for the lower and upper bounds in the first column
            ax[i, 0].plot(r, xi_lower, label=rf'{self.latex_labels[self.params[i]]} = {cosmo_lower[i]}')
            ax[i, 0].plot(r, xi_upper, label=rf'{self.latex_labels[self.params[i]]} = {cosmo_upper[i]}', ls='--')
            
            ax[i, 0].set_ylabel(r'$\xi_{gg}(r)$')
            ax[i, 0].set_xscale('log')
            ax[i, 0].set_yscale('log')
            ax[i, 0].legend()
            ax[i, 0].grid(which='both', linestyle='--', linewidth=0.5)

            # Plot the ratio of xi_upper to xi_lower in the second column
            frac = xi_upper / xi_lower - 1
            ax[i, 1].plot(r, frac, label='Ratio')
            
            ax[i, 1].set_ylabel(r'$\xi_{gg,\ \mathrm{upper}} / \xi_{gg,\ \mathrm{lower}} - 1$')
            ax[i, 1].set_xscale('log')
            ax[i, 1].grid(which='both', linestyle='--', linewidth=0.5)
            ax[i, 1].legend()
        fig.tight_layout()

    def pk_gg_cosmo_sensitivity(self):
        """
        Plot the sensitivity of the powerspectrum emulator to cosmological parameters
        """
        fig, ax = plt.subplots(10, 2, figsize=(10, 40))
        for i in range(10):
            self.logger.info(f'Plotting {self.params[i]}')

            # The 25th and 75th percentiles of the cosmological prior range
            cosmo_lower = np.copy(self.cosmo_mid)
            cosmo_lower[i] = self.cosmo_mid[i] - np.abs(self.g.xi_emu.cosmo_max[i] - self.g.xi_emu.cosmo_min[i])/4
            cosmo_upper = np.copy(self.cosmo_mid)
            cosmo_upper[i] = self.cosmo_mid[i] + np.abs(self.g.xi_emu.cosmo_max[i] - self.g.xi_emu.cosmo_min[i])/4

            # Get the xi for the lower and upper bounds
            self.g.reset_cosmo(cosmo_lower)
            k, pk_lower = self.g.get_pk_gg()
            self.g.reset_cosmo(cosmo_upper)
            k, pk_upper = self.g.get_pk_gg()
            
            # Plot the original xi for the lower and upper bounds in the first column
            ax[i, 0].plot(k, pk_lower, label=rf'{self.latex_labels[self.params[i]]} = {cosmo_lower[i]}')
            ax[i, 0].plot(k, pk_upper, label=rf'{self.latex_labels[self.params[i]]} = {cosmo_upper[i]}', ls='--')
            
            ax[i, 0].set_ylabel(r'$\xi_{gg}(r)$')
            ax[i, 0].set_xscale('log')
            ax[i, 0].set_yscale('log')
            ax[i, 0].legend()
            ax[i, 0].grid(which='both', linestyle='--', linewidth=0.5)

            # Plot the ratio of xi_upper to xi_lower in the second column
            frac = pk_upper / pk_lower - 1
            ax[i, 1].plot(k, frac, label='Ratio')
            
            ax[i, 1].set_ylabel(r'$P_{gg,\ \mathrm{upper}} / P_{gg,\ \mathrm{lower}} - 1$')
            ax[i, 1].set_xscale('log')
            ax[i, 1].grid(which='both', linestyle='--', linewidth=0.5)
            ax[i, 1].legend()
        fig.tight_layout()


    def dndm_planck18(self):
        """
        Plot the predicted halo mass function dndm for the Planck18 cosmology.
        """
        from astropy.cosmology import Planck18
        from astropy.cosmology import LambdaCDM
        from astropy import units as u
        import camb
        fig, ax = plt.subplots(figsize=(8, 6))

        # Planck18 parameters with ns explicitly defined
        cosmo_planck18 = np.array([
            Planck18.Om0,                # omega0
            Planck18.Ob0,                # omegab
            Planck18.h,                  # hubble
            2.1e-9,                      # scalar_amp (approx. Planck18 value)
            0.9649,                      # ns, Planck18 best-fit value from Planck Collaboration 2018 (Table 2)
            -1.0,                        # w0_fld
            0.0,                         # wa_fld
            Planck18.Neff,               # N_ur
            0.0,                         # alpha_s
            np.sum(Planck18.m_nu.value)  # m_nu
        ])

        # Reset cosmology to Planck18 parameters
        self.g.reset_cosmo(cosmo_planck18)
        # Finer steps than the default config['logMh']
        logMh = np.arange(11.1, 12.5, 0.01)
        dndlogm_emu = self.g.dndlog_m(logMh)

        # Set cosmology to Planck18
        # Use the cosmo_planck18 array we defined above
        # Compute sigma8 from scalar_amp (A_s), ns, and other cosmological parameters

        # Unpack parameters
        Om0 = cosmo_planck18[0]
        Ob0 = cosmo_planck18[1]
        h = cosmo_planck18[2]
        A_s = cosmo_planck18[3]
        ns = cosmo_planck18[4]
        nnu = cosmo_planck18[7]  # Number of massless neutrinos
        mnu = cosmo_planck18[9]


        # Use CAMB to compute sigma8
        pars = camb.CAMBparams()
        pars.set_cosmology(H0=h*100, ombh2=Ob0*h**2, omch2=(Om0-Ob0)*h**2, 
                           nnu=nnu, mnu=mnu)
        pars.InitPower.set_params(As=A_s, ns=ns)
        pars.set_matter_power(redshifts=[0.0], kmax=2.0)
        results = camb.get_results(pars)
        sigma8 = results.get_sigma8()

        params = {
            'flat': True,
            'H0': h * 100,  # hubble param in km/s/Mpc
            'Om0': Om0,
            'Ob0': Ob0,
            'sigma8': sigma8,
            'ns': ns
        }
        col_cosmology.setCosmology('Our planck18', params)

        # Define mass range and redshift
        m = np.logspace(np.log10(5e10), 12.5, 100)
        z = 2.5

        # Compute halo mass function (dn/dlogM)
        h =col_mass_function.massFunction(m, z, mdef = 'fof', model = 'sheth99', 
                                                q_out = 'dndlnM')
        # Convert dn/dln(M) to dn/dlog10(M): dn/dllog10(M) = dn/dln(M) * log(10)
        h *= np.log(10)
        # Plot the Colossus mass function on the same ax
        ax.plot(m, h, label='Linear Theory approximation', lw=5, alpha=1, ls='--')
        ax.plot(10**logMh, dndlogm_emu, label='Current Emulator', lw=5, alpha=1)
        ax.set_xlabel(r'$\log_{10}(M_h/M_\odot)$')
        
        ax.set_ylabel(r'$dn/dM$ [$(h^3/Mpc^3)/M_\odot$]')
        #ax.set_title('Halo Mass Function dndm (Planck18 Cosmology)')
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.legend(fontsize=18, frameon=False, loc='upper right')
        fig.tight_layout()



