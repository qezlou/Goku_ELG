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

        # Define the refrence cosmology to get the ratio to
        self.cosmo_ref = [0.31, 0.048, 0.68, 
                          2.1e-9, 0.97, -1,    
                          0,   3.08, 0, 
                          0.1
                          ]
        # Interesting results from DESI etc. to also plot
        
        cosmo_bounds = [[0.053, 0.193], # m_nu: arxiv:2503.14744 
                                        #The upper 95th bound, one from DESI-DR1 BAO 
                                        # and DESI-DR1BAO+Full-shape+BAO
                        ]
        # The parameters to plot the sensitivity for
        self.plot_range= {}
        for i, param in enumerate(self.params):
            if param == 'w0_fld':
                # For w0_fld, we use a range from -1.15 to -0.5
                start = -1.15
                end = 0
            else:
                # For all other parameters, we use the range from the emulator
                start = self.g.xi_emu.cosmo_min[i]
                end = self.g.xi_emu.cosmo_max[i]
            param_range = end - start
            self.plot_range[param] = [start + f * param_range for f in np.linspace(0.1, 0.9, 9)]

    def configure_logging(self, logging_level):
        """Sets up logging based on the provided logging level."""
        logger = logging.getLogger('ProjCorrEmus')
        logger.setLevel(logging_level)
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        return logger

    def xi_gg_cosmo_sensitivity(self, savefig_dir=None):
        """
        Plot the sensitivity of the galaxy correlation function emulator to cosmological parameters.
        Uses 10 percentiles (10% to 90%) for each parameter, ratio curves only, compact 5x2 layout,
        viridis colormap, LaTeX font, and publication-style formatting.
        """
        # Set global matplotlib style for publication
        plt.rcParams.update({
            'font.size': 10,
            'font.family': 'serif',
            'text.usetex': True,
            'axes.labelsize': 12,
            'axes.titlesize': 12,
            'legend.fontsize': 8
        })

        fig, ax = plt.subplots(5, 2, figsize=(10, 10), sharex=True)
        ax = ax.reshape(5, 2)
        cmap = plt.cm.viridis
        colors = [cmap(j) for j in np.linspace(0.1, 0.9, 9)]

        # get the reference correlation function, i.e. Planck18 cosmology
        self.g.reset_cosmo(np.copy(self.cosmo_ref))
        rvals, ref = self.g.get_xi_gg()

        for i in range(10):
            r = i // 2
            c = i % 2
            self.logger.info(f'Plotting {self.params[i]}')
            xi_curves = []
            for val in self.plot_range[self.params[i]]:
                cosmo_tmp = np.copy(self.cosmo_ref)
                cosmo_tmp[i] = val
                self.g.reset_cosmo(cosmo_tmp)
                rvals, xi = self.g.get_xi_gg()
                xi_curves.append(xi)
            # We don't use the median comosology as the reference here anymore
            #ref = xi_curves[len(xi_curves)//2]
            
            for xi, color, val in zip(xi_curves, colors, self.plot_range[self.params[i]]):
                label = self._set_param_label(i, val)
                ax[r, c].plot(rvals, xi / ref - 1, label=label, color=color, lw=1.5)
            # Only show y-labels on first column
            if i % 2 == 0:
                ax[r, c].set_ylabel(r'$\xi_{gg} / \xi_{gg,\mathrm{med}} - 1$')
            # Only show x-labels on last row
            if r == 4:
                ax[r, c].set_xlabel(r'$r\ (h^{-1}\mathrm{Mpc})$')
            ax[r, c].set_xscale('log')
            ax[r, c].grid(which='both', linestyle='--', linewidth=0.5)
            # Place legend outside right of subplot for clarity
            ax[r, c].legend(fontsize=8, loc='center left', bbox_to_anchor=(1.05, 0.5))
        fig.subplots_adjust(hspace=0.4, wspace=0.2, right=0.82)
        fig.suptitle(r'$\xi_{gg}$ Sensitivity to Cosmological Parameters', fontsize=14, y=1.02)
        fig.tight_layout(rect=[0, 0, 0.8, 1])
        if savefig_dir is not None:
            fig.savefig(os.path.join(savefig_dir, 'xi_gg_cosmo_sensitivity.pdf'))

    def pk_gg_cosmo_sensitivity(self, savefig_dir=None):
        """
        Plot the sensitivity of the power spectrum emulator to cosmological parameters
        using ratio plots and compact layout for paper-quality figures.
        """
        # Set global matplotlib style for publication
        plt.rcParams.update({
            'font.size': 10,
            'font.family': 'serif',
            'text.usetex': True,
            'axes.labelsize': 12,
            'axes.titlesize': 12,
            'legend.fontsize': 8
        })

        fig, ax = plt.subplots(5, 2, figsize=(10, 10), sharex=True)
        ax = ax.reshape(5, 2)
        cmap = plt.cm.viridis
        colors = [cmap(j) for j in np.linspace(0.1, 0.9, 9)]

        # get the reference power spectrum, i.e. Planck18 cosmology
        self.g.reset_cosmo(np.copy(self.cosmo_ref))
        k, ref = self.g.get_pk_gg()

        for i in range(10):
            r = i // 2
            c = i % 2
            self.logger.info(f'Plotting {self.params[i]}')

            pk_curves = []
            for val in self.plot_range[self.params[i]]:
                cosmo_tmp = np.copy(self.cosmo_ref)
                cosmo_tmp[i] = val
                self.g.reset_cosmo(cosmo_tmp)
                k, pk = self.g.get_pk_gg()
                pk_curves.append(pk)

            # We don't use the median comosology as the reference here anymore
            #ref = pk_curves[len(pk_curves)//2]

            for pk, color, val in zip(pk_curves, colors, self.plot_range[self.params[i]]):
                label = self._set_param_label(i, val)
                ax[r, c].plot(k, pk / ref - 1, label=label, color=color, lw=1.5)

            if i % 2 == 0:
                ax[r, c].set_ylabel(r'$P_{gg} / P_{gg,\mathrm{med}} - 1$')
            if r == 4:
                ax[r, c].set_xlabel(r'$k\ (h/\mathrm{Mpc})$')

            ax[r, c].set_xscale('log')
            ax[r, c].grid(which='both', linestyle='--', linewidth=0.5)
            ax[r, c].legend(fontsize=8, loc='center left', bbox_to_anchor=(1.05, 0.5))

        fig.subplots_adjust(hspace=0.4, wspace=0.25, right=0.82)
        fig.suptitle('P$_{gg}$ Sensitivity to Cosmological Parameters', fontsize=14, y=1.02)
        fig.tight_layout(rect=[0, 0, 0.8, 1])
        if savefig_dir is not None:
            fig.savefig(os.path.join(savefig_dir, 'pk_gg_cosmo_sensitivity.pdf'))

    def _set_param_label(self, i, val):
        """
        Generate sting for curve labels on the legend
        """
        latex = self.latex_labels[self.params[i]]
        if self.params[i] == 'scalar_amp':
            val *= 1e9
            label = rf'{latex} = {val:.3f} e9'
        else:
            label = rf'{latex} = {val:.3f}'
        return label
    
    def hmf_cosmo_sensitivity(self, savefig_dir=None):
        """
        Plot the sensitivity of the halo mass function emulator to cosmological parameters
        """
        # Set global matplotlib style for publication
        # This ensures consistent and professional plot appearance
        plt.rcParams.update({
            'font.size': 10,
            'font.family': 'serif',
            'text.usetex': True,
            'axes.labelsize': 12,
            'axes.titlesize': 12,
            'legend.fontsize': 8
        })
        # Set up subplot layout: 5 rows x 2 columns for 10 cosmology parameters
        fig, ax = plt.subplots(5, 2, figsize=(10, 10), sharex=True)
        ax = ax.reshape(5, 2)  # Ensure 2D shape for easy indexing
        # Set up colormap for different parameter samples
        cmap = plt.cm.viridis
        # 10 colors for 10 percentiles (from 10% to 90%)
        colors = [cmap(j) for j in np.linspace(0.1, 0.9, 9)]

         # Evaluate HMF over a fine grid of halo masses
        logMh = np.arange(11.1, 12.5, 0.01)
        # get the reference halo mass function, i.e. Planck18 cosmology
        self.g.reset_cosmo(np.copy(self.cosmo_ref))       
        ref = self.g.dndlog_m(logMh)
        for i in range(10):
            # Determine subplot row and column
            r = i // 2
            c = i % 2
            self.logger.info(f'Plotting {self.params[i]}')
            dndlogm_curves = []
            for val in self.plot_range[self.params[i]]:
                # For each sampled parameter value, update cosmology and compute HMF
                cosmo_tmp = np.copy(self.cosmo_ref)
                cosmo_tmp[i] = val
                self.g.reset_cosmo(cosmo_tmp)
                dndlogm_curves.append(self.g.dndlog_m(logMh))
            # We don't use the median comosology as the reference here anymore
            # ref = dndlogm_curves[len(dndlogm_curves)//2]

            for dndlogm, color, val in zip(dndlogm_curves, colors, self.plot_range[self.params[i]]):
                # Plot fractional difference relative to the median curve
                label = self._set_param_label(i, val)
                ax[r, c].plot(10**logMh, dndlogm / ref - 1, label=label, color=color, lw=1.5)
            # Only show y-labels on first column
            if i % 2 == 0:
                ax[r, c].set_ylabel(r'$\delta \  dn/dlog_{10}(M$)')
            # Only show x-labels on last row
            if r == 4:
                ax[r, c].set_xlabel(r'$M_h\ (M_\odot/h)$')
            # Axis formatting for log-scale and grid
            ax[r, c].set_xscale('log')
            ax[r, c].grid(which='both', linestyle='--', linewidth=0.5)
            # Place legend outside right of subplot for clarity
            ax[r, c].legend(fontsize=8, loc='center left', bbox_to_anchor=(1.05, 0.5))
        # Adjust subplot spacing and overall layout
        fig.subplots_adjust(hspace=0.4, wspace=0.2, right=0.82)
        fig.suptitle('Fractional chnage in Halo mass Fucntion', fontsize=14, y=1.02)
        fig.tight_layout(rect=[0, 0, 0.8, 1])
        # Save figure if directory is provided
        if savefig_dir is not None:
            fig.savefig(os.path.join(savefig_dir, 'hmf_cosmo_sensitivity.pdf'))


    def dndm_planck18(self):
        """
        Plot the predicted halo mass function dndm for the Planck18 cosmology.
        """
        from astropy.cosmology import Planck18
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



