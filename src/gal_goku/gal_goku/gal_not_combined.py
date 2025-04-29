# -*- coding: utf-8 -*-

import numpy as np
import mcfit
from gal_goku import emu_cosmo_not_combined

class GalBase:
    """
    The main class to convert halo summary statistics to galaxy observables, e.g. xi(r) and
    w_p(r), the 3d and projected correlation functions, respectively.
    """

    def __init__(self):
        pass

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
    

class XiGal(GalBase):
    """
    Galaxy correlation functions for a cosmology, redshift, and HOD model.
    This class contains methods to calculate the 3D correlation function xi(r) and the
    projected correlation function w_p(r) from the halo summary statistics.
    """

    def __init__(self, data_dir):
        super().__init__()
        self.dataa_dir = data_dir
        # Load the emulator for xi(r) on the grid
        # of mass thresholds. This take 20s to load
        self.xi_emu = emu_cosmo_not_combined.XiEmulator(dat_dir=self.data_dir)
    
    def _xi_hh_direct(self, cosmo, mthresh1, mthresh2, r):
        """
        Calculate the 3D correlation function xi(r) for a given set of halo masses.
        Parameters
        ----------
        mthresh1 : float
            The minimum halo mass for the first halo sample.
        mthresh2 : float
            The minimum halo mass for the second halo sample.
        r : array_like
            The distances at which to calculate the correlation function.
        Returns
        -------
        xi : array_like
            The 3D correlation function xi(r).
        """
        # Load the trained emualtor
        emu = emu_cosmo_not_combined.XiEmulator(dat_dir=self.data_dir, loggin_level='ERROR')
        # Get the emulator for the given mass pair
        xi_pred, xi_err = emu.predict_xi(mthresh1=mthresh1,
                                        mthresh2=mthresh2,
                                        cosmo=cosmo)
        return emu.rbins, xi_pred, xi_err

    def _phh_m1_m2(self, cosmo, mthresh1, mthresh2):
        """
        Get the 3D correlation function xi(r) for a given set of halo masses.
        Parameters
        ----------
        mthresh1 : float
            The minimum halo mass for the first halo sample.
        mthresh2 : float
            The minimum halo mass for the second halo sample.
        r : array_like
            The distances at which to calculate the correlation function.
        cosmo : list or np.ndarrays
            The desired cosmology to predict the xi_hh for
        Returns
        -------
        xi : array_like
            The 3D correlation function xi(r).
        """
        # Get the emulator for the given mass pair
        rbins, xi_pred, _ = self._xi_hh_direct(cosmo=cosmo,
                                                 mthresh1=mthresh1,
                                                 mthresh2=mthresh2)
        # Placeholder for the tree-level correlation function
        # and the stitching between the two
        
        # use mcfit to convert xi to P
        k, phh = mcfit.xi2P(rbins, l=0, lowring=True)(xi_pred, extrap=True)
        return  k, phh
    
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
    

    
        
        


        