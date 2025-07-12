"""
Temporary place:

to get G(k) and predict the power spectrum
"""
import importlib
import matplotlib.pyplot as plt
from gal_goku import gal
importlib.reload(gal)
from scipy.interpolate import interp1d
import mcfit
import numpy as np

import importlib
from gal_goku import summary_stats
importlib.reload(summary_stats)




class Predictor:
    def __init__(self, data_dir, fid='HF', z=2.5):
        self.data_dir = data_dir
        self.fid = fid
        self.z = z
        self.k = np.logspace(np.log10(7e-3), np.log10(0.5), num=600)
        self.prop = summary_stats.Propagator(data_dir=data_dir, z=z, fid=fid)
        self.halo_pow = summary_stats.HaloHaloPower(data_dir, 
                                                    log_k_interp=np.log10(self.k), 
                                                    fid=fid, z=z)
        self.gk_model = self.prop.get_model_gk(self.k)


    def get_init_power(self, cosmo_pars):
        gal_base = gal.GalBase()
        cosmo_pars
        k_lin_init, p_lin_init = gal_base.get_init_linear_power(cosmo_pars=cosmo_pars)
        k_init = k_lin_init
        p_lin_init = p_lin_init

        # Resample p_hlin_model to logarithmic k within the same range as k_init
        interp_pinit = interp1d(k_init, np.copy(p_lin_init), axis=-1, bounds_error=False, fill_value="extrapolate")
        p_lin_init = interp_pinit(self.k)
        return p_lin_init

    def predict_phh(self, sim=15, mpairs=(11.3, 11.3)):


        assert self.prop.sim_tags[sim] == self.halo_pow.sim_tags[sim], \
            f'Simulation tags do not match: {self.prop.sim_tags[sim]} vs {self.halo_pow.sim_tags[sim]}'
        
        print(f'sim_tag = {self.prop.sim_tags[sim]}', flush=True)
        ind_m_hh = np.where(np.isclose(self.halo_pow.mpairs[:,0], mpairs[0], atol=1e-3) & 
                         np.isclose(self.halo_pow.mpairs[:,1], mpairs[1], atol=1e-3))[0][0]
        ind_m_1 = np.where(np.isclose(self.prop.mbins, mpairs[0], atol=1e-3))[0]
        ind_m_2 = np.where(np.isclose(self.prop.mbins, mpairs[1], atol=1e-3))[0]

        print(f'mass bins { self.halo_pow.mpairs[ind_m_hh]} and {(self.prop.mbins[ind_m_1][0], self.prop.mbins[ind_m_2][0])}')
        
        p_init = self.get_init_power(cosmo_pars=self.prop.params[sim])
        p_hh_model = self.gk_model[sim, ind_m_1]*self.gk_model[sim, ind_m_2] * p_init
        p_hh_sim = self.halo_pow.pk[sim, ind_m_hh,:]

        print(f'k shape: {self.k.shape}, p_hh_model shape: {p_hh_model.shape}, p_hh_sim shape: {p_hh_sim.shape}')

        # Convert power spectrum to correlation function
        r_hh_model, xi_hh_model = mcfit.P2xi(self.k, l=0, lowring=True)(np.copy(p_hh_model), extrap=True)
        r_hh_sim, xi_hh_sim = mcfit.P2xi(self.k, l=0, lowring=True)(np.copy(p_hh_sim), extrap=True)

        results = {
            'k': self.k,
            'p_hh_model': p_hh_model,
            'p_hh_sim': p_hh_sim,
            'xi_hh_model': xi_hh_model,
            'xi_hh_sim': xi_hh_sim,
            'r_hh_model': r_hh_model,
            'r_hh_sim': r_hh_sim
        }

        return results


