import numpy as np
import os.path as op
import matplotlib.pyplot as plt
import emcee
import corner
from nbodykit import cosmology
import h5py 
from glob import glob
import argparse
from multiprocessing import Pool
from scipy.optimize import minimize
from tqdm import tqdm

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--stype', type=str, default='lae', help='Type of source to fit')
    argparser.add_argument('--optimize', type=int,  default=1, help='Optimize the parameters')
    argparser.add_argument('--r_range', type=float, nargs=2, default=[60, 200], help='Range of r to fit')
    argparser.add_argument('--stack', type=int,  default=0, help='Stack the mocks')
    argparser.add_argument('--data_dir', type=str,  default='/home/qezlou/HD2/HETDEX/cosmo/data/lognormal-mocks/')
    args = argparser.parse_args()

    
    def get_fiducial_linear_corr(r, z):
        """
        Parameters
        ----------
        params : list
            List of parameters to fit. In this case, [h, sigma8]
        fixed_params : dict
            Dictionary of fixed parameters. In this case, {'r': r, 'z': z}
        Returns
        -------
        array
            Array of correlation function values
        """
        cosmo = cosmology.Planck15
        Plin = cosmology.ZeldovichPower(cosmo, redshift=z, transfer='EisensteinHu')
        cf_lin = cosmology.CorrelationFunction(Plin)
        return cf_lin(r)

    def log_prior(params):
        # Define a log-prior function with bounds on the parameters
        for i, p in enumerate(params):
            if (p < param_range[i][0])*(param_range[i][1] > p):
                return -np.inf # Log of 0, meaning impossible outside the range
        return 0.0 # Log of 1, meaning no penalty within this range

    def log_likelihood(params):
        """Define a log-likelihood function"""
        # ```model = model_fid(alpha * r)```
        pred =  params[1]*np.interp(data_r, params[0]*fid_r, model_fid)
        
        return -0.5 * np.sum((data - pred) ** 2 / sigma ** 2)

    def log_posterior(params):
        # Define the log-posterior function including the prior term
        log_pr = log_prior(params)
        if not np.isfinite(log_pr):
            return -np.inf  # If the prior is not finite, return -inf
        return log_pr + log_likelihood(params)

    def run_mcmc(pool):

        sampler = emcee.EnsembleSampler(nwalkers, ndim, 
                                        log_posterior,
                                        pool=pool)
        pos = np.zeros((nwalkers, ndim))
        for p in range(ndim):
            pos[:, p] = np.random.uniform(param_range[p][0], param_range[p][1], nwalkers)
        sampler.run_mcmc(pos, n_iter, progress=True)
        return sampler

    fnames = glob(op.join(args.data_dir,f'corr_{args.stype}_*.hdf5'))

    all_corr = {'oii': {'corr':[],
                        'r': [],
                        'Header': {}},
                'lae': {'corr':[],
                        'r': [],
                        'Header': {}}}
    
    for fn in fnames:
        with h5py.File(fn, 'r') as f:
            corr = f['corr'][:]
            r = f['r'][:]
            all_corr[args.stype]['corr'].append(f['corr'][:])
            all_corr[args.stype]['r'] = r
            all_corr[args.stype]['z'] = f['Header'].attrs['z']
    
    print(args.stype, all_corr[args.stype]['z'], args.r_range)
    # Fnd the correlation function values within the range
    ind = np.where((r >= args.r_range[0])*(r <= args.r_range[1]))
    data_r = r[ind]
    z = all_corr[args.stype]['z']
    # prior range for alpha and scale
    param_range = [(0.85, 1.15), (1, 5)]
    # get the percentiles, mostly for sigma which is used in the posterior
    percents = np.array([2.28, 15.87, 50, 84.13, 97.27])
    pcnt = np.percentile(np.array(all_corr[args.stype]['corr']), percents, axis=0)
    sigma = 0.5*(pcnt[3] - pcnt[1])[ind]
    data = pcnt[2][ind]

    fid_r = np.linspace(np.log10(r[ind][0]), np.log10(r[ind][-1]), 1000)
    fid_r = 10**fid_r

    print(f'data_r = {(data_r[0], data_r[-1])}, fid_r = {(fid_r[0], fid_r[-1])}')

    ## Fiducial Model: Linear power at Planc15 cosmology
    model_fid = get_fiducial_linear_corr(r=fid_r, z=all_corr[args.stype]['z'])


    # fit for the median correlation function
    print('Stacking the mocks')

    # Run MCMC
    nwalkers = 10
    n_iter = 1_000_000
    # Two parameters to fit: alpha and scale
    # `alpha`` is the scaling factor for the coordinates
    # `scale`` is the scaling factor for the correlation function
    ndim = 2 
    
    with Pool() as pool:
        sampler = run_mcmc(pool)
    samples = sampler.get_chain(discard=100, thin=15, flat=True)

    save_file = op.join(args.data_dir,f'samples_{args.stype}.hdf5')
    print(f'saving chains on file : {save_file}')
    

    with h5py.File(save_file, 'w') as f:
        f.create_dataset('samples', data=samples)
        f.create_dataset('lnprob', data=sampler.get_log_prob(discard=100, thin=15, flat=True))
        f.create_dataset('acceptance', data=sampler.acceptance_fraction)
        f.create_dataset('acor', data=sampler.get_autocorr_time())
        f.create_dataset('params', data=['alpha'])
        f.create_dataset('param_range', data=param_range)
        f.create_dataset('nwalkers', data=nwalkers)
        f.create_dataset('n_iter', data=n_iter)
        f.create_dataset('ndim', data=ndim)
        f.create_dataset('data', data=data)
        f.create_dataset('sigma', data=sigma)
        f.create_dataset('stype', data=args.stype)