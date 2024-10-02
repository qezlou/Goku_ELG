import numpy as np
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
    args = argparser.parse_args()
    
    def get_linear_corr(params, fixed_params):
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
        h, sigma8 = params
        r = fixed_params['r']
        z = fixed_params['z']
        cosmo = cosmology.Planck15
        cosmo = cosmo.clone(h=h)
        Plin = cosmology.LinearPower(cosmo, redshift=z, transfer='EisensteinHu')
        Plin.sigma8 = sigma8
        cf_lin = cosmology.CorrelationFunction(Plin)
        return cf_lin(r)

    def fit_param():
        param_fit = []
        for p in param_range:
            model.set_params(**{param: p})
            model.fit(data)
            param_fit.append(model.score(data))
    
    def log_prior(params, param_range):
        # Define a log-prior function with bounds on the parameters
        for i, p in enumerate(params):
            if (p < param_range[i][0])*(param_range[i][1] > p):
                return -np.inf # Log of 0, meaning impossible outside the range
        return 0.0 # Log of 1, meaning no penalty within this range

    def log_likelihood(params, fixed_params, data, sigma):
        # Define a log-likelihood function
        pred = get_linear_corr(params, fixed_params)
        return -0.5 * np.sum((data - pred) ** 2 / sigma ** 2)

    def log_posterior(params, fixed_params, param_range, data, sigma):
        # Define the log-posterior function including the prior term
        log_pr = log_prior(params, param_range)
        if not np.isfinite(log_pr):
            return -np.inf  # If the prior is not finite, return -inf
        return log_pr + log_likelihood(params, fixed_params, data, sigma)

    def run_mcmc(nwalkers, n_iter, ndim, data, sigma, fixed_params, param_range, pool):

        sampler = emcee.EnsembleSampler(nwalkers, ndim, 
                                        log_posterior, 
                                        args=(fixed_params, param_range, data, sigma),
                                        pool=pool)
        pos = np.zeros((nwalkers, ndim))
        for p in range(ndim):
            pos[:, p] = np.random.uniform(param_range[p][0], param_range[p][1], nwalkers)
        sampler.run_mcmc(pos, n_iter, progress=True)
        return sampler

    fnames = glob(f'corr_{args.stype}_*.hdf5')

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
    fixed_params = {'r': r[ind], 'z': all_corr[args.stype]['z']}
    param_range = [(0.60, 0.70), (0.7, 1.7)]
    # get the percentiles, mostly for sigma which is used in the posterior
    percents = np.array([2.28, 15.87, 50, 84.13, 97.27])
    pcnt = np.percentile(np.array(all_corr[args.stype]['corr']), percents, axis=0)
    sigma = 0.5*(pcnt[3] - pcnt[1])[ind]

    if args.stack:
        # fit for the median correlation function
        print('Stacking the mocks')
        data = pcnt[2]
        data = data[ind]
        sigma = sigma[ind]

        if not args.optimize:
            # Run MCMC
            nawalkers = 10
            n_iter = 1_000
            ndim = 2
            
            with Pool() as pool:
                sampler = run_mcmc(nawalkers, n_iter, ndim, data, sigma, fixed_params, param_range, pool)
            samples = sampler.get_chain(discard=100, thin=15, flat=True)

            with h5py.File('samples.hdf5', 'w') as f:
                f.create_dataset('samples', data=samples)
                f.create_dataset('lnprob', data=sampler.get_log_prob(discard=100, thin=15, flat=True))
                f.create_dataset('acceptance', data=sampler.acceptance_fraction)
                f.create_dataset('acor', data=sampler.get_autocorr_time())
                f.create_dataset('params', data=['h', 'sigma8'])
                f.create_dataset('fixed_params', data=[all_corr[args.stype]['r'], all_corr[args.stype]['z']])
                f.create_dataset('param_range', data=param_range)
                f.create_dataset('nwalkers', data=nawalkers)
                f.create_dataset('n_iter', data=n_iter)
                f.create_dataset('ndim', data=ndim)
                f.create_dataset('data', data=data)
                f.create_dataset('sigma', data=sigma)
                f.create_dataset('stype', data=args.stype)
        else:
            # Get the maximum a posteriori estimate, fatser
            res = minimize(lambda x: -log_posterior(x, fixed_params, param_range, 
                                                    data, sigma), x0=[0.65, 1.0], bounds=param_range,
                                                    options={'disp': False})
    else:   
            # Fit for individual correlation functions
            all_fits = np.zeros((len(all_corr[args.stype]['corr']), 2))
            for i in tqdm(range(len(all_corr[args.stype]['corr'])), desc="Fitting individual mocks"):
                data = all_corr[args.stype]['corr'][i][ind]
                res = minimize(lambda x: -log_posterior(x, fixed_params, param_range,
                                                        data, sigma), x0=[0.65, 1.0], bounds=param_range,
                                                        options={'disp': False}).x
                all_fits[i] = np.array(res)
            with h5py.File(f'fits_{args.stype}_r_{args.r_range[0]}_{args.r_range[1]}.hdf5', 'w') as f:
                f.create_dataset('fits', data=all_fits)
                f.create_dataset('r', data=fixed_params['r'])
                f.create_dataset('z', data=fixed_params['z'])
                f.create_dataset('param_range', data=param_range)
                f.create_dataset('stype', data=args.stype)

