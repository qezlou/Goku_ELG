import json
import re
import logging
from glob import glob
from os import path as op
import h5py
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter1d as gf1d

from . import summary_stats
from . import wp_emus
from . import single_fid
import warnings
warnings.filterwarnings("ignore")

class BasePlot():
    def __init__(self, logging_level='INFO', show_full_params=False):
        self.logger = self.configure_logging(logging_level)
        if show_full_params:
            self.params=['hubble', 'scalar_amp', 'ns', 'omega0', 'omegab', 
                         'w0_fld', 'wa_fld', 'N_ur', 'alpha_s', 'm_nu']
        else:
            self.params = ['scalar_amp', 'ns']
        self.latex_labels = {'omega0': r'$\Omega_m$', 'omegab': r'$\Omega_b$', 
                             'hubble': r'$h$', 'scalar_amp': r'$A_s$', 'ns': r'$n_s$', 
                             'w0_fld': r'$w_0$', 'wa_fld': r'$w_a$', 'N_ur': r'$N_{ur}$', 
                             'alpha_s': r'$\alpha_s$', 'm_nu': r'$m_{\nu}$'}

    def configure_logging(self, logging_level):
        """Sets up logging based on the provided logging level."""
        logger = logging.getLogger('ProjCorrEmus')
        logger.setLevel(logging_level)
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        return logger
   
    def pred_truth(self, pred, truth, bins, model_err=None, seed=None, title=None, log_y=True):
        """
        Plot the leave one out cross validation
        Parameters:
        -----------
        pred: Array
            Predicted values, SHOULD be in log space, e.g. log(w_p)
        truth: Array
            Truth values, SHOULD be in log space, e.g. log(w_p)
        
        """
        if model_err is not None:
            fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        else:
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        if seed is not None:
            np.random.seed(seed)
        ind = np.random.randint(0, pred.shape[0], 10)
        for c,i in enumerate(ind):
            if log_y:
                ax[0].plot(bins, 10**truth[i], label='Truth', color=f'C{c}', lw=5, alpha=0.5 )
                ax[0].plot(bins, 10**pred[i], label='Pred', color=f'C{c}', ls='--', alpha=1)
            else:
                ax[0].plot(bins, truth[i], label='Truth', color=f'C{c}', lw=5, alpha=0.5)
                ax[0].plot(bins, pred[i], label='Pred', color=f'C{c}', ls='--', alpha=1)
            if c == 0:
                ax[0].legend()
        if log_y:
            ax[0].set_yscale('log')
        if title is not None:
            ax[0].set_title(title, fontsize=8)
        
        # plot histogram of LOO error
        assert np.all(10**truth > 0), 'Truth has negative values'
        try:
             assert np.all(10**pred > 0), f'Pred has negative values, pred nans: = {pred}'
             percentile_method = np.percentile
        except AssertionError as e:
            percentile_method = np.nanpercentile
        
        if log_y:
            err = np.abs(10**pred / 10**truth - 1).flatten()
        else:
            err = np.abs(pred / truth - 1).flatten()
        bins = np.logspace(-3, 0.5, 20)
        ax[1].hist(err, bins = bins, alpha=0.5)
        ax[1].set_xscale('log')
        percentiles = percentile_method(err, [84, 95])
        ax[1].set_title(f'Error distribution, 84, 95th percentiles = {np.round(percentiles,2)}', fontsize=10)

        if model_err is not None:
            if log_y:
                relative_err = (10**pred - 10**truth) / 10**model_err
            else:
                relative_err = (pred - truth) / model_err
            ax[2].hist(relative_err.flatten(), bins=np.linspace(-5, 5, 50), alpha=0.5)
            ax[2].set_xlabel(r'$(pred - truth)/\sigma_{pred}$')
            frac_within_1sigma = np.sum(np.abs(relative_err) < 1) / relative_err.size
            ax[2].set_title('Fraction within '+r'$1 \sigma_{pred} = \ $'+f'{frac_within_1sigma:.2f}', fontsize=10)
        
        return fig, ax

class PlotCorr(BasePlot):
    def __init__(self, logging_level='INFO', show_full_params=False):
        super().__init__(logging_level, show_full_params)

    def compare_cosmos(self, corr_files, fig=None, ax=None, mode='projected', savefig=None, r_range=(0,100), legend=False, show_cosmo=False, errorbar=False):
        """
        Compare the correlation functions of different cosmologies
        Parameters:
        -----------
        save_files: list
            List of files to compare
        """
        if fig is None:
            fig, ax = plt.subplots(figsize=(5, 5))
        if show_cosmo:
            plt_labels = self.get_cosmo_params(self.get_labels(corr_files))

        numbers = [int(re.search(r'\d{4}.hdf5', op.basename(f)).group(0).split('.')[0])  for f in corr_files]
        for c, svf in enumerate(corr_files):
            if not show_cosmo:
                if 'Box250_Part750' in svf:
                    pref = 'L2'
                elif 'Box1000_Part3000' in svf:
                    pref = 'HF'
                else:
                    pref='L1'
                plt_lb = f'{pref} | {str(numbers[c])}'
            else:
                plt_lb = plt_labels[c]
                for k, v in plt_lb.items():
                    plt_lb[k] = round(v, 3)
            
            with h5py.File(svf, 'r') as f:
                r = f['r'][:]
                if mode=='projected' or mode=='1d':
                    corr = f['corr'][:]
                    std = np.std(f['corr'][:], axis=0)
                corr = np.mean(corr, axis=0)
            ind = np.where((r>r_range[0]) & (r<r_range[1]))
            #ax.scatter(r[ind], corr[ind], alpha=0.5, marker='o', s=5)
            if errorbar:
                ax.errorbar(r[ind], corr[ind], yerr=std[ind], label=plt_lb, marker='o', ls='--', alpha=0.35, capsize=5)
            else:
                ax.plot(r[ind], corr[ind], label=plt_lb, marker='o', alpha=0.5)

        if mode == 'projected':
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel(r'$r_p [Mpc/h]$')
            ax.set_ylabel(r'$w_p(r)$')
        if show_cosmo or legend:
            ax.legend()
        if savefig is not None:
            plt.savefig(savefig)
            plt.close()
    
    def bin_in_param_space(self, data_dir, fid, num_bins=2, mode='projected', r_range=(0, 100), savefig=None):
        """
        Bin the correlation functions in the parameter space
        """
        fig, ax = plt.subplots(2, 5, figsize=(16, 8))
        fig_ratio, ax_ratio = plt.subplots(2, 5, figsize=(16, 8))

        if mode == 'projected':
            proj_corr = summary_stats.ProjCorr(data_dir=data_dir, fid=fid)
            param_names = proj_corr.param_names
            r, av_corr, av_corr_std, bins = proj_corr.bin_in_param_space(num_bins=num_bins, r_range=r_range)
        else:
            raise NotImplementedError('Only projected correlation function is implemented')
        ls = ['solid', 'dashed', 'dotted', 'dashdot']
        colors = ['C0', 'C1', 'C2', 'C3']
        for p in range(av_corr.shape[1]):
            ax_indx, ax_indy = p//5, p%5
            pname = self.latex_labels[param_names[p]]
            # For the sake of labeling on the plot
            if param_names[p] == 'scalar_amp':
                bins[p] *= 1e9
            for b in range(num_bins):
                ax[ax_indx, ax_indy].plot(r, av_corr[b,p], label=f' {bins[p,b]:.2f} < {pname} < {bins[p,b+1]:.2f}', alpha=0.5, color=colors[b], ls=ls[b])
            ax_ratio[ax_indx, ax_indy].plot(r, av_corr[0,p]/av_corr[1,p], label=f' {bins[p,b]:.2f} < {pname} < {bins[p,b+1]:.2f}', alpha=0.5, color='C0', ls='solid')
            ax_ratio[ax_indx, ax_indy].hlines(1, r.min(), r.max(), color='C1', ls='--')
            
            ax[ax_indx, ax_indy].set_yscale('log')
            ax[ax_indx, ax_indy].set_ylabel(r'$w_p(r)$')
            ax_ratio[ax_indx, ax_indy].set_ylabel(r'$w_p(r) \ ratio$')
            for a in [ax[ax_indx, ax_indy], ax_ratio[ax_indx, ax_indy]]:
                a.set_xscale('log')
                a.set_xlabel(r'$r_p [Mpc/h]$')
                a.legend()
                a.set_title(pname)
        fig_ratio.tight_layout()
        fig.tight_layout()
        if savefig is not None:
            fig.savefig(savefig)

    def compare_fidelities(self, save_dir, r_range=(0,100), fids=['HF','L2'], show_cosmo=False, errorbar=False, narrow=False):
        """
        Compare the fidelities of the different cosmologies
        """

        first_corrs, second_corrs = self.get_comman_pairs(save_dir, fids=fids, narrow=narrow)
        pair_count = len(first_corrs)
        fig, ax = plt.subplots(int(np.ceil(pair_count/3)), 3, figsize=(10, 3*pair_count//3))
        for i in range(pair_count):
            indx, indy = i//3, i%3
            self.compare_cosmos([first_corrs[i], second_corrs[i]], fig=fig, ax=ax[indx, indy],  mode='projected', legend=True, show_cosmo=show_cosmo, errorbar=errorbar, r_range=r_range)
        fig.tight_layout()
    
    def get_comman_pairs(self, save_dir, fids=['HF','L2'], narrow=False):
        """
        Get the common pairs between the different cosmologies
        """
        patterns = {'HF': 'Box1000_Part3000', 'L2': 'Box250_Part750', 'L1': 'Box1000_Part750'}
        if narrow:
            raise NotImplementedError('Goku-Narrow not implemented yet')
        if 'HF' in fids:
            fids.remove('HF')
            first_corrs = glob(op.join(save_dir, f'*{patterns["HF"]}*'))
            self.logger.info(f'Number of HF files = {len(first_corrs)}')
        elif 'L2' in fids:
            fids.remove('L2')
            first_corrs = glob(op.join(save_dir,f'*{patterns["L2"]}*'))
            self.logger.info(f'Number of L2 files = {len(first_corrs)}')
        
        numbers = [int(re.search(r'\d{4}.hdf5', op.basename(f)).group(0).split('.')[0]) for f in first_corrs]

        second_corrs = [glob(op.join(save_dir,  f'*{patterns[fids[0]]}_{str(n).rjust(4, "0")}.hdf5')) for n in numbers]
        print(f'first_corrs {first_corrs}')
        print(f'second_corrs {second_corrs}')
        self.logger.info(f'Number of {fids[0]} files = {len(second_corrs)}')
        first_corrs = [f for f in first_corrs if len(second_corrs[numbers.index(int(op.basename(f).split('_')[-1].split('.')[0]))])>0]
        second_corrs = [item for sublist in second_corrs for item in sublist]
        self.logger.info(f'Number of common pairs = {len(first_corrs)}')
        return first_corrs, second_corrs
    
    def load_ics(self, ic_file='all_ICs.json'):
        """
        Load the IC json file
        """
        # Load JSON file as a dictionary
        with open(ic_file, 'r') as file:
            data = json.load(file)
        return data
        

    def get_cosmo_params(self, labels=[]):
        """
        get comological parameters from the simulations listed in the labels
        """
        ics = self.load_ics()
        cosmo_params = []
        for lb in labels:
            for ic in ics:
                if ic['label'] == lb:
                    cosmo_params.append({self.latex_labels[k]:ic[k] for k in self.params})
                    break
        assert len(cosmo_params) == len(labels), f'Some labels not found in the ICs file, found = {len(cosmo_params)}, asked for = {len(labels)}, labels = {labels}'
        return cosmo_params

    def get_labels(self, path_list):
        
        labels = [re.search(r'10p_Box\d+_Part\d+_\d{4}',pl).group(0) for pl in path_list]
        return labels
    
    def fft_vs_paircount(self, fft_files, pcount_files, r_range):

        fig, ax = plt.subplots(len(fft_files), 1, figsize=(8, 8))
        labels = self.get_labels(fft_files)
        for c, svf in enumerate(fft_files):
            with h5py.File(svf, 'r') as f:
                r = f['r'][:]
                corr = np.squeeze(f['corr'][:])
            ind = np.where((r>r_range[0]) & (r<r_range[1]))
            ax[c].scatter(r[ind], corr[ind], alpha=0.5, marker='o', s=5)
            ax[c].plot(r[ind], corr[ind], label='FFT', alpha=0.5)

        for c, svf in enumerate(pcount_files):
            with h5py.File(svf, 'r') as f:
                r = f['r'][:]
                corr = np.squeeze(f['corr'][:])
            ind = np.where((r>r_range[0]) & (r<r_range[1]))
            ax[c].scatter(r[ind], corr[ind], alpha=0.5, marker='o', s=5)
            ax[c].plot(r[ind], corr[ind], label='Paircount', alpha=0.5)

            ax[c].set_xscale('log')
            ax[c].set_yscale('log')
            ax[c].set_xlabel(r'$r_p [Mpc/h]$')
            ax[c].set_ylabel(r'$\xi(r)$')
            ax[c].legend()
            ax[c].set_title(labels[c])
        fig.tight_layout()
        plt.show()

    def param_distribution(self, bad_ind, param_arrays, param_names):
        """
        Plotting the distribution of the cosmo paramters
         for a list of files
        """

        fig, ax = plt.subplots(2, 5, figsize=(16, 8))
        for p in range(param_arrays.shape[1]):
            ax_indx, ax_indy = p//5, p%5
            ax[ax_indx, ax_indy].hist(param_arrays[:,p], density=True ,bins=20, alpha=0.5, label='All sims')
            ax[ax_indx, ax_indy].hist(param_arrays[bad_ind,p], density=True, bins=20, alpha=0.5, label='bad sims')
            ax[ax_indx, ax_indy].set_xlabel(f"{self.latex_labels[param_names[p]]}")
            if p==0:
                ax[ax_indx, ax_indy].legend()
            

        fig.tight_layout()

    def outlier_sims(self, data_dir, thresh=0.5, r_range=(0,30), mode='projected', fid='L2', savefig=None):
        """
        Plotting the sims with large varation in HOD realizaitons
        """
        if mode == 'projected':
            proj = summary_stats.ProjCorr(data_dir=data_dir, fid='L2', logging_level='ERROR')
            rp, wp, model_err = proj.get_mean_std(r_range=r_range)
            relative_err = model_err / wp
        else:
            raise NotImplementedError('Only projected correlation function is implemented')
        
        fig, ax = plt.subplots(1, 1, figsize=(4, 3))
        ## Plot the HOD mdoel error distribution
        _ = ax.hist(relative_err.flatten(), bins = np.linspace(0, 1, 100), histtype='step', color='k', label='Model Error')
        ax.set_title(f'fraction of w_p bins with rel error > {thresh} = {(np.sum(relative_err > thresh) / relative_err.size):.2f}')
        ax.set_xlabel(r'$\sigma_{HOD} / \mu_{HOD}$')

        ind = np.where(relative_err > thresh)
        bad_sims = np.unique(ind[0])

        params_array = proj.get_params_array()
        self.param_distribution(bad_sims, params_array, proj.param_names)




    
class PlotProjCorrEmu(PlotCorr):
    """
    Plot routines for the emulator
    """
    def __init__(self, logging_level='INFO', data_dir=None, emu_type='LogLogSingleFid', **kwargs):
        
        if data_dir is None:
            self.data_dir = '/home/qezlou/HD2/HETDEX/cosmo/data/corr_projected_corrected/'
        else:
            self.data_dir = data_dir
        self.emu_type = emu_type
        
        PlotCorr.__init__(self, logging_level=logging_level, **kwargs)

    def pred_truth(self, pred, truth, rp, model_err=None, seed=None, title=None, log_y=True):
        """
        Plot the leave one out cross validation
        """
        if model_err is not None:
            fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        else:
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        if seed is not None:
            np.random.seed(seed)
        ind = np.random.randint(0, pred.shape[0], 10)
        for c,i in enumerate(ind):
            if log_y:
                ax[0].plot(rp, 10**truth[i], label='Truth', color=f'C{c}', lw=5, alpha=0.5 )
                ax[0].plot(rp, 10**pred[i], label='Pred', color=f'C{c}', ls='--', alpha=1)
            else:
                ax[0].plot(rp, truth[i], label='Truth', color=f'C{c}', lw=5, alpha=0.5)
                ax[0].plot(rp, pred[i], label='Pred', color=f'C{c}', ls='--', alpha=1)
            if c == 0:
                ax[0].legend()
        ax[0].set_xscale('log')
        ax[0].set_xlim(0, 30)
        ax[0].set_ylim(1, 1e4)
        ax[0].set_yscale('log')
        ax[0].set_ylabel(r'$w_p(r_p)$')
        ax[0].set_xlabel(r'$r_p$')
        if title is not None:
            ax[0].set_title(title, fontsize=8)
        
        # plot histogram of LOO error
        assert np.all(10**truth > 0), 'Truth has negative values'
        try:
             assert np.all(10**pred > 0), f'Pred has negative values, pred nans: = {pred}'
             percentile_method = np.percentile
        except AssertionError as e:
            percentile_method = np.nanpercentile
        
        if log_y:
            err = np.abs(10**pred / 10**truth - 1).flatten()
        else:
            err = np.abs(pred / truth - 1).flatten()
        bins = np.logspace(-3, 0.5, 20)
        ax[1].hist(err, bins = bins, alpha=0.5)
        ax[1].set_xscale('log')
        percentiles = percentile_method(err, [84, 95])
        ax[1].set_title(f'Error distribution, 84, 95th percentiles = {np.round(percentiles,2)}', fontsize=10)

        if model_err is not None:
            if log_y:
                relative_err = (10**pred - 10**truth) / 10**model_err
            else:
                relative_err = (pred - truth) / model_err
            ax[2].hist(relative_err.flatten(), bins=np.linspace(-5, 5, 50), alpha=0.5)
            ax[2].set_xlabel(r'$(pred - truth)/\sigma_{pred}$')
            frac_within_1sigma = np.sum(np.abs(relative_err) < 1) / relative_err.size
            ax[2].set_title('Fraction within '+r'$1 \sigma_{pred} = \ $'+f'{frac_within_1sigma:.2f}', fontsize=10)
        
        fig.tight_layout()

    def loo_pred_truth(self, savefile, seed=None, title=None, log_y=True):
        """
        """
        with h5py.File(savefile, 'r') as f:
            pred = f['pred'][:]
            var_pred = f['var_pred'][:]
            truth = f['truth'][:].squeeze()
            X = f['X'][:]
            rp = f['rp'][:]
        
        self.logger.info(f'Number of simualtions {pred.shape[0]}')
        
        self.pred_truth(pred, truth, rp, model_err=np.sqrt(var_pred), seed=seed, title=title, log_y=log_y)
    
    def leave_bunch_out(self, data_dir, fid='L2', n_out=5, narrow=False):
        """
        Leaves out a random bunch of samples out
        n_out: Number of samples to leave out
        """ 
        emu = wp_emus.SingleFid(data_dir=data_dir, fid='L2', logging_level='INFO')
        X_test, Y_test, Y_pred, var_pred = emu.leave_bunch_out(n_out=n_out, narrow=narrow)
        Y_pred = Y_pred.numpy()
        var_pred = var_pred.numpy()
        self.pred_truth(Y_pred, Y_test, emu.rp, model_err=np.sqrt(var_pred), seed=None, title='Leave bunch out', log_y=True)

    def pred_truth_input_space(self, n_out=5, r_range=(0, 30)):
        """
        """

        if self.emu_type == 'LogLogSingleFid':
            emu = wp_emus.LogLogSingleFid(data_dir=self.data_dir, r_range=r_range)

        model, X_test, Y_test = emu.leave_bunch_out(n_out=n_out)

        fig, ax = plt.subplots(5, 2, figsize=(10, 15))
        for i in range(X_test.shape[1]):
            xlabel = list(self.latex_labels.values())[i]
            pred, var = model.predict_y(X_test)
            for c, k in enumerate([10]):
                ax_ind_i = i//2
                ax_ind_j = i%2
                sort_ind = np.argsort(X_test[:, i])
                #ax[ax_ind_i, ax_ind_j].plot(X_test[sort_ind, i], Y_test[sort_ind,k], label='Truth', alpha=0.5, color=f'C{c}', marker='o')
                #ax.fill_between(grid, (pred[:,k]-np.sqrt(var[:,k])).squeeze(), (pred[:,k]+np.sqrt(var[:,k])).squeeze(), alpha=0.6, label='Pred', color=f'C{c}')
                ax[ax_ind_i, ax_ind_j].errorbar(X_test[:, i], pred[:, k], np.sqrt(var[:,k]), label=f'Pred, b= {k}', color=f'C{c}', marker='x', alpha=0.5, ls='none',ms=10)
                ax[ax_ind_i, ax_ind_j].set_xlabel(xlabel)
                ax[ax_ind_i, ax_ind_j].set_ylabel(f'Wp at bins')
                if ax_ind_j == 0:
                    ax[ax_ind_i, ax_ind_j].legend()
            
        fig.tight_layout()

    def param_sensitivity(self, r_range=(0, 30), cleaning_method='linear_interp'):
        """
        Plot the sensitivity of the emulator to changing the 
        parameters one at a time
        """

        if self.emu_type == 'LogLogSingleFid':
            emu = wp_emus.SingleFid(data_dir=self.data_dir, r_range=r_range, cleaning_method=cleaning_method)
        emu.evaluate.train()
        X_min, X_max = emu.evaluate.sf.X_min.numpy(), emu.evaluate.sf.X_max.numpy()

        lower, mid, upper = 0.1, 0.5, 0.9
        lower *= (X_max - X_min)
        lower += X_min
        mid *= (X_max - X_min)
        mid += X_min
        upper *= (X_max - X_min)
        upper += X_min
        
        fig, ax = plt.subplots(2, 5, figsize=(16, 8))
        fig_ratio, ax_ratio = plt.subplots(2, 5, figsize=(16, 8))
        ndim_input = emu.evaluate.sf.ndim_input

        
        for p in range(ndim_input):
            lower_upper_pred = []
            label = list(self.latex_labels.values())[p]
            X_eval = mid
            for b in range(2):
                if b == 0:
                    X_eval[p] = lower[p]
                else:
                    X_eval[p] = upper[p]
                pred, var = emu.evaluate.predict(X_eval[None,:])
                pred = pred.numpy().squeeze()
                var = var.numpy().squeeze()
                lower_upper_pred.append(10**pred)
                ax_ind_i, ax_ind_j = p//5, p%5
                # Just for the sake of plotting:
                if list(self.latex_labels.keys())[p] == 'scalar_amp':
                    param_value = X_eval[p]*1e9
                else:
                    param_value = X_eval[p]
                ax[ax_ind_i, ax_ind_j].plot(emu.rp,  10**pred, label=f'{label} = {param_value:.2f}', alpha=0.5, color=f'C{b}')
            
            #ax[ax_ind_i, ax_ind_j].set_ylim(3e-1, 1e1)
            ax[ax_ind_i, ax_ind_j].set_title(label)
            ax[ax_ind_i, ax_ind_j].set_xscale('log')
            ax[ax_ind_i, ax_ind_j].set_yscale('log')
            ax[ax_ind_i, ax_ind_j].set_ylabel(r'$W_p(r)$')
            ax[ax_ind_i, ax_ind_j].set_xlabel(r'$r_p$')
            ax[ax_ind_i, ax_ind_j].legend()


            ax_ratio[ax_ind_i, ax_ind_j].hlines(0, emu.rp.min(), emu.rp.max(), color='C0', ls='--')
            ax_ratio[ax_ind_i, ax_ind_j].plot(emu.rp, lower_upper_pred[0]/lower_upper_pred[1] - 1, label=f'{label} = {param_value:.2f}', alpha=0.5, color=f'C{b}')
            ax_ratio[ax_ind_i, ax_ind_j].set_title(label)
            ax_ratio[ax_ind_i, ax_ind_j].set_xscale('log')
            ax_ratio[ax_ind_i, ax_ind_j].set_ylabel(r'$W_{p,2} / W_{p,1} (r) - 1$')
            ax_ratio[ax_ind_i, ax_ind_j].set_xlabel(r'$r_p$')
            #ax_ratio[ax_ind_i, ax_ind_j].set_ylim(0.9, 2.5)
        fig.suptitle(f'replcing negative wp bins with  {cleaning_method}')
        fig.tight_layout()
        fig_ratio.tight_layout()

    def plot_planck_prediction(self, data_dir):

        planck_cosmo = np.array([0.31, 0.048, 0.68, 2.1e-9, 0.97, -1.0, 0, 3.08, 0, 0.1])[None,:]
        emu = wp_emus.SingleFid(data_dir=data_dir, fid='L2', logging_level='INFO')
        planck_cosmo = np.array([0.31, 0.048, 0.68, 2.1e-9, 0.97, -1.0, 0, 3.08, 0, 0.1])[None,:]
        wp_planck , _ = emu.predict(planck_cosmo) 
        fig, ax = plt.subplots()
        ax.plot(emu.rp, 10**wp_planck[0], label='Planck')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(which='both')
        ax.set_xlabel(r'$r_p$')
        ax.set_ylabel(r'$w_p(r_p)$')





class PlotTestEmus():
    def __init__(self, n_samples=200):
        self.test_sf = single_fid.TestSingleFiled(n_samples=n_samples, show_full_params=True)
    
    def pred_truth_output_space(self, X=None):
        
        fig, ax= plt.subplots(2, 1, figsize=(4, 8))
        if X is None:
            X = self.test_sf.X_train
            title = 'training set'
        else:
            title = 'for testing set'
        Y_pred, Y_var = self.test_sf.predict(X)
        Y_pred = Y_pred.numpy()
        Y_var = Y_var.numpy()
        Y_true = self.test_sf.get_truth(X)

        for c,i in enumerate(range(Y_pred.shape[0])):
            ax[0].scatter(np.arange(Y_true[i].size), Y_true[i], color=f'C{c}', marker='o', label='truth')
            #ax[0].plot(, label='Truth', color=f'C{c}', lw=5, alpha=0.5 )
            ax[0].fill_between(np.arange(Y_pred[i].size), Y_pred[i] - np.sqrt(Y_var[i]), Y_pred[i] + np.sqrt(Y_var[i]), label='Pred', color=f'C{c}',  alpha=0.6)
            if c == 0:
                ax[0].legend()
            ax[1].plot(Y_pred[i]/Y_true[i] - 1, label='Var', color=f'C{c}', ls='--', alpha=1)
        ax[0].grid()
        ax[1].set_xlabel('Output space')
        ax[0].set_ylabel('Output value')
        ax[1].set_ylabel('Fractional error')
        fig.suptitle(f'Prediction vs Truth for {title}')
        fig.tight_layout()

    
    def pred_truth_input_space(self, X=None):
        a = np.linspace(0.01, 10, 100)

        for s in range(self.test_sf.ndim_input):
            fig, ax = plt.subplots(1,3, figsize=(12, 4))
            for i, x in enumerate([0.1, 9.2]):
                if s==1:
                    X_test = np.vstack(([x]*a.size, a.flatten())).T
                else:
                    X_test = np.vstack((a.flatten(), [x]*a.size)).T
                # The columns of the axes grid are the different output dimensions
                Y_true = self.test_sf.get_truth(X_test)
                for dout in range(self.test_sf.ndim_output):     
                    ax[dout].plot(X_test[:,s],Y_true[:,dout], color=f'C{i}', ls='--', label='Truth')
                    ind = np.where((self.test_sf.X_train[:,(s+1)%2] >= x-0.05)*(self.test_sf.X_train[:,(s+1)%2] <= x+0.05))[0]
                    ax[dout].scatter(self.test_sf.X_train[ind,s], self.test_sf.Y_train[ind, dout], color=f'C{i}', label='Training', alpha=0.5)

                    mean, var = self.test_sf.predict(X_test)

                    ax[dout].fill_between(X_test[:,s], (mean[:, dout]-np.sqrt(var[:,dout])).numpy().squeeze(), (mean[:, dout]+np.sqrt(var[:, dout])).numpy().squeeze(), alpha=0.6, color=f'C{i}', label='Pred')
                    ax[dout].set_ylabel(f'Output space {dout}')
                    ax[dout].set_xlabel(f'Intput space {s}')
                    if dout == self.test_sf.ndim_output-1:
                        ax[dout].legend()
            fig.tight_layout()


            fig.tight_layout()
    
    def latent_space(self):
        """
        """
        fig, ax = plt.subplots(1, 1)
        ax.scatter(self.test_sf.X_train[:, 0], self.test_sf.X_train[:, 1], label='Training', alpha=0.5)
        ax.set_xlabel('Latent 1')
        ax.set_ylabel('Latent 2')
        ax.legend()
        plt.show()


class PlotHMF(BasePlot):
    """
    Class to plot Halo Mass fucntion and the meualtor
    
    """
    def __init__(self, data_dir, logging_level='INFO', show_full_params=False):
        super().__init__(logging_level, show_full_params)
        self.data_dir = data_dir

    def sim_hmf(self, fids=['HF'], fig=None, ax=None):
        """
        Plots the  halo mass function for simulations
        Parameters:
        --------------
        save_file: str
            h5 file storing the halpo mass functions
        """
        ws = [7, 2, 2]
        alphas = [0.5, 0.9, 0.8]
        if fig is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 4))

        hmfs, mbins, _ = self.hmf.load_hmf_sims(fids=fids)
        for i, fd in enumerate(fids):
            for j in range(hmfs[fd].shape[0]):
                ax.plot(mbins, hmfs[fd][j], alpha=alphas[i], lw=ws[i])
        ax.set_ylim((1e-8, 1e-1))
        ax.set_xscale('log')
        ax.legend()
        ax.set_yscale('log')
        ax.set_ylabel(r'$\psi \ [dex^{-1} cMph^{-3}]$')
        ax.set_xlim(1e11, 1e14)
        ax.set_xlim(1e11, 1e14)
        ax.grid()
        fig.tight_layout()
    
    def compare_fids(self, fids=['HF','L2']):
        """
        Compare the Halo Mass functions for different fidelities
        """
        ws = [4, 2]
        alphas = [0.5, 0.9, 0.8]
        fig, ax = plt.subplots(2, 1, figsize=(8, 10), gridspec_kw={'height_ratios': [6, 1]})

        (hmfs, mbins, sim_tags, hmfs_coarse, mbins_coarse, hmfs_fine) = self.hmf.common_pairs(fids, load_coarse=True)
        for i, fd in enumerate(fids):
            for j in range(hmfs[fd].shape[0]):
                ax[0].plot(mbins, hmfs[fd][j], alpha=alphas[i], lw=ws[i], color=f'C{j}')
                ax[0].plot(mbins, hmfs_fine[fd][j], alpha=alphas[i], lw=ws[i], color=f'C{j}', ls='--')
        if 'HF' in fids:
            ax[1].plot(mbins, np.zeros((hmfs['HF'].shape[1])), ls='dashed')
            ref = 'HF'
        elif 'L2' in fids:
            ax[1].plot(mbins, np.zeros((hmfs['L2'].shape[1])), ls='dashed')
            ref = 'L2'
        
        for i, k in enumerate(fids):
            if i > 0:
                for j in range(hmfs[k].shape[0]):
                    ax[1].plot(mbins, hmfs[k][j]/hmfs[ref][j] - 1, alpha=0.5, color=f'C{j}')

        ax[1].set_ylim((-0.5,0.5))
        ax[0].set_ylim((1e-5, 1e-1))
        ax[1].grid()
        for i in range(2):
            ax[i].set_xscale('log')
            ax[i].legend()
        ax[0].set_yscale('log')
        ax[1].set_xlabel('FOF halo Mass')
        ax[0].set_ylabel(r'$\psi \ [dex^{-1} cMph^{-3}]$')
        ax[0].set_xlim(1e11, 1e14)
        ax[0].set_xlim(1e11, 1e14)
        ax[1].set_ylabel('Ratio to HF')
        fig.tight_layout()

    def check_smoothed_hmf(self, fids=['HF', 'L2']):

        for i, fd in enumerate(fids):        
            # Use summary_stats to load the HMF
            halo_func = summary_stats.HMF(self.data_dir, fid=fd)
            (hmfs, mbins, sim_tags, hmfs_coarse, mbins_coarse, hmfs_fine) = halo_func.load_hmf_sims(load_coarse=True)
            sim_nums = hmfs.shape[0]
            rand_sample = np.random.randint(0, sim_nums, 21)
            fig, ax = plt.subplots(1,1, figsize=(8,8))

            for p, j in enumerate(rand_sample):
                ax.plot(mbins, hmfs[j], alpha=0.5, lw=2, color=f'C{j}', label='Interpolated')
                ax.plot(mbins_coarse, hmfs_coarse[j], alpha=0.7, lw=2, color=f'C{j}', ls='--', label='coarse bin')
                ax.plot(mbins, hmfs_fine[j], alpha=0.9, lw=2, color=f'C{j}', ls='dotted', label='fine bin')
                if p==0:
                    ax.legend()

                ax.set_ylim((1e-7, 1e-1))
                ax.set_xscale('log')
                ax.set_yscale('log')
                ax.set_ylabel(r'$\psi \ [dex^{-1} cMph^{-3}]$')
                ax.set_xlim(1e11, 1e14)
                ax.set_xlim(1e11, 1e14)
                fig.tight_layout()

class PlotHmfEmu(BasePlot):
    def __init__(self, logging_level='INFO', show_full_params=False):
        super().__init__(logging_level, show_full_params)
    
    def pred_truth(self, pred, truth, mbins, model_err=None, seed=None, title=None, log_y=True):
        """
        Plot the difference between the predicted and the truth
        """
        fig, ax = plt.subplots(1, 3, figsize=(12, 4))

    def pred_truth(self, pred, truth, mbins, model_err=None, seed=None, title=None, log_y=True):
        
        fig, ax = super().pred_truth(pred, truth, mbins, model_err=model_err, seed=seed, title=title, log_y=log_y)
        ax[0].set_yscale('log')
        ax[0].set_xscale('log')
        fig.tight_layout()
        return fig, ax
    
    def loo_pred_truth(self, savefile, seed=None, title=None):
        """
        """
        with h5py.File(savefile, 'r') as f:
            pred = f['pred'][:]
            var_pred = f['var_pred'][:]
            truth = f['truth'][:].squeeze()
            X = f['X'][:]
            mbins = f['bins'][:]
        
        self.logger.info(f'Number of simualtions {pred.shape[0]}')
        
        fig, ax = self.pred_truth(pred, truth, mbins, model_err=np.sqrt(var_pred), seed=seed, title=title, log_y=True)
        ax[0].set_ylim((1e-7, 1e-1))
    