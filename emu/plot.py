import h5py
import numpy as np
from matplotlib import pyplot as plt
from glob import glob
from os import path as op
import json
import re


class PlotCorr():
    def __init__(self):
        pass

    def compare_cosmos(self, corr_files, mode='projected', savefig=None, r_range=(0,100), legend=False, show_cosmo=False):
        """
        Compare the correlation functions of different cosmologies
        Parameters:
        -----------
        save_files: list
            List of files to compare
        """
        fig, ax = plt.subplots(figsize=(8, 8))
        if show_cosmo:
            plt_labels = self.get_cosmo_params(self.get_labels(corr_files))

        numbers = [int(op.basename(f).split('_')[-1].split('.')[0]) for f in corr_files]
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
                plt_lb['scalar_amp'] *= 1e9
                for k, v in plt_lb.items():
                    plt_lb[k] = round(v, 3)
            
            with h5py.File(svf, 'r') as f:
                r = f['r'][:]
                if mode=='projected' or mode=='1d':
                    corr = f['corr'][:]
                    corr = f['corr'][:]
                corr = np.mean(corr, axis=0)
            ind = np.where((r>r_range[0]) & (r<r_range[1]))
            ax.scatter(r[ind], corr[ind], alpha=0.5, marker='o', s=5)
            ax.plot(r[ind], corr[ind], label=plt_lb, alpha=0.5)

        if mode == 'projected':
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel(r'$r_p [Mpc/h]$')
            ax.set_ylabel(r'$w_p(r)$')
        if show_cosmo or legend:
            ax.legend()
        if savefig is not None:
            plt.savefig(op.join(self.save_dir, savefig))
            plt.close()

    def compare_fidelities(self, save_dir, show_cosmo=False):
        """
        Compare the fidelities of the different cosmologies
        """

        hf_corrs, l2_corrs = self.get_comman_pairs(save_dir)
        for i in range(len(hf_corrs)):
            self.compare_cosmos([hf_corrs[i], l2_corrs[i]], mode='projected', legend=True, show_cosmo=show_cosmo)
    
    def get_comman_pairs(self, save_dir):
        """
        Get the common pairs between the different cosmologies
        """
        hf_corrs = glob(op.join(save_dir, '*Box1000_Part3000*'))
        numbers = [int(op.basename(f).split('_')[-1].split('.')[0]) for f in hf_corrs]
        l2_corrs = [glob(op.join(save_dir,  f'*Box250_Part750_{str(n).rjust(4, "0")}*')) for n in numbers]
        hf_corrs = [f for f in hf_corrs if len(l2_corrs[numbers.index(int(op.basename(f).split('_')[-1].split('.')[0]))])>0]
        l2_corrs = [item for sublist in l2_corrs for item in sublist]

        return hf_corrs, l2_corrs
    
    def load_ics(self, ic_file='all_ICs.json'):
        """
        Load the IC json file
        """
        # Load JSON file as a dictionary
        with open(ic_file, 'r') as file:
            data = json.load(file)
        return data
        

    def get_cosmo_params(self, labels=[], params=['hubble', 'scalar_amp', 'ns']):
        """
        get comological parameters from the simulations listed in the labels
        """
        ics = self.load_ics()
        cosmo_params = []
        for lb in labels:
            for ic in ics:
                if ic['label'] == lb:
                    cosmo_params.append({k:ic[k] for k in params})
                    break
        assert len(cosmo_params) == len(labels), f'Some labels not found in the ICs file, found = {len(cosmo_params)}, asked for = {len(labels)}'
        return cosmo_params

    def get_labels(self, path_list):
        
        labels = [re.search(r'cosmo_10p_Box\d+_Part\d+_\d{4}',pl).group(0) for pl in path_list]
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

        