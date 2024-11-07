import h5py
import numpy as np
from matplotlib import pyplot as plt
from glob import glob
from os import path as op


class PlotCorr():
    def __init__(self):
        pass

    def compare_cosmos(self, corr_files, mode='projected', savefig=None, r_range=(0,100), legend=False):
        """
        Compare the correlation functions of different cosmologies
        Parameters:
        -----------
        save_files: list
            List of files to compare
        """
        fig, ax = plt.subplots(figsize=(8, 8))
        numbers = [int(op.basename(f).split('_')[-1].split('.')[0]) for f in corr_files]
        for c, svf in enumerate(corr_files):
            if 'Box250_Part750' in svf:
                pref = 'L2'
            elif 'Box1000_Part3000' in svf:
                pref = 'HF'
            else:
                pref='L1'
            with h5py.File(svf, 'r') as f:
                r = f['r'][:]
                if mode=='projected':
                    corr = np.mean(f['corr'][:], axis=2)
                elif mode=='1d':
                    corr = f['corr'][:]
                corr = np.mean(corr, axis=0)
            ind = np.where((r>r_range[0]) & (r<r_range[1]))
            ax.scatter(r[ind], corr[ind], alpha=0.5, marker='o', s=5)
            ax.plot(r[ind], corr[ind], label=f'{pref} | {numbers[c]}', alpha=0.5)

        if mode == 'projected':
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel(r'$r_p [Mpc/h]$')
            ax.set_ylabel(r'$w_p(r)$')
        if legend:
            ax.legend()
        if savefig is not None:
            plt.savefig(op.join(self.save_dir, savefig))
            plt.close()

    def compare_fidelities(self, save_dir):
        """
        Compare the fidelities of the different cosmologies
        """

        hf_corrs, l2_corrs = self.get_comman_pairs(save_dir)
        for i in range(len(hf_corrs)):
            self.compare_cosmos([hf_corrs[i], l2_corrs[i]], mode='projected', legend=True)
    
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