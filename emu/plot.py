import h5py
import numpy as np
from matplotlib import pyplot as plt
from glob import glob
from os import path as op


class PlotCorr():
    def __init__(self):
        pass

    def compare_cosmos(self, corr_files, mode='projected', savefig=None, r_range=(0,100)):
        """
        Compare the correlation functions of different cosmologies
        Parameters:
        -----------
        save_files: list
            List of files to compare
        """
        fig, ax = plt.subplots(figsize=(8, 8))
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
            ax.plot(r[ind], corr[ind], label=f'{c}', alpha=0.5)

        if mode == 'projected':
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel('r_p [Mpc/h]')
            ax.set_ylabel(r'$w_p(r)$')
        if savefig is not None:
            plt.savefig(op.join(self.save_dir, savefig))
            plt.close()