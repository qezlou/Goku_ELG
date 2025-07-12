"""
Load the training and validation datasets.
"""
import numpy as np
from gal_goku import summary_stats

def training_data(data_dir, fid='L2', emu_type={'wide_and_narrow':True}):
    """
    Load the training and validation datasets.
    """
    xi = summary_stats.Xi(data_dir=data_dir, fid=fid, narrow=False, MPI=None, logging_level='INFO')
    X, Y, mask, labels = [], [], [], []
    mbins, Y_wide, err_Wide, X_wide, labels_wide = xi.get_wt_err(rcut=(0.2, 61))
    # The missing bins are set to large error, so here we only use a mask for them
    mask_wide = (err_Wide > 0.1).astype(float)

    if not emu_type['wide_and_narrow']:
        X.append(X_wide)
        Y.append(Y_wide)
        mask.append(mask_wide)
        labels.append(labels_wide)
    else:
        # Load narrow data
        xi_narrow = summary_stats.Xi(data_dir=data_dir, fid=fid, narrow=True, MPI=None, logging_level='INFO')
        X_narrow, Y_narrow, err_narrow, labels_narrow = xi_narrow.get_wt_err(rcut=(0.2, 61))
        mask_narrow = (err_narrow > 0.1).astype(float)
        

        X.append(np.concatenate((X_wide, X_narrow), axis=0))
        Y.append(np.concatenate((Y_wide, Y_narrow), axis=0))
        mask.append(np.concatenate((mask_wide, mask_narrow), axis=0))
        labels.append(np.concatenate((labels_wide, labels_narrow), axis=0))

    return X, Y, mask, labels

