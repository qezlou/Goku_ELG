"""
To evaluate all the emulators for the Halo Mass Function.
Evaluating multifid emulator, HF and L2 pairs. Using Ming-Feng's thin wrappers of Emukit.
"""

import logging
import pickle
import h5py
import numpy as np
from . import summary_stats
import torch
from nflows import flows as nf_flows
from nflows import distributions as nf_distributions
from nflows import transforms as nf_transforms
from nflows.nn import nets as nf_nets
from nflows.utils import torchutils
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.functional as F
import sys
import os
import os.path as op
import copy
from glob import glob
import time

try :
    #raise ImportError
    import mpi4py
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    mpi_size = comm.Get_size()
except ImportError:
    MPI = None
    comm = None
    rank = 0
    mpi_size = 1

class BaseStatEmu():

    def __init__(self, X, Y, 
                 logging_level='info', 
                 emu_type={'multi-fid':False, 'single-bin':False, 'linear':True, 'mf-svgp':False},
                 n_optimization_restarts=5, emu_args=None):
        """The base emu to be inherited by single fidelity emulators built on any summary statistics
        This is the interface for all classses above.
        :param X_train:  (n_fidelities, n_points, n_dims) list of parameter vectors.
        :param Y_train:  (n_fidelities, n_points, n_nins) list of matter power spectra.
        """
        self.logger = self.configure_logging(logging_level)
        # This if statement is to make sure that the emu_type has all the keys
        if 'mf-svgp' not in emu_type:
            emu_type['mf-svgp'] = False
        self.emu_type = emu_type
        self.n_optimization_restarts = n_optimization_restarts

        ## TO DO: Better layout for the emu_type
        if emu_type['mf-svgp'] and emu_type['multi-fid'] and emu_type['linear']:
            # Class for Linear multi-fidelity emulator using gpflow's SVGP
            # This helps with dimensionality reduction of the output space
            self.emu = LatentMFCoregionalizationSVGP

        elif emu_type['multi-fid'] and emu_type['linear'] and emu_type['single-bin']:
            # Class for Linear multi-fidelity emulators
            self.emu = gpemu.SingleBinLinearGP
        
        elif emu_type['multi-fid'] and not emu_type['linear'] and emu_type['single-bin']:
            # Class for Non-linear multi-fidelity emulators
            self.emu = gpemu.SingleBinNonLinearGP

        elif not emu_type['multi-fid'] and not emu_type['single-bin']:
            # Class for single-fidelity emulators for all bins
            self.emu = gpemu.SingleBinGP

        else:
            raise NotImplementedError("This type of emulator is not implemented")

        self.X = X
        self.Y = Y
        # If multi-fid = True, then X and Y are lists of arrays
        #if emu_type['multi-fid']:
        self.n_fidelities = len(X)
        self.n_points = []
        self.n_dims = []
        self.n_bins = []
        for n in range(self.n_fidelities):
            self.n_points.append(X[n].shape[0])
            self.n_dims.append(X[n].shape[1])
            self.n_bins.append(Y[n].shape[1])
        # If multi-fid = False, then X and Y are arrays
        #else:
        #    self.n_fidelities = 1
        #    self.n_points = X.shape[0]
        #    self.n_dims = X.shape[1]
        #    self.n_bins = Y.shape[1]

        if rank == 0:
            self.logger.info(f'Fidelities: {self.n_fidelities}, Points: {self.n_points}, Dimensions: {self.n_dims}, Bins: {self.n_bins}')

    def configure_logging(self, logging_level):
        """Sets up logging based on the provided logging level in an MPI environment."""
        logger = logging.getLogger('BaseStatEmu')
        logger.setLevel(logging_level)

        # Create a console handler with flushing
        console_handler = logging.StreamHandler(sys.stdout)

        # Include Rank, Logger Name, Timestamp, and Message in format
        formatter = logging.Formatter(
            f'%(name)s | %(asctime)s | Rank {rank} | %(levelname)s  |  %(message)s',
            datefmt='%m/%d/%Y %I:%M:%S %p'
        )
        console_handler.setFormatter(formatter)

        # Ensure logs flush immediately
        console_handler.flush = sys.stdout.flush  

        # Add handler to logger
        logger.addHandler(console_handler)
        
        return logger

        
    def loo_train_pred(self, savefile):
        """
        Get the leave one out predictions
        """
        mean_pred = np.zeros((self.n_points[-1], self.n_bins[-1]))
        var_pred = np.zeros((self.n_points[-1], self.n_bins[-1]))
        if self.emu_type['multi-fid']:
            for i, s in enumerate(self.labels[-1]):
                if  rank ==0:
                    self.logger.info(f'Leaving out {s}, progress {i}/{len(self.labels[-1])}')
                X_train = [self.X[0]]
                X_train.append(np.delete(self.X[-1], i, axis=0))
                Y_train = [self.Y[0]]
                Y_train.append(np.delete(self.Y[-1], i, axis=0))
                X_test = self.X[-1][i][np.newaxis, :]
                Y_test = self.Y[-1][i]
                #self.logger.info(X_train[0].shape, X_train[1].shape, Y_train[0].shape, Y_train[1].shape, X_test.shape, Y_test.shape)
                model = self.emu(copy.deepcopy(X_train), copy.deepcopy(Y_train), n_fidelities=self.n_fidelities, kernel_list=None)
                model.optimize(n_optimization_restarts=self.n_optimization_restarts)
                mean_pred[i], var_pred[i] = model.predict(X_test)
        else:
            for i, s in enumerate(self.labels[0]):
                self.logger.info(f'Leaving out {s}')
                X_train = np.delete(self.X[0], i, axis=0)
                Y_train = np.delete(self.Y[0], i, axis=0)
                X_test = self.X[0][i][np.newaxis, :]
                Y_test = self.Y[0][i]
                model = self.emu(X_train, Y_train, kernel_list=None, single_bin=self.emu_type['single-bin'])
                model.optimize(n_optimization_restarts=self.n_optimization_restarts)
                mean_pred[i], var_pred[i] = model.predict(X_test)
        if MPI is not None:
            comm.Barrier()
        if rank==0:
            with h5py.File(savefile, 'w') as f:
                self.logger.info(f'Writing on {savefile}')
                f.create_dataset('pred', data=mean_pred)
                f.create_dataset('var_pred', data=var_pred)
                f.create_dataset('bins', data=self.mbins)
                if self.emu_type['multi-fid']:
                    f.create_dataset('truth', data=self.Y[-1])
                    f.create_dataset('X', data=self.X[-1])
                    # Writing a string dataset on h5py is a bit tricky
                    labels = np.array(self.labels[-1], dtype='S')
                    # Define an HDF5-compatible string data type
                    string_dtype = h5py.string_dtype(encoding='utf-8')
                    f.create_dataset('labels', data=labels.astype(string_dtype), dtype=string_dtype)
                else:
                    f.create_dataset('truth', data=self.Y)
                    f.create_dataset('X', data=self.X)
        if MPI is not None:
            comm.Barrier()
    
    def train_pred_all_sims(self, savefile=None):
        """
        Train the model on all simulations and comapre with the truth
        """
        if self.emu_type['mf-svgp']:
            # Add the fidelity indocators
            X_l2_aug = np.hstack([self.X[0], np.zeros((self.X[0].shape[0], 1))])
            X_hf_aug = np.hstack([self.X[-1], np.ones((self.X[-1].shape[0], 1))])
            # Stack the L2 and HF data vertically
            self.X = np.vstack([X_l2_aug, X_hf_aug])
            self.Y = np.vstack([self.Y[0], self.Y[-1]])
            # Base kernel of the MF GP
            kernel_L = gpflow.kernels.SquaredExponential(lengthscales=np.ones(self.X.shape[1]-1), variance=1.0)
            kernel_delta = gpflow.kernels.SquaredExponential(lengthscales=np.ones(self.X.shape[1]-1), variance=1.0)
            self.emu = LatentMFCoregionalizationSVGP(self.X, self.Y, kernel_L, kernel_delta, 
                                                     num_latents=5, num_inducing=100,
                                                     num_outputs=self.n_bins[0])
            self.logger.info(f'shapes passed to LMF : {self.X.shape, self.Y.shape}')
            self.emu.optimize(data=(self.X, self.Y))

        
        elif self.emu_type['multi-fid']:
            model = self.emu(copy.deepcopy(self.X), copy.deepcopy(self.Y), n_fidelities=self.n_fidelities, kernel_list=None)
            model.optimize(n_optimization_restarts=self.n_optimization_restarts)
        else:
            model = self.emu(self.X[0], self.Y[0], kernel_list=None, single_bin=self.emu_type['single-bin'])
            model.optimize(n_optimization_restarts=self.n_optimization_restarts)
            #if self.emu_type['multi-fid']:
        mean_pred, var_pred = model.predict(copy.deepcopy(self.X[-1]))
        #else:
        #    mean_pred, var_pred = model.predict(self.X)
        if MPI is not None:
            comm.Barrier()
        if savefile is not None:
            if rank == 0:
                self.logger.info(f'Writing on {savefile}')
                with h5py.File(savefile, 'w') as f:
                    f.create_dataset('pred', data=mean_pred)
                    f.create_dataset('var_pred', data=var_pred)
                    if self.emu_type['multi-fid']:
                        f.create_dataset('truth', data=self.Y[-1])
                        f.create_dataset('X', data=self.X[-1])
                        labels = np.array(self.labels[-1], dtype='S')
                        # Define an HDF5-compatible string data type
                        string_dtype = h5py.string_dtype(encoding='utf-8')
                        f.create_dataset('labels', data=labels.astype(string_dtype), dtype=string_dtype)
                    else:
                        f.create_dataset('truth', data=self.Y)
                        f.create_dataset('X', data=self.X)
                    f.create_dataset('bins', data=self.mbins)

                    
            if MPI is not None:
                comm.Barrier()
    
    def train(self, save_dir=None):
        """
        Train the model and save this in `save_dir`, furthur instruction in 
        `save` routines of `gal_goku.gpemulator_singlebin`
        """
        if self.emu_type['multi-fid']:
            model = self.emu(copy.deepcopy(self.X), copy.deepcopy(self.Y), n_fidelities=self.n_fidelities, kernel_list=None)
            model.optimize(n_optimization_restarts=self.n_optimization_restarts)
        else:
            raise NotImplementedError
        if save_dir is not None:
            if rank ==0:
                model.save(save_dir=save_dir)
            if MPI is not None:
                comm.Barrier()
        else:
            return model


class Hmf(BaseStatEmu):
    def __init__(self, data_dir, fid=['L2'], logging_level='INFO', no_merge=True, emu_type={'multi-fid':False, 'single-bin':False, 'linear':True, 'wide_and_narrow':True}):
        """
        emu_type : dict
            A dictionary with the emulator types. 
            linear : bool
                If True, use the linear emulator.
            multi-fid : bool
                If True, use the multi-fidelity emulator.
            wide_and_narrow : bool
                If True, use both wide and narrow simulations.
            mf-svgp : bool
        """
        self.logging_level = logging_level
        self.logger = self.configure_logging(logging_level)
        self.data_dir = data_dir
        self.no_merge = no_merge
        self.emu_type = emu_type
        self.X = []
        self.Y = []
        self.labels = []
        # Train on both Goku-wide and goku-narrow sims
        if emu_type['multi-fid']:
            fids = ['L2', 'HF']
        else:
            fids = ['L2']
        for fd in fids:
            # Goku-wide sims
            hmf = summary_stats.HMF(data_dir=data_dir, fid = fd,  narrow=False, no_merge=no_merge, logging_level=logging_level)
            # Trainthe spline coefficients
            Y, self.mbins = hmf.get_coeffs()
            # For now, get rid of the lastbins with 0 value
            Y_wide = Y[:, :-3]
            X_wide = hmf.get_params_array()
            labels_wide = hmf.get_labels()
            # Only use Goku-wide
            if not emu_type['wide_and_narrow']:
                self.Y.append(Y_wide)
                self.X.append(X_wide)
                self.labels.append(labels_wide)
            # Use both Goku-wide and narrow
            else:
                # Goku-narrow sims
                hmf = summary_stats.HMF(data_dir=data_dir, fid = fd,  narrow=True, no_merge=no_merge, logging_level=logging_level)
                # Trainthe spline coefficients
                Y, self.mbins = hmf.get_coeffs()
                # For now, get rid of the lastbins with 0 value
                self.Y.append(np.concatenate((Y_wide, Y[:, :-3]), axis=0))
                self.X.append(np.concatenate((X_wide, hmf.get_params_array()), axis=0))
                self.labels.append(np.concatenate((labels_wide, hmf.get_labels()), axis=0))
        if rank==0:
            self.logger.info(f'X: {len(self.X), np.array(self.X[0]).shape}, Y: {len(self.Y), np.array(self.Y[0]).shape}')

        super().__init__(X=self.X, Y=self.Y, logging_level=logging_level, emu_type=emu_type, n_optimization_restarts=5)
    

class _TypedStandardNormal(nf_distributions.normal.StandardNormal):
    """
    StandardNormal that samples with the buffer dtype so we avoid float/double mismatches.
    """
    def __init__(self, shape, dtype=torch.float32):
        super().__init__(shape=shape)
        # keep the log_z buffer in the requested dtype
        self._log_z.data = self._log_z.to(dtype=dtype)

    def _sample(self, num_samples, context):
        if context is None:
            return torch.randn(num_samples, *self._shape, device=self._log_z.device, dtype=self._log_z.dtype)
        context_size = context.shape[0]
        samples = torch.randn(context_size * num_samples, *self._shape,
                              device=context.device, dtype=self._log_z.dtype)
        return torchutils.split_leading_dim(samples, [context_size, num_samples])


class MultiFidelityNormalizingFlow:
    """
    Conditional normalizing flow that models HF and LF jointly by conditioning
    on the cosmological parameters plus a fidelity indicator.
    """

    def __init__(self, input_dim, output_dim, num_bijectors=4, hidden_units=(128, 128),
                 learning_rate=1e-3, name="mf_flow", device=None, dtype=torch.float32,
                 residual_flow=True, mean_hidden_units=None, mean_loss_weight=0.1):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_bijectors = num_bijectors
        self.hidden_units = hidden_units
        self.learning_rate = learning_rate
        self.residual_flow = residual_flow
        self.mean_loss_weight = mean_loss_weight
        self.mean_hidden_units = mean_hidden_units or hidden_units
        self.device = self._select_device(device)
        self.dtype = dtype
        self.loss_history = []
        self._build_flow()
        self._build_mean_net()
        self.optimizer = torch.optim.Adam(self._optimizer_parameters(), lr=self.learning_rate)

    def _select_device(self, device):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        resolved = torch.device(device)
        if resolved.type == "cuda" and not torch.cuda.is_available():
            print("Requested CUDA device but no GPU detected; falling back to CPU", flush=True)
            resolved = torch.device("cpu")
        return resolved

    def _build_flow(self):
        transforms = []
        for _ in range(self.num_bijectors):
            maf = nf_transforms.autoregressive.MaskedAffineAutoregressiveTransform(
                features=self.output_dim,
                hidden_features=self.hidden_units[0],
                context_features=self.input_dim,
                num_blocks=max(1, len(self.hidden_units))
            )
            transforms.append(maf)
            transforms.append(nf_transforms.permutations.ReversePermutation(features=self.output_dim))
        transform = nf_transforms.CompositeTransform(transforms)
        distribution = _TypedStandardNormal(shape=[self.output_dim], dtype=self.dtype)
        self.flow = nf_flows.base.Flow(transform, distribution).to(device=self.device, dtype=self.dtype)

    def _build_mean_net(self):
        """
        Lightweight deterministic head that predicts the conditional mean.
        The flow is then trained on residuals around this mean to avoid
        inflating variance when the conditional mean is off.
        """
        layers = []
        in_features = self.input_dim
        for width in self.mean_hidden_units:
            layers.append(torch.nn.Linear(in_features, width))
            layers.append(torch.nn.SiLU())
            in_features = width
        layers.append(torch.nn.Linear(in_features, self.output_dim))
        self.mean_net = torch.nn.Sequential(*layers).to(device=self.device, dtype=self.dtype)

    def _optimizer_parameters(self):
        params = list(self.flow.parameters())
        if self.mean_net is not None:
            params += list(self.mean_net.parameters())
        return params

    def _ensure_device_dtype(self):
        """Keep the flow parameters, buffers, and optimizer states on the configured device/dtype."""
        self.flow = self.flow.to(device=self.device, dtype=self.dtype)
        if self.mean_net is not None:
            self.mean_net = self.mean_net.to(device=self.device, dtype=self.dtype)
        # nflows stores masks as buffers; keep them in sync with the flow dtype
        for _, buf in self.flow.named_buffers():
            if torch.is_tensor(buf):
                if torch.is_floating_point(buf):
                    buf.data = buf.to(device=self.device, dtype=self.dtype)
                else:
                    # keep integer buffers (e.g., permutation indices) in their original dtype
                    buf.data = buf.to(device=self.device)
        for _, buf in self.mean_net.named_buffers(recurse=True):
            if torch.is_tensor(buf):
                if torch.is_floating_point(buf):
                    buf.data = buf.to(device=self.device, dtype=self.dtype)
                else:
                    buf.data = buf.to(device=self.device)
        for state in self.optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    if torch.is_floating_point(value):
                        state[key] = value.to(self.device, dtype=self.dtype)
                    else:
                        state[key] = value.to(self.device)

    def log_prob(self, y, context):
        return self.flow.log_prob(inputs=y, context=context)

    def _train_step(self, x_batch, y_batch):
        self._ensure_device_dtype()
        x_batch = x_batch.to(self.device, dtype=self.dtype, non_blocking=True)
        y_batch = y_batch.to(self.device, dtype=self.dtype, non_blocking=True)
        self.optimizer.zero_grad()
        if self.residual_flow:
            mean_pred = self.mean_net(x_batch)
            residual = y_batch - mean_pred
        else:
            mean_pred = None
            residual = y_batch
        nll = -self.log_prob(residual, x_batch).mean()
        mse_term = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        if self.residual_flow and self.mean_loss_weight > 0:
            mse_term = F.mse_loss(mean_pred, y_batch)
        loss = nll + self.mean_loss_weight * mse_term
        loss.backward()
        self.optimizer.step()
        return loss

    def fit(self, x, y, max_iters=2_000, batch_size=64, log_every=200,
            lr_scheduler=None, early_stopping=None):
        x_tensor = torch.as_tensor(x, dtype=self.dtype)
        y_tensor = torch.as_tensor(y, dtype=self.dtype)
        dataset = torch.utils.data.TensorDataset(x_tensor, y_tensor)
        pin_mem = self.device.type == 'cuda'
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False, pin_memory=pin_mem)
        iterator = iter(loader)
        scheduler = None
        scheduler_step_every = 1
        if lr_scheduler:
            sched_type = lr_scheduler.get("type", "plateau")
            scheduler_step_every = lr_scheduler.get("step_every", 1)
            if sched_type == "plateau":
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer,
                    mode="min",
                    factor=lr_scheduler.get("gamma", 0.5),
                    patience=lr_scheduler.get("patience", 200),
                    min_lr=lr_scheduler.get("min_lr", 1e-6)
                )
            elif sched_type == "step":
                scheduler = torch.optim.lr_scheduler.StepLR(
                    self.optimizer,
                    step_size=lr_scheduler.get("step_size", log_every),
                    gamma=lr_scheduler.get("gamma", 0.5)
                )
        best_loss = float("inf")
        steps_since_improvement = 0
        early_cfg = early_stopping or {}
        es_patience = early_cfg.get("patience")
        es_min_delta = early_cfg.get("min_delta", 0.0)
        for step in range(1, max_iters + 1):
            try:
                x_batch, y_batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                x_batch, y_batch = next(iterator)
            loss = self._train_step(x_batch, y_batch)
            loss_val = float(loss.detach().cpu().item())
            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(loss_val)
                elif step % scheduler_step_every == 0:
                    scheduler.step()
            if loss_val + es_min_delta < best_loss:
                best_loss = loss_val
                steps_since_improvement = 0
            else:
                steps_since_improvement += 1
                if es_patience is not None and steps_since_improvement >= es_patience:
                    print(f"Early stopping at step {step}: no improvement for {es_patience} steps", flush=True)
                    break
            if step % log_every == 0:
                current_lr = self.optimizer.param_groups[0]["lr"]
                print(f"Step {step}/{max_iters}, Loss: {loss_val:.4f}, lr: {current_lr:.2e}", flush=True)
                self.loss_history.append(loss_val)
        return self.loss_history

    def sample(self, context, num_samples=200):
        with torch.no_grad():
            self._ensure_device_dtype()
            context = context.to(self.device, dtype=self.dtype, non_blocking=True)
            residual_samples = self.flow.sample(num_samples=num_samples, context=context)
            if self.residual_flow:
                mean_pred = self.mean_net(context).unsqueeze(1)
                samples = residual_samples + mean_pred
            else:
                samples = residual_samples
        return samples.to(dtype=self.dtype).detach().cpu().numpy()

    def save(self, prefix):
        os.makedirs(op.dirname(prefix), exist_ok=True)
        path = f"{prefix}.pt"
        torch.save(
            {
                "state_dict": self.flow.state_dict(),
                "mean_state_dict": self.mean_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "loss_history": self.loss_history,
                "dtype": self.dtype,
                "residual_flow": self.residual_flow,
                "mean_hidden_units": self.mean_hidden_units,
                "mean_loss_weight": self.mean_loss_weight,
            },
            path,
        )
        return path

    def restore(self, prefix):
        ckpt_path = prefix if prefix.endswith(".pt") else f"{prefix}.pt"
        if not op.exists(ckpt_path):
            return None
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        self.dtype = checkpoint.get("dtype", self.dtype)
        self.flow.load_state_dict(checkpoint["state_dict"])
        if "mean_state_dict" in checkpoint:
            self.mean_net.load_state_dict(checkpoint["mean_state_dict"])
        self.residual_flow = checkpoint.get("residual_flow", self.residual_flow)
        self.mean_hidden_units = checkpoint.get("mean_hidden_units", self.mean_hidden_units)
        self.mean_loss_weight = checkpoint.get("mean_loss_weight", self.mean_loss_weight)
        try:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        except Exception as exc:
            print(f"Optimizer restore failed ({exc}); reinitializing", flush=True)
            self.optimizer = torch.optim.Adam(self._optimizer_parameters(), lr=self.learning_rate)
        for state in self.optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(self.device, dtype=self.dtype)
        self.loss_history = checkpoint.get("loss_history", [])
        self._ensure_device_dtype()
        return ckpt_path

    def to_device(self, device):
        """Move flow and optimizer state to the requested device."""
        target = self._select_device(device)
        if target == self.device:
            return self.device
        self.device = target
        self._ensure_device_dtype()
        return self.device


class BaseMFCoregEmu():
    """
    Emulator for the Halo Mass Function using the native bins
    This now models the joint LF/HF distribution with a conditional
    normalizing flow instead of two separate GPs for the mean and variance.
    """
    def __init__(self, DataLoader, data_dir, z, emu_type={'wide_and_narrow':True}, norm_type='subtract_mean', noise_floor=0.0, get_counts=False, logging_level='INFO', device=None):
        """
        Parameters
        ----------
        data_dir : str
            The directory where the data is stored.
        mass_pair : tuple
            The mass pair for which the correlation function is to be emulated.
        interp : str
            If 'spline', interpolate the nan values in the correlation function
            using a spline. Else, remove the sims with even a single nan values.
        emu_type : dict
            wide_and_narrow : bool
                If True, use both wide and narrow simulations.
        norm_type : str
            The type of normalization to be applied to the data.
            'subtract_mean' : subtract the mean of the LF and let
            the MF GP match the HF mean.
            'std_gaussian' : normalize each bin to have mean 0 and std 1. Mean
            and std are calculated based on the LF sims. Both LF and HF sims
            are normalized using the same mean and std.
        logging_level : str
            The logging level. 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
        """
        self.logging_level = logging_level
        self.logger = self.configure_logging(logging_level)
        self.data_dir = data_dir
        self.z = z
        self.norm_type = norm_type
        self.noise_floor = noise_floor
        self.device = self._resolve_device(device)
        # Load the data
        self.X = []
        self.Y = []
        self.Y_err = []
        self.labels = []
        self.wide_array = np.array([])
        # Keeping the id for good HF sims
        self.good_sim_ids = []
        # Keep a copy of raw outputs for later denormalization / diagnostics
        self.Y_raw = None
        # Fix a few features of the emulator
        #emu_type.update({'multi-fid':True, 'single-bin':False, 'linear':True, 'mf-svgp':True})
        self.emu_type = emu_type
        fids = ['L2', 'HF']
        for fd in fids:
            # Goku-wide sims
            data_loader = DataLoader(data_dir=data_dir, fid =fd, z=z, narrow=False, no_merge=True, logging_level=logging_level)
            # Load xi((m1, m2), r) for wide
            self.mbins, Y_wide, err_wide, X_wide, labels_wide = data_loader.get_data(noise_floor=noise_floor, get_counts=get_counts)
            self.wide_array= np.append(self.wide_array, np.ones(Y_wide.shape[0]))
            self.logger.debug(f'Y_wide: {Y_wide.shape}')
            # Only use Goku-wide
            if not emu_type['wide_and_narrow']:
                self.Y.append(Y_wide)
                self.X.append(X_wide)
                self.Y_err.append(err_wide)
                self.labels.append(labels_wide)
            # Use both Goku-wide and narrow
            else:
                # Goku-narrow sims
                data_loader = DataLoader(data_dir=data_dir, fid = fd, z=z, narrow=True, no_merge=True, logging_level=logging_level)
                # Load xi((m1, m2), r) for wide
                _, Y_narrow, err_narrow, X_narrow, labels_narrow = data_loader.get_data(noise_floor=noise_floor, get_counts=get_counts)
                self.wide_array= np.append(self.wide_array, np.zeros(Y_narrow.shape[0]))
                self.logger.debug(f'Y_narrow: {Y_narrow.shape}')
                # For now, get rid of the lastbins with 0 value
                self.Y.append(np.concatenate((Y_wide, Y_narrow), axis=0))
                self.X.append(np.concatenate((X_wide, X_narrow), axis=0))
                self.Y_err.append(np.concatenate((err_wide, err_narrow), axis=0))
                self.labels.append(np.concatenate((labels_wide, labels_narrow), axis=0))
        # Keep raw (unnormalized) outputs for diagnostics
        self.Y_raw = [arr.copy() for arr in self.Y]
        # X is normalized between 0 and 1; Y is shifted by the LF median for both fidelities
        if norm_type == 'subtract_mean':
            self.logger.info('Normalizing X between 0 and 1, subtracting the LF median from both LF and HF Y')
            self.X, self.Y, self.X_min, self.X_max, self.lf_mean_func = self.normalize(self.X, self.Y)
        elif norm_type == 'std_gaussian':
            self.logger.info('Normalizing each bin to have mean 0 and std 1')
            self.X, self.Y, self.Y_err, self.X_min, self.X_max, self.mean_Y, self.std_Y = self.normalize_std_gaussian(self.X, self.Y, self.Y_err)
        self.output_dim = self.Y[0].shape[1]
        # Concatenate the errors to Y, so self.Y is a list of fidelities: [array([Y_wide ... err_wide]), array([Y_narrow ... err_narrow])]

        #self.Y[0] = np.concatenate((self.Y[0][:, :], Y_err[0][:,:]), axis=1)
        #self.Y[1] = np.concatenate((self.Y[1][:,:], Y_err[1][:,:]), axis=1)

        self.Y[0] = self.Y[0].astype(np.float32)
        self.Y[1] = self.Y[1].astype(np.float32)
        self.Y_err[0] = self.Y_err[0].astype(np.float32)
        self.Y_err[1] = self.Y_err[1].astype(np.float32)
        self.X[0] = self.X[0].astype(np.float32)
        self.X[1] = self.X[1].astype(np.float32)

        assert not np.isnan(self.X[0]).any(), f'X[0] has nans {np.where(np.isnan(self.X[0]))}'
        assert not np.isnan(self.X[1]).any(), f'X[1] has nans {np.where(np.isnan(self.X[1]))}'
        assert not np.isnan(self.Y[0]).any(), f'Y[0] has nans {np.where(np.isnan(self.Y[0]))}'
        assert not np.isnan(self.Y[1]).any(), f'Y[1] has nans {np.where(np.isnan(self.Y[1]))}'
        self.logger.debug(f'X: ({np.array(self.X[0]).shape}, {np.array(self.X[1]).shape}, Y: ({np.array(self.Y[0]).shape}, {np.array(self.Y[1]).shape}, Y_err: ({np.array(self.Y_err[0]).shape}, {np.array(self.Y_err[1]).shape})')
        self.logger.info(f'norm_type {norm_type}')
        self.logger.info(f'noise_floor {noise_floor}')
        # Normalizing flow gets built during training to keep config flexible
        self.flow_model = None
        self.flow_config = {}
        # Deterministic LF predictor (trained on LF only)
        self.lf_mean_net = None
        self.lf_mean_config = {}
        # Deterministic LF predictor (trained on LF only)
        self.lf_mean_net = None
        self.lf_mean_config = {}

    def configure_logging(self, logging_level):
        """Sets up logging based on the provided logging level in an MPI environment."""
        logger = logging.getLogger('BaseMFCoregEmu')
        logger.setLevel(logging_level)

        # Create a console handler with flushing
        console_handler = logging.StreamHandler(sys.stdout)

        # Include Rank, Logger Name, Timestamp, and Message in format
        formatter = logging.Formatter(
            f'%(name)s | %(asctime)s | Rank {rank} | %(levelname)s  |  %(message)s',
            datefmt='%m/%d/%Y %I:%M:%S %p'
        )
        console_handler.setFormatter(formatter)

        # Ensure logs flush immediately
        console_handler.flush = sys.stdout.flush  

        # Add handler to logger
        logger.addHandler(console_handler)
        
        return logger

    def _resolve_device(self, device):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        resolved = torch.device(device)
        if resolved.type == 'cuda' and not torch.cuda.is_available():
            self.logger.warning('CUDA requested but no GPU available; falling back to CPU')
            resolved = torch.device('cpu')
        self.logger.info(f'Using torch device {resolved}')
        return resolved

    def _lf_dtype(self):
        """Torch dtype matching the LF inputs."""
        return torch.as_tensor(self.X[0]).dtype
    
    def normalize(self, X, Y):
        """
        Normalize all input, X, such it is between 0 and 1
        Subtract the output, Y, by the LF median for both LF and HF.

        Returns:
        --------
        X_normalized: normalized input data between 0 and 1
        X_min: minimum value of the input data
        X_max: maximum value of the input data
        mean_func: the mean of the output to be used as the mean function
        in the GP model
        """
        X_min, X_max = np.min(X[0], axis=0), np.max(X[0], axis=0)
        X_normalized = []
        for i in range(len(X)):
            X_normalized.append((X[i]-X_min)/(X_max-X_min))
        Y_normalized = []
        # The zeros row is the LF
        lf_median_func =  np.nanmedian(Y[0], axis=0)
        Y_normalized.append(Y[0] -lf_median_func)
        # Apply the same shift to HF to keep both fidelities aligned
        Y_normalized.append(Y[1] - lf_median_func)

        return X_normalized, Y_normalized, X_min, X_max, lf_median_func
    
    def normalize_std_gaussian(self, X, Y, Y_err):
        """
        Normalize all input, X, such it is between 0 and 1
        Normalize Y at each bin to have mean 0 and std 1 -- should
        help with forcing the GP to spend similar focus on all bins

        """
        X_min, X_max = np.min(X[0], axis=0), np.max(X[0], axis=0)
        X_normalized = []
        for i in range(len(X)):
            X_normalized.append((X[i]-X_min)/(X_max-X_min))
        
        Y_normalized = []
        Y_err_normalized = []
        # We have more LF sims, so normalize based on LF
        mean = np.mean(Y[0], axis=0)
        std = np.std(Y[0], axis=0)
        for i in range(len(Y)):
            Y_normalized.append((Y[i] - mean) / std)
            Y_err_normalized.append(Y_err[i] / std)

        return X_normalized, Y_normalized, Y_err_normalized, X_min, X_max, mean, std

    def train(self, ind_train=None, ind_test=None, model_file='Xi_Native_emu_mapirs2.pkl', opt_params={}, force_train=True, train_subdir = 'train', composite_kernel=None, w_type='diagonal', loss_type='gaussian'):
        """
        Train the LF deterministic model on LF only, then train an additive residual flow on HF residuals (Y_HF - Y_LF_pred).
        Parameters
        ----------
        model_file : str
            The file prefix to save the Emulator checkpoints and attributes.
        opt_params : dict
            flow training options. Keys supported:
              - max_iters (int)
              - initial_lr (float)
              - batch_size (int)
              - num_bijectors (int)
              - hidden_units (tuple)
              - log_every (int)
              - lr_scheduler_type (str: 'plateau' or 'step')
              - lr_scheduler_gamma (float)
              - lr_scheduler_step (int)
              - lr_scheduler_patience (int)
              - lr_scheduler_min_lr (float)
              - lr_scheduler_step_every (int)
              - early_stopping_patience (int)
              - early_stopping_min_delta (float)
              - lf_mean_hidden_units (tuple)
              - lf_mean_max_iters (int)
              - lf_mean_lr (float)
              - lf_mean_batch_size (int)
              - residual_flow (bool; learn deterministic mean then flow on residuals)
              - mean_loss_weight (float; weight for auxiliary MSE on the mean head)
              - mean_hidden_units (tuple; hidden sizes for mean head)
        """
        if ind_train is None:
            ind_train = np.arange(self.X[1].shape[0])
        # Train LF deterministic model on LF only
        lf_mean_hidden_units = opt_params.get('lf_mean_hidden_units', (128, 128))
        lf_mean_max_iters = opt_params.get('lf_mean_max_iters', 30000)
        lf_mean_lr = opt_params.get('lf_mean_lr', 1e-3)
        lf_mean_batch_size = opt_params.get('lf_mean_batch_size', 128)
        lf_mean_dropout = opt_params.get('lf_mean_dropout', 0.0)
        lf_es_patience = opt_params.get('lf_mean_early_stopping_patience')
        lf_es_min_delta = opt_params.get('lf_mean_early_stopping_min_delta', 0.0)
        lf_sched_type = opt_params.get('lf_mean_scheduler_type', 'plateau')
        lf_sched_gamma = opt_params.get('lf_mean_scheduler_gamma', 0.5)
        lf_sched_patience = opt_params.get('lf_mean_scheduler_patience', 200)
        lf_sched_min_lr = opt_params.get('lf_mean_scheduler_min_lr', 1e-6)
        self.logger.info(f'Training LF mean net on LF data: hidden_units={lf_mean_hidden_units}, max_iters={lf_mean_max_iters}, lr={lf_mean_lr}, batch_size={lf_mean_batch_size}, dropout={lf_mean_dropout}, sched={lf_sched_type}')
        self._train_lf_mean_net(max_iters=lf_mean_max_iters, lr=lf_mean_lr, batch_size=lf_mean_batch_size, log_every=200,
                                hidden_units=lf_mean_hidden_units, scheduler_type=lf_sched_type,
                                scheduler_gamma=lf_sched_gamma, scheduler_patience=lf_sched_patience,
                                scheduler_min_lr=lf_sched_min_lr, early_stopping_patience=lf_es_patience,
                                early_stopping_min_delta=lf_es_min_delta, dropout=lf_mean_dropout)

        # Prepare HF residuals: Y_HF - Y_LF_pred(X_HF)
        with torch.no_grad():
            x_hf_tensor = torch.as_tensor(self.X[1][ind_train], device=self.device, dtype=self.lf_mean_net[0].weight.dtype)
            lf_pred_on_hf = self.lf_mean_net(x_hf_tensor).cpu().numpy()
        residuals = (self.Y[1][ind_train] - lf_pred_on_hf).astype(np.float32)
        X_train = self.X[1][ind_train].astype(np.float32)
        self.logger.debug(f'X_train (HF only): {X_train.shape}, residuals: {residuals.shape}')

        checkpoint_dir = op.join(self.data_dir, train_subdir)
        os.makedirs(checkpoint_dir, exist_ok=True)
        ckpt_prefix = op.join(checkpoint_dir, model_file)

        max_iters = opt_params.get('max_iters', 4_000)
        initial_lr = opt_params.get('initial_lr', 5e-3)
        batch_size = opt_params.get('batch_size', 64)
        num_bijectors = opt_params.get('num_bijectors', 4)
        hidden_units = opt_params.get('hidden_units', (128, 128))
        log_every = opt_params.get('log_every', 200)
        num_samples = opt_params.get('num_samples', 256)
        lr_scheduler_type = opt_params.get('lr_scheduler_type')
        lr_scheduler_gamma = opt_params.get('lr_scheduler_gamma', 0.5)
        lr_scheduler_step = opt_params.get('lr_scheduler_step', log_every)
        lr_scheduler_patience = opt_params.get('lr_scheduler_patience', 200)
        lr_scheduler_min_lr = opt_params.get('lr_scheduler_min_lr', 1e-6)
        lr_scheduler_step_every = opt_params.get('lr_scheduler_step_every', 1)
        early_stopping_patience = opt_params.get('early_stopping_patience')
        early_stopping_min_delta = opt_params.get('early_stopping_min_delta', 0.0)
        residual_flow = opt_params.get('residual_flow', False)  # we already remove LF mean; flow sees residuals only
        mean_loss_weight = opt_params.get('mean_loss_weight', 0.0)
        mean_hidden_units = opt_params.get('mean_hidden_units', hidden_units)

        lr_scheduler_cfg = None
        if lr_scheduler_type is not None:
            lr_scheduler_cfg = {
                'type': lr_scheduler_type,
                'gamma': lr_scheduler_gamma,
                'step_size': lr_scheduler_step,
                'patience': lr_scheduler_patience,
                'min_lr': lr_scheduler_min_lr,
                'step_every': lr_scheduler_step_every
            }
        early_stop_cfg = None
        if early_stopping_patience is not None:
            early_stop_cfg = {
                'patience': early_stopping_patience,
                'min_delta': early_stopping_min_delta
            }

        self.flow_config = dict(
            num_bijectors=num_bijectors,
            hidden_units=hidden_units,
            learning_rate=initial_lr,
            num_samples=num_samples,
            lr_scheduler=lr_scheduler_cfg,
            early_stopping=early_stop_cfg,
            residual_flow=residual_flow,
            mean_loss_weight=mean_loss_weight,
            mean_hidden_units=mean_hidden_units
        )
        if self.flow_model is None:
            self.flow_model = MultiFidelityNormalizingFlow(
                input_dim=X_train.shape[1],
                output_dim=self.output_dim,
                num_bijectors=num_bijectors,
                hidden_units=hidden_units,
                learning_rate=initial_lr,
                device=self.device,
                residual_flow=residual_flow,
                mean_hidden_units=mean_hidden_units,
                mean_loss_weight=mean_loss_weight
            )
        else:
            self.logger.info('Reusing existing flow model instance')
            self.flow_model.to_device(self.device)
        # Try to restore if a checkpoint exists
        ckpt_path = f"{ckpt_prefix}.pt"
        if op.exists(ckpt_path):
            self.logger.info(f'Loading flow checkpoint from {ckpt_path}')
            self.flow_model.restore(ckpt_path)

        self.logger.info(f'Built flow with output_dim {self.output_dim}, num_bijectors {num_bijectors}, hidden_units {hidden_units}')
        self.logger.info(f'flow residual_flow={residual_flow}, mean_loss_weight={mean_loss_weight}, mean_hidden_units={mean_hidden_units}')
        self.logger.info(f'flow batch_size {batch_size}, max_iters {max_iters}, log_every {log_every}')
        if lr_scheduler_cfg is not None:
            self.logger.info(f"Using lr scheduler {lr_scheduler_cfg.get('type')} with gamma={lr_scheduler_cfg.get('gamma')}, step_size={lr_scheduler_cfg.get('step_size')}, patience={lr_scheduler_cfg.get('patience')}")
        if early_stop_cfg is not None:
            self.logger.info(f"Early stopping enabled with patience={early_stop_cfg.get('patience')}, min_delta={early_stop_cfg.get('min_delta')}")

        if force_train:
            self.logger.debug(f'Training flow on shapes {X_train.shape}, {residuals.shape}')
            history = self.flow_model.fit(
                x=X_train,
                y=residuals,
                max_iters=max_iters,
                batch_size=batch_size,
                log_every=log_every,
                lr_scheduler=lr_scheduler_cfg,
                early_stopping=early_stop_cfg
            )
            ckpt_path = self.flow_model.save(ckpt_prefix)
            with open(f'{ckpt_prefix}.attrs', 'wb') as f:
                self.logger.debug(f'Writing flow attrs to {ckpt_prefix}.attrs')
                self.model_attrs = {
                    'loss_history': history,
                    'emu_type': self.emu_type,
                    'flow_config': self.flow_config,
                    'num_samples': num_samples,
                    'lf_mean_config': self.lf_mean_config
                }
                pickle.dump(self.model_attrs, f)
            self.logger.info(f'done with optimization {max_iters}, saved {ckpt_path}')

    def predict(self, ind_test, model_file, train_subdir = 'train', num_samples=None):
        """
        Posterior prediction of the emulator using the trained normalizing flow.
        Returns mean and variance estimated from flow samples.
        """
        t0 = time.time()
        base_name = op.splitext(model_file)[0]
        attr_file = op.join(self.data_dir, train_subdir, f'{base_name}.attrs')
        legacy_attr_file = op.join(self.data_dir, train_subdir, f'{model_file}.attrs')
        try:
            with open(attr_file, 'rb') as f:
                self.model_attrs = pickle.load(f)
        except Exception:
            try:
                with open(legacy_attr_file, 'rb') as f:
                    self.model_attrs = pickle.load(f)
                    self.logger.info(f'Loaded legacy attrs from {legacy_attr_file}')
            except Exception:
                self.logger.warning(f'No model attributes found for {attr_file}')
                self.model_attrs = {}
        print(f'[timer] predict: load attrs {time.time() - t0:.2f}s', flush=True)

        flow_cfg = self.model_attrs.get('flow_config', self.flow_config)
        num_bijectors = flow_cfg.get('num_bijectors', 4) if flow_cfg is not None else 4
        hidden_units = flow_cfg.get('hidden_units', (128, 128)) if flow_cfg is not None else (128, 128)
        learning_rate = flow_cfg.get('learning_rate', 5e-3) if flow_cfg is not None else 5e-3
        residual_flow = flow_cfg.get('residual_flow', True) if flow_cfg is not None else True
        mean_loss_weight = flow_cfg.get('mean_loss_weight', 0.1) if flow_cfg is not None else 0.1
        mean_hidden_units = flow_cfg.get('mean_hidden_units', hidden_units) if flow_cfg is not None else hidden_units
        num_samples = self.model_attrs.get('num_samples', 256) if num_samples is None else num_samples
        lf_mean_cfg = self.model_attrs.get('lf_mean_config', {'hidden_units': (128, 128)})

        t1 = time.time()
        if self.flow_model is None:
            self.flow_model = MultiFidelityNormalizingFlow(
                input_dim=self.X[0].shape[1],
                output_dim=self.output_dim,
                num_bijectors=num_bijectors,
                hidden_units=hidden_units,
                learning_rate=learning_rate,
                device=self.device,
                residual_flow=residual_flow,
                mean_hidden_units=mean_hidden_units,
                mean_loss_weight=mean_loss_weight
            )
        else:
            self.flow_model.to_device(self.device)
        print(f'[timer] predict: build/restore model {time.time() - t1:.2f}s (device={self.device})', flush=True)

        t2 = time.time()
        ckpt_prefix = op.join(self.data_dir, train_subdir, base_name)
        ckpt_path = f"{ckpt_prefix}.pt"
        if not op.exists(ckpt_path):
            self.logger.warning(f'No flow checkpoint found at {ckpt_path}; predictions will use current (possibly untrained) weights')
        else:
            self.flow_model.restore(ckpt_path)
        print(f'[timer] predict: load checkpoint {time.time() - t2:.2f}s', flush=True)

        t3 = time.time()
        # HF inputs only; first predict LF mean, then add sampled residuals
        X_test = self.X[1][ind_test].astype(np.float32)
        context = torch.as_tensor(X_test, dtype=self.flow_model.dtype)
        if self.lf_mean_net is None:
            self.logger.info('Rebuilding LF mean net for prediction')
            self.lf_mean_net = self._build_lf_mean_net(hidden_units=lf_mean_cfg.get('hidden_units', (128, 128)))
        self.lf_mean_net = self.lf_mean_net.to(device=self.device, dtype=self.flow_model.dtype)
        with torch.no_grad():
            lf_pred = self.lf_mean_net(context.to(self.device)).cpu().numpy()
        samples = self.flow_model.sample(context, num_samples=num_samples)
        samples = samples + lf_pred[:, None, :]
        print(f'samples.shape: {samples.shape}', flush=True)
        mean_pred = np.mean(samples, axis=1).squeeze()
        var_pred = np.var(samples, axis=1).squeeze()
        print(f'[timer] predict: sampling + moments {time.time() - t3:.2f}s (num_samples={num_samples})', flush=True)

        if self.norm_type == 'subtract_mean':
            mean_pred += self.lf_mean_func
        if self.norm_type == 'std_gaussian':
            mean_pred = mean_pred * self.std_Y + self.mean_Y
            var_pred = var_pred * (self.std_Y ** 2)
        return mean_pred, var_pred

    def denormalize_outputs(self, Y_list):
        """
        Undo normalization for a list of outputs (LF, HF) or a single array.
        Useful when comparing predictions against stored normalized targets.
        """
        def _denorm(arr):
            if self.norm_type == 'subtract_mean':
                return arr + self.lf_mean_func
            if self.norm_type == 'std_gaussian':
                return arr * self.std_Y + self.mean_Y
            return arr

        if isinstance(Y_list, list):
            return [_denorm(arr) for arr in Y_list]
        return _denorm(Y_list)

    def find_hf_in_lf(self):
        """
        Find indices of LF simulations that exactly match each HF simulation in X-space.
        Returns an array of length n_hf with LF indices; raises if a match is missing.
        """
        lf_map = {tuple(row.tolist()): idx for idx, row in enumerate(self.X[0])}
        hf_to_lf = []
        for row in self.X[1]:
            key = tuple(row.tolist())
            if key not in lf_map:
                raise ValueError("HF simulation not found in LF set; cannot build residual mapping")
            hf_to_lf.append(lf_map[key])
        return np.array(hf_to_lf, dtype=int)

    # ------------------------------------------------------------------
    # LF deterministic model (trained on LF only)
    # ------------------------------------------------------------------
    def _build_lf_mean_net(self, hidden_units=(128, 128), dropout=0.0):
        layers = []
        in_features = self.X[0].shape[1]
        for width in hidden_units:
            layers.append(nn.Linear(in_features, width))
            layers.append(nn.SiLU())
            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_features = width
        layers.append(nn.Linear(in_features, self.output_dim))
        net = nn.Sequential(*layers).to(device=self.device, dtype=self._lf_dtype())
        return net

    def _train_lf_mean_net(self, max_iters=2000, lr=1e-3, batch_size=128, log_every=200, hidden_units=(128, 128),
                           scheduler_type='plateau', scheduler_gamma=0.5, scheduler_patience=200, scheduler_min_lr=1e-6,
                           early_stopping_patience=None, early_stopping_min_delta=0.0, dropout=0.0):
        if self.lf_mean_net is None:
            self.lf_mean_net = self._build_lf_mean_net(hidden_units, dropout=dropout)
        else:
            self.lf_mean_net = self.lf_mean_net.to(device=self.device, dtype=self._lf_dtype())
        self.lf_mean_config = dict(hidden_units=hidden_units, max_iters=max_iters, lr=lr, batch_size=batch_size, log_every=log_every,
                                   scheduler_type=scheduler_type, scheduler_gamma=scheduler_gamma, scheduler_patience=scheduler_patience,
                                   scheduler_min_lr=scheduler_min_lr, early_stopping_patience=early_stopping_patience,
                                   early_stopping_min_delta=early_stopping_min_delta, dropout=dropout)
        x = torch.as_tensor(self.X[0], dtype=self.lf_mean_net[0].weight.dtype)
        y = torch.as_tensor(self.Y[0], dtype=self.lf_mean_net[0].weight.dtype)
        dataset = torch.utils.data.TensorDataset(x, y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False, pin_memory=self.device.type == 'cuda')
        optimizer = torch.optim.Adam(self.lf_mean_net.parameters(), lr=lr)
        scheduler = None
        if scheduler_type == 'plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=scheduler_gamma,
                patience=scheduler_patience,
                min_lr=scheduler_min_lr
            )
        best_loss = float('inf')
        steps_since_improve = 0
        iterator = iter(loader)
        for step in range(1, max_iters + 1):
            try:
                xb, yb = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                xb, yb = next(iterator)
            xb = xb.to(self.device)
            yb = yb.to(self.device)
            optimizer.zero_grad()
            pred = self.lf_mean_net(xb)
            loss = F.mse_loss(pred, yb)
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step(loss.item())
            if loss.item() + early_stopping_min_delta < best_loss:
                best_loss = loss.item()
                steps_since_improve = 0
            else:
                steps_since_improve += 1
            if early_stopping_patience is not None and steps_since_improve >= early_stopping_patience:
                self.logger.info(f'LF mean net early stop at step {step}, best_loss {best_loss:.4e}, lr = {optimizer.param_groups[0]["lr"]:.2e}')
                break
            if log_every and step % log_every == 0:
                self.logger.info(f'LF mean net step {step}/{max_iters}, loss {loss.item():.4e}')
        return self.lf_mean_net

class HmfNativeBins(BaseMFCoregEmu):
    """
    Emulator for the Halo Mass Function using the native bins
    This does the full dimensionality reduction of the output space using
    the conditional normalizing flow defined in BaseMFCoregEmu.
    """

    def __init__(self, data_dir, z, emu_type={ 'wide_and_narrow': True }, norm_type='subtract_mean', noise_floor=0.0, get_counts=False, logging_level='INFO'):
        
        DataLoader = summary_stats.HMF
        super().__init__(DataLoader, data_dir, z, emu_type=emu_type, norm_type=norm_type, noise_floor=noise_floor, get_counts=get_counts, logging_level=logging_level)

class XiNativeBins(BaseMFCoregEmu):
    """
    Emulator for the Correlation Function, xi(r, m1, m2) using the native bins
    This does the full dimensionality reduction of the output space using
    the conditional normalizing flow defined in BaseMFCoregEmu.
    """

    def __init__(self, data_dir, z, emu_type={ 'wide_and_narrow': True }, norm_type='subtract_mean', noise_floor=0.0, get_counts=False, logging_level='INFO'):
        
        DataLoader = summary_stats.Xi
        super().__init__(DataLoader, data_dir, z, emu_type=emu_type, norm_type=norm_type, noise_floor=noise_floor, get_counts=get_counts, logging_level=logging_level)

class XiNativeBinsFullDimReduc():
    """
    Emulator for the Correlation Function, xi(r, n1, n2) using the native bins
    This does the full dimensionality reduction of the output space using
    `LatentMFCoregionalizationSVGP` which allows each output to have a different
    observational (simualtion quality) uncertainty.
    """
    def __init__(self, data_dir, num_latents, num_inducing, noise_num_latents=None,
                 use_rho=True, emu_type={'wide_and_narrow':True}, 
                 logging_level='INFO'):
        """
        Parameters
        ----------
        data_dir : str
            The directory where the data is stored.
        mass_pair : tuple
            The mass pair for which the correlation function is to be emulated.
        interp : str
            If 'spline', interpolate the nan values in the correlation function
            using a spline. Else, remove the sims with even a single nan values.
        emu_type : dict
            wide_and_narrow : bool
                If True, use both wide and narrow simulations.
        remove_sims : list
            A list of simulation indices to remove from the training/test set.
        logging_level : str
            The logging level. 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
        """
        self.logging_level = logging_level
        self.logger = self.configure_logging(logging_level)
        self.data_dir = data_dir
        self.num_latents = num_latents
        self.num_inducing = num_inducing
        self.use_rho = use_rho
        # Laod the data
        self.X = []
        self.Y = []
        self.Y_err = []
        self.labels = []
        self.wide_array = np.array([])
        # Keeping the id for good HF sims
        self.good_sim_ids = []
        # Fix a few features of the emulator
        emu_type.update({'multi-fid':True, 'single-bin':False, 'linear':True, 'mf-svgp':True})
        self.emu_type = emu_type
        fids = ['L2', 'HF']
        for fd in fids:
            # Goku-wide sims
            xi = summary_stats.Xi(data_dir=data_dir, fid = fd,  narrow=False, MPI=None, logging_level=logging_level)
            # Load xi((m1, m2), r) for wide
            self.mbins, Y_wide, err_wide, X_wide, labels_wide = xi.get_wt_err(rcut=(0.2, 61))
            self.wide_array= np.append(self.wide_array, np.ones(Y_wide.shape[0]))
            self.logger.debug(f'Y_wide: {Y_wide.shape}')
            # Only use Goku-wide
            if not emu_type['wide_and_narrow']:
                self.Y.append(Y_wide)
                self.X.append(X_wide)
                self.Y_err.append(err_wide)
                #self.labels.append(labels_wide)
            # Use both Goku-wide and narrow
            else:
                # Goku-narrow sims
                xi = summary_stats.Xi(data_dir=data_dir, fid = fd,  narrow=True, MPI=None, logging_level=logging_level)
                # Load xi((m1, m2), r) for wide
                _, Y_narrow, err_narrow, X_narrow, labels_narrow = xi.get_wt_err(rcut=(0.2, 61))
                self.wide_array= np.append(self.wide_array, np.zeros(Y_narrow.shape[0]))
                self.logger.debug(f'Y_narrow: {Y_narrow.shape}')
                # For now, get rid of the lastbins with 0 value
                self.Y.append(np.concatenate((Y_wide, Y_narrow), axis=0))
                self.X.append(np.concatenate((X_wide, X_narrow), axis=0))
                self.Y_err.append(np.concatenate((err_wide, err_narrow), axis=0))
                self.labels.append(np.concatenate((labels_wide, labels_narrow), axis=0))
        # X is normalized between 0 and 1, but for Y only HF fideliy is not normalized
        # the MF GP will match the HF mean
        self.X, self.Y, self.X_min, self.X_max = self.normalize(self.X, self.Y)
        self.output_dim = self.Y[0].shape[1]
        # Concatenate the errors to Y, so self.Y is a list of fidelities: [array([Y_wide ... err_wide]), array([Y_narrow ... err_narrow])]

        #self.Y[0] = np.concatenate((self.Y[0][:, :], Y_err[0][:,:]), axis=1)
        #self.Y[1] = np.concatenate((self.Y[1][:,:], Y_err[1][:,:]), axis=1)

        self.Y[0] = self.Y[0].astype(np.float32)
        self.Y[1] = self.Y[1].astype(np.float32)
        self.Y_err[0] = self.Y_err[0].astype(np.float32)
        self.Y_err[1] = self.Y_err[1].astype(np.float32)
        self.X[0] = self.X[0].astype(np.float32)
        self.X[1] = self.X[1].astype(np.float32)

        assert not np.isnan(self.X[0]).any(), f'X[0] has nans {np.where(np.isnan(self.X[0]))}'
        assert not np.isnan(self.X[1]).any(), f'X[1] has nans {np.where(np.isnan(self.X[1]))}'
        assert not np.isnan(self.Y[0]).any(), f'Y[0] has nans {np.where(np.isnan(self.Y[0]))}'
        assert not np.isnan(self.Y[1]).any(), f'Y[1] has nans {np.where(np.isnan(self.Y[1]))}'

    def configure_logging(self, logging_level):
        """Sets up logging based on the provided logging level in an MPI environment."""
        logger = logging.getLogger('XiNativeBinsFullDimReduc')
        logger.setLevel(logging_level)

        # Create a console handler with flushing
        console_handler = logging.StreamHandler(sys.stdout)

        # Include Rank, Logger Name, Timestamp, and Message in format
        formatter = logging.Formatter(
            f'%(name)s | %(asctime)s | Rank {rank} | %(levelname)s  |  %(message)s',
            datefmt='%m/%d/%Y %I:%M:%S %p'
        )
        console_handler.setFormatter(formatter)

        # Ensure logs flush immediately
        console_handler.flush = sys.stdout.flush  

        # Add handler to logger
        logger.addHandler(console_handler)
        
        return logger
    def normalize(self, X, Y):
        """
        Normalize all input, X, such it is between 0 and 1
        Subtract the output, Y, by it's mean, only for LF
        and leave the HF uncrouched

        Returns:
        --------
        X_normalized: normalized input data between 0 and 1
        X_min: minimum value of the input data
        X_max: maximum value of the input data
        mean_func: the mean of the output to be used as the mean function
        in the GP model
        """
        X_min, X_max = np.min(X[0], axis=0), np.max(X[0], axis=0)
        X_normalized = []
        for i in range(len(X)):
            X_normalized.append((X[i]-X_min)/(X_max-X_min))
        Y_normalized = []
        # The zeros row is the LF
        lf_median_func =  np.nanmedian(Y[0], axis=0)
        Y_normalized.append(Y[0] - lf_median_func)
        Y_normalized.append(Y[1])

        return X_normalized, Y_normalized, X_min, X_max
    
    def train(self, ind_train=None, model_file='Xi_Native_emu_mapirs2.pkl', opt_params={}, force_train=True, train_subdir = 'train'):
        """
        Train the model and save this in `model_file`
        Parameters
        ----------
        model_file : str
            The file to save the Emulator. Two files
            will be saved, one with the model and the other
            with the loss history.
        """
        if ind_train is None:
            ind_train = np.arange(self.X[1].shape[0])

        # Also subtract median from HF sims, I had noticed the f_HF - f_LF
        # is < 3% for all 36 LF-HF pairs, it gets slightly larger 
        # closer to r =60 CMpc/h, but still similar for all smulations

        # We subtract the median of the HF sims for training if use_rho is False
        if not self.use_rho:
            self.hf_median_func = np.nanmedian(self.Y[1][ind_train], axis=0)

        # Add the fidelity indocators, 0 for L2 and 1 for HF
        X_l2_aug = np.hstack([self.X[0], np.zeros((self.X[0].shape[0], 1), dtype=np.float32)])
        X_hf_aug = np.hstack([self.X[1][ind_train], np.ones((ind_train.size, 1), dtype=np.float32)])
        # Stack the L2 and HF data vertically
        X_train = np.vstack([X_l2_aug, X_hf_aug])
        if self.use_rho:
            Y_train = np.vstack([self.Y[0], self.Y[1][ind_train]])
        else:
            # We subtract the median of the HF sims for training if use_rho is False
            Y_train = np.vstack([self.Y[0], self.Y[1][ind_train] - self.hf_median_func])
        self.logger.debug(f'X_train: {X_train.shape}, Y_train: {Y_train.shape}')

        X_train = X_train.astype(np.float32)
        Y_train = Y_train.astype(np.float32)

        # Base kernel of the MF GP
        kernel_L = gpflow.kernels.SquaredExponential(lengthscales=np.ones(X_train.shape[1]-1,  dtype=np.float32), variance=np.float32(1.0))
        kernel_delta = gpflow.kernels.SquaredExponential(lengthscales=np.ones(X_train.shape[1]-1,  dtype=np.float32), variance=np.float32(1.0))
        self.emu = LatentMFCoregionalizationSVGP(
            X_train, Y_train, kernel_L, kernel_delta,
            num_latents=self.num_latents, num_inducing=self.num_inducing,
            num_outputs=self.output_dim, heterosed=True, use_rho=self.use_rho)
        
        model_file = op.join(self.data_dir, train_subdir, model_file)
        if op.exists(model_file):
            self.logger.info(f'Loading model from {model_file}')
            with open(model_file, "rb") as f:
                params = pickle.load(f)
                # TODO: Save the model already in float32:
                # Convert all parameters to float32 type
                # This won't be necessary if the saved model is already in float32
                for key, value in params.items():
                    if isinstance(value, dict):
                        for inner_key, inner_value in value.items():
                            if isinstance(inner_value, np.ndarray):
                                params[key][inner_key] = inner_value.astype(np.float32)
                            elif isinstance(inner_value, (int, float)):
                                params[key][inner_key] = np.float32(inner_value)
                    elif isinstance(value, np.ndarray):
                        params[key] = value.astype(np.float32)
                    elif isinstance(value, (int, float)):
                        params[key] = np.float32(value)
                gpflow.utilities.multiple_assign(self.emu, params)
            # load the loss_history:
            try:
                with open(f'{model_file}.attrs', 'rb') as f:
                    attrs = pickle.load(f)
                    # Reload the loss history, so it will be appended
                    # during the new training
                    self.emu.loss_history = attrs['loss_history']
                    current_iters = len(self.emu.loss_history)
            except:
                current_iters = None
                self.logger.warning(f'No loss history found for {model_file}.attrs, but model exists')
        else:

            current_iters = 0
        # Log the model specifications
        self.logger.info(f'Built the model with')
        self.logger.info(f'#num_latents {self.num_latents}')
        self.logger.info(f'output_dim {self.output_dim}')
        self.logger.info(f'num_inducing {self.num_inducing}')
        self.logger.info(f'varaince dim {self.emu.likelihood.variance.numpy().shape}')
        self.logger.info(f'trained epochs {current_iters}')


        if len(list(opt_params)) == 0:
            max_iters = 4_000
            initial_lr = 5e-3
            iter_save = max_iters
            kl_multiplier=1.0
        else:
            iter_save = opt_params.get('iter_save', 4000)
            max_iters = opt_params['max_iters']
            initial_lr = opt_params['initial_lr']
            kl_multiplier= opt_params['kl_multiplier']
        self.logger.info(f'opt_params: {opt_params}')
        # It won't train unless instructed
        if force_train:
            self.logger.debug(f'Training. shapes passed to LMF : {X_train.shape, Y_train.shape}')
            if len(self.emu.loss_history) >= max_iters:
                self.logger.info(f'{model_file} already trained for {max_iters} iterations')
                return
            # Do the training in batches of iter_save, so we defenitely save
            # the model every iter_save iterations
            iter_stop_point = np.append(np.arange(current_iters, max_iters, iter_save), max_iters) if max_iters % iter_save != 0 else np.arange(current_iters, max_iters + 1, iter_save)
            iter_stop_point = iter_stop_point[1:]
            for it_stp in iter_stop_point:
                current_iters = len(self.emu.loss_history)
                self.logger.info(f'Continue optimization from {current_iters} to {it_stp}')
                # The decaying learning rate
                start_lr = tf.keras.optimizers.schedules.CosineDecay(initial_lr, max_iters)(current_iters)
                # Optimize on mean targets; aleatoric noise is predicted by the model
                self.emu.optimize(data=(X_train, Y_train), max_iters=it_stp, 
                                  initial_lr=start_lr, unfix_noise_after=500,
                                  kl_multiplier=kl_multiplier)
                self.emu.save_model(model_file)
                # Save loss_history, ind_train and emu_type
                with open(f'{model_file}.attrs', 'wb') as f:
                    self.logger.debug(f'Writing the model on {model_file}')
                    self.model_attrs = {}
                    self.model_attrs['loss_history'] = self.emu.loss_history
                    self.model_attrs['kl_history'] = self.emu.kl_history
                    #self.model_attrs['ind_train'] = ind_train
                    self.model_attrs['emu_type'] = self.emu_type
                    pickle.dump(self.model_attrs, f)
            self.logger.info(f'done with optimization {max_iters}')

    def predict(self, ind_test, model_file, train_subdir = 'train'):
        """
        Posteroir prediction of the emulator
        Parameters
        ----------
        ind_train : array
            The indices of the HF sims to be used for training
        ind_test : array
            The indices of the HF sims to be used for testing
        model_file : str
            The file to save the Emulator. If the file exists, 
            the model is loaded from the file.
        Returns
        -------
        mean_pred, var_pred : (array, array)
            The mean and variance of the predicted 
            log10(xi(r)) for the test sims.
        """
        # Get the median function for the HF sims used for training
        if not hasattr(self, 'hf_median_func'):
            mask = np.ones(self.Y[1].shape[0], dtype=bool)
            mask[ind_test] = False
            self.hf_median_func = np.nanmedian(self.Y[1][mask], axis=0)
        try:
            with open(op.join(self.data_dir, train_subdir, f'{model_file}.attrs'), 'rb') as f:
                self.model_attrs = pickle.load(f)
        except:
            self.logger.warning(f'No model attributes found for {model_file}.attrs')
            self.model_attrs = {}
        #ind_train = self.model_attrs['ind_train']
        #self.emu_type = self.model_attrs['emu_type']
        #self.train(ind_train, model_file, force_train=False, train_subdir=train_subdir)
        self.train(model_file=model_file, force_train=False, train_subdir=train_subdir)
        
        # Add the fidelity indocators
        X_test = np.hstack([self.X[1][ind_test], np.ones((ind_test.size, 1))]).astype(np.float32)
        Fmu, Fvar = self.emu.predict_f(X_test)
        P = self.output_dim
        mean_pred = Fmu[:, :P]
        mean_var = Fvar[:, :P]
        logvar_mu = Fmu[:, P:]
        logvar_var = Fvar[:, P:]
        aleatoric_var = np.exp(logvar_mu + 0.5 * logvar_var)
        base_noise = np.array(self.emu.likelihood.variance)
        var_pred = mean_var + (base_noise + aleatoric_var)
        if not self.use_rho:
            mean_pred += self.hf_median_func
        
        return mean_pred, mean_var, base_noise, aleatoric_var
