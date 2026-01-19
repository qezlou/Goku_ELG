import argparse
import json
import os
import os.path as op
from typing import Dict, Any, Optional

import numpy as np
import optuna
import torch
from gal_goku import emus_multifid


def _resolve_data_dir(machine: str) -> str:
    if machine == 'stampede3':
        return '/scratch/06536/qezlou/Goku/processed_data/'
    if machine == 'vista':
        return '/scratch/06536/qezlou/goku/processed_data/'
    if machine == 'ucr':
        return '/rhome/mqezl001/bigdata/HETDEX/data/'
    if machine == 'pc':
        return '/home/qezlou/HD2/HETDEX/cosmo/data/'
    raise ValueError('machine not recognized')


def _ensure_hidden_units(config: Dict[str, Any]):
    if 'flow_hidden_units' in config and isinstance(config['flow_hidden_units'], list):
        config = config.copy()
        config['flow_hidden_units'] = tuple(config['flow_hidden_units'])
    if 'mean_hidden_units' in config and isinstance(config['mean_hidden_units'], list):
        config = config.copy()
        config['mean_hidden_units'] = tuple(config['mean_hidden_units'])
    if 'lf_mean_hidden_units' in config and isinstance(config['lf_mean_hidden_units'], list):
        config = config.copy()
        config['lf_mean_hidden_units'] = tuple(config['lf_mean_hidden_units'])
    return config


def _save_config(config: Dict[str, Any], path: str):
    os.makedirs(op.dirname(path), exist_ok=True)
    with open(path, 'w') as fh:
        json.dump(config, fh)


def _holdout_neg_log_likelihood(emu: emus_multifid.HmfNativeBins, ind_test: int) -> float:
    if ind_test is None:
        raise ValueError('ind_test must be provided for validation')
    if emu.flow_model is None:
        raise RuntimeError('flow_model is not trained; cannot score holdout')
    x_hf = emu.X[1][[ind_test]].astype(np.float32)
    y_hf = emu.Y[1][[ind_test]].astype(np.float32)
    
    # Prepare input for LF mean net
    x_tensor = torch.as_tensor(x_hf, dtype=emu.flow_model.dtype, device=emu.flow_model.device)
    
    with torch.no_grad():
        lf_pred = emu.lf_mean_net(x_tensor)
    
    # Prepare context for Flow [X, LF_pred]
    context = torch.cat([x_tensor, lf_pred], dim=1)
    
    # Target is raw HF; log_prob handles mean subtraction internally now
    target = torch.as_tensor(y_hf, dtype=emu.flow_model.dtype, device=emu.flow_model.device)
    with torch.no_grad():
        log_prob = emu.flow_model.log_prob(target, context)
    return -float(log_prob.mean().cpu().item())


def run_it(ind_test, z, train_subdir, machine='stampede3',
           norm_type='subtract_mean', noise_floor=0.0,
           flow_max_iters=20000, flow_initial_lr=1e-3, flow_scheduler_type='plateau',
           flow_scheduler_gamma=0.5, flow_scheduler_patience=150, flow_scheduler_min_lr=1e-6,
           flow_early_stopping_patience=400, flow_early_stopping_min_delta=1e-4,
           flow_batch_size=128, flow_num_bijectors=6, flow_hidden_units=(256, 256),
           mean_hidden_units=(256, 256), mean_loss_weight=0.0,
           lf_mean_hidden_units=(128, 128), lf_mean_max_iters=2000, lf_mean_lr=1e-3, lf_mean_batch_size=128,
           lf_mean_dropout=0.0,
           lf_mean_scheduler_type='plateau', lf_mean_scheduler_gamma=0.5, lf_mean_scheduler_patience=200,
           lf_mean_scheduler_min_lr=1e-6, lf_mean_early_stopping_patience=None, lf_mean_early_stopping_min_delta=0.0,
           flow_log_every=500, flow_num_samples=512, save_config=True, model_suffix: Optional[str] = None):
    data_dir = _resolve_data_dir(machine)
    os.makedirs(op.join(data_dir, train_subdir), exist_ok=True)

    cfg_payload = {
        'train_subdir': train_subdir,
        'norm_type': norm_type,
        'flow_max_iters': flow_max_iters,
        'flow_initial_lr': flow_initial_lr,
        'flow_scheduler_type': flow_scheduler_type,
        'flow_scheduler_gamma': flow_scheduler_gamma,
        'flow_scheduler_patience': flow_scheduler_patience,
        'flow_scheduler_min_lr': flow_scheduler_min_lr,
        'flow_early_stopping_patience': flow_early_stopping_patience,
        'flow_early_stopping_min_delta': flow_early_stopping_min_delta,
        'flow_batch_size': flow_batch_size,
        'flow_num_bijectors': flow_num_bijectors,
        'flow_hidden_units': list(flow_hidden_units),
        'flow_log_every': flow_log_every,
        'flow_num_samples': flow_num_samples,
        'noise_floor': noise_floor,
        'mean_hidden_units': list(mean_hidden_units),
        'mean_loss_weight': mean_loss_weight,
        'lf_mean_hidden_units': list(lf_mean_hidden_units),
        'lf_mean_max_iters': lf_mean_max_iters,
        'lf_mean_lr': lf_mean_lr,
        'lf_mean_batch_size': lf_mean_batch_size,
        'lf_mean_dropout': lf_mean_dropout,
        'lf_mean_scheduler_type': lf_mean_scheduler_type,
        'lf_mean_scheduler_gamma': lf_mean_scheduler_gamma,
        'lf_mean_scheduler_patience': lf_mean_scheduler_patience,
        'lf_mean_scheduler_min_lr': lf_mean_scheduler_min_lr,
        'lf_mean_early_stopping_patience': lf_mean_early_stopping_patience,
        'lf_mean_early_stopping_min_delta': lf_mean_early_stopping_min_delta,
        'lf_finetune_on_hf': False,
        'lf_finetune_max_iters': 500,
        'lf_finetune_lr': lf_mean_lr,
        'lf_finetune_batch_size': lf_mean_batch_size,
    }
    if save_config:
        _save_config(cfg_payload, op.join(data_dir, train_subdir, 'config.json'))

    emu = emus_multifid.HmfNativeBins(data_dir=data_dir,
                                     z=z,
                                     norm_type=norm_type,
                                     noise_floor=noise_floor,
                                     logging_level='DEBUG')

    ind_train = np.delete(np.arange(emu.Y[1].shape[0]), [ind_test])
    model_file = f'hmf_emu_combined_z{z}_leave{ind_test}'
    if model_suffix:
        model_file = f'{model_file}_{model_suffix}'
    emu.logger.info(f'will save on {model_file}')

    emu.train(ind_train,
              train_subdir=train_subdir,
              opt_params={
                  'max_iters': flow_max_iters,
                  'initial_lr': flow_initial_lr,
                  'lr_scheduler_type': flow_scheduler_type,
                  'lr_scheduler_gamma': flow_scheduler_gamma,
                  'lr_scheduler_patience': flow_scheduler_patience,
                  'lr_scheduler_min_lr': flow_scheduler_min_lr,
                  'early_stopping_patience': flow_early_stopping_patience,
                  'early_stopping_min_delta': flow_early_stopping_min_delta,
                  'batch_size': flow_batch_size,
                  'num_bijectors': flow_num_bijectors,
                  'hidden_units': flow_hidden_units,
                  'mean_hidden_units': mean_hidden_units,
                  'mean_loss_weight': mean_loss_weight,
                  'lf_mean_hidden_units': lf_mean_hidden_units,
                  'lf_mean_max_iters': lf_mean_max_iters,
                  'lf_mean_lr': lf_mean_lr,
                  'lf_mean_batch_size': lf_mean_batch_size,
                  'lf_mean_dropout': lf_mean_dropout,
                  'lf_mean_scheduler_type': lf_mean_scheduler_type,
                  'lf_mean_scheduler_gamma': lf_mean_scheduler_gamma,
                  'lf_mean_scheduler_patience': lf_mean_scheduler_patience,
                  'lf_mean_scheduler_min_lr': lf_mean_scheduler_min_lr,
                  'lf_mean_early_stopping_patience': lf_mean_early_stopping_patience,
                  'lf_mean_early_stopping_min_delta': lf_mean_early_stopping_min_delta,
                  'lf_finetune_on_hf': base_config.get('lf_finetune_on_hf', False),
                  'lf_finetune_max_iters': base_config.get('lf_finetune_max_iters', 500),
                  'lf_finetune_lr': base_config.get('lf_finetune_lr', lf_mean_lr),
                  'lf_finetune_batch_size': base_config.get('lf_finetune_batch_size', lf_mean_batch_size),
                  'log_every': flow_log_every,
                  'num_samples': flow_num_samples
              },
              model_file=model_file)
    return emu, model_file, data_dir


def _suggest_hyperparams(trial: optuna.Trial) -> Dict[str, Any]:
    return {
        'flow_hidden_units': trial.suggest_categorical(
            'flow_hidden_units',
            [(128, 128), (256, 256), (256, 256, 128)]
        ),
        'lf_mean_hidden_units': trial.suggest_categorical(
            'lf_mean_hidden_units',
            [(64, 64), (128, 128), (256, 256)]
        ),
    }


def _build_objective(args, base_config: Dict[str, Any]):
    base_config = _ensure_hidden_units(base_config)

    def objective(trial: optuna.Trial) -> float:
        trial_cfg = base_config.copy()
        trial_cfg.update(_suggest_hyperparams(trial))
        trial_cfg['flow_hidden_units'] = tuple(trial_cfg['flow_hidden_units'])
        trial_cfg['lf_mean_hidden_units'] = tuple(trial_cfg['lf_mean_hidden_units'])
        trial_subdir = op.join(trial_cfg['train_subdir'], 'optuna_trials')
        emu, _, _ = run_it(
            args.ind_test,
            z=args.z,
            train_subdir=trial_subdir,
            machine=args.machine,
            norm_type=trial_cfg.get('norm_type', 'subtract_mean'),
            noise_floor=trial_cfg.get('noise_floor', 0.0),
            flow_max_iters=trial_cfg.get('flow_max_iters', 20000),
            flow_initial_lr=trial_cfg.get('flow_initial_lr', 1e-3),
            flow_scheduler_type=trial_cfg.get('flow_scheduler_type', 'plateau'),
            flow_scheduler_gamma=trial_cfg.get('flow_scheduler_gamma', 0.5),
            flow_scheduler_patience=trial_cfg.get('flow_scheduler_patience', 150),
            flow_scheduler_min_lr=trial_cfg.get('flow_scheduler_min_lr', 1e-6),
            flow_early_stopping_patience=trial_cfg.get('flow_early_stopping_patience', 400),
            flow_early_stopping_min_delta=trial_cfg.get('flow_early_stopping_min_delta', 1e-4),
            flow_batch_size=trial_cfg.get('flow_batch_size', 128),
            flow_num_bijectors=trial_cfg.get('flow_num_bijectors', 6),
            flow_hidden_units=trial_cfg.get('flow_hidden_units', (256, 256)),
            mean_hidden_units=trial_cfg.get('mean_hidden_units', (256, 256)),
            mean_loss_weight=trial_cfg.get('mean_loss_weight', 0.0),
            lf_mean_hidden_units=trial_cfg.get('lf_mean_hidden_units', (128, 128)),
            lf_mean_max_iters=trial_cfg.get('lf_mean_max_iters', 2000),
            lf_mean_lr=trial_cfg.get('lf_mean_lr', 1e-3),
            lf_mean_batch_size=trial_cfg.get('lf_mean_batch_size', 128),
            lf_mean_dropout=trial_cfg.get('lf_mean_dropout', 0.0),
            lf_mean_scheduler_type=trial_cfg.get('lf_mean_scheduler_type', 'plateau'),
            lf_mean_scheduler_gamma=trial_cfg.get('lf_mean_scheduler_gamma', 0.5),
            lf_mean_scheduler_patience=trial_cfg.get('lf_mean_scheduler_patience', 200),
            lf_mean_scheduler_min_lr=trial_cfg.get('lf_mean_scheduler_min_lr', 1e-6),
            lf_mean_early_stopping_patience=trial_cfg.get('lf_mean_early_stopping_patience'),
            lf_mean_early_stopping_min_delta=trial_cfg.get('lf_mean_early_stopping_min_delta', 0.0),
            flow_log_every=trial_cfg.get('flow_log_every', 500),
            flow_num_samples=trial_cfg.get('flow_num_samples', 512),
            save_config=False,
            model_suffix=f'trial{trial.number}'
        )
        return _holdout_neg_log_likelihood(emu, args.ind_test)

    return objective

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Xi LOOCV')
    parser.add_argument('--ind_test', default=None, type=int, help='')
    parser.add_argument('--z', default=2.5, type=float, help='Redshift')
    parser.add_argument('--machine', default='stampede3', type=str, help='Machine name')
    parser.add_argument('--config', default='config.json', type=str, help='Path to config file')
    parser.add_argument('--n_trials', default=0, type=int, help='Number of Optuna trials to run')
    parser.add_argument('--study_name', default='hmf_nflow_hyperopt', type=str, help='Optuna study name')
    parser.add_argument('--storage', default=None, type=str, help='Optuna storage (e.g., sqlite:///path/to/study.db)')
    parser.add_argument('--train_best', action='store_true', default=True, help='Train final model with best params after search')

    args = parser.parse_args()
    with open(args.config, 'r') as f:
        base_config = json.load(f)
    base_config = _ensure_hidden_units(base_config)

    if args.n_trials and args.n_trials > 0:
        if args.ind_test is None:
            raise ValueError('Optuna search requires --ind_test to compute validation loss')
        study = optuna.create_study(
            study_name=args.study_name,
            direction='minimize',
            storage=args.storage,
            load_if_exists=True
        )
        study.optimize(_build_objective(args, base_config), n_trials=args.n_trials)
        best_config = base_config.copy()
        best_config.update(study.best_params)
        best_config['flow_hidden_units'] = list(best_config['flow_hidden_units'])
        if 'lf_mean_hidden_units' in best_config:
            best_config['lf_mean_hidden_units'] = list(best_config['lf_mean_hidden_units'])
        best_cfg_path = op.join(_resolve_data_dir(args.machine), base_config['train_subdir'], 'optuna_best_config.json')
        _save_config(best_config, best_cfg_path)
        print(f'Best Optuna value: {study.best_value:.4f}, config saved to {best_cfg_path}', flush=True)
        if args.train_best:
            run_it(args.ind_test, z=args.z, train_subdir=base_config['train_subdir'],
                   machine=args.machine,
                   norm_type=best_config.get('norm_type', 'subtract_mean'),
                   noise_floor=best_config.get('noise_floor', 0.0),
                   flow_max_iters=best_config.get('flow_max_iters', 20000),
                   flow_initial_lr=best_config.get('flow_initial_lr', 1e-3),
                   flow_scheduler_type=best_config.get('flow_scheduler_type', 'plateau'),
                   flow_scheduler_gamma=best_config.get('flow_scheduler_gamma', 0.5),
                   flow_scheduler_patience=best_config.get('flow_scheduler_patience', 150),
                   flow_scheduler_min_lr=best_config.get('flow_scheduler_min_lr', 1e-6),
                   flow_early_stopping_patience=best_config.get('flow_early_stopping_patience', 400),
                   flow_early_stopping_min_delta=best_config.get('flow_early_stopping_min_delta', 1e-4),
                   flow_batch_size=best_config.get('flow_batch_size', 128),
                   flow_num_bijectors=best_config.get('flow_num_bijectors', 6),
                   flow_hidden_units=tuple(best_config.get('flow_hidden_units', (256, 256))),
                   mean_hidden_units=tuple(best_config.get('mean_hidden_units', (256, 256))),
                   mean_loss_weight=best_config.get('mean_loss_weight', 0.0),
                   lf_mean_hidden_units=tuple(best_config.get('lf_mean_hidden_units', (128, 128))),
                   lf_mean_max_iters=best_config.get('lf_mean_max_iters', 2000),
                   lf_mean_lr=best_config.get('lf_mean_lr', 1e-3),
                   lf_mean_batch_size=best_config.get('lf_mean_batch_size', 128),
                   lf_mean_dropout=best_config.get('lf_mean_dropout', 0.0),
                   lf_mean_scheduler_type=best_config.get('lf_mean_scheduler_type', 'plateau'),
                   lf_mean_scheduler_gamma=best_config.get('lf_mean_scheduler_gamma', 0.5),
                   lf_mean_scheduler_patience=best_config.get('lf_mean_scheduler_patience', 200),
                   lf_mean_scheduler_min_lr=best_config.get('lf_mean_scheduler_min_lr', 1e-6),
                   lf_mean_early_stopping_patience=best_config.get('lf_mean_early_stopping_patience'),
                   lf_mean_early_stopping_min_delta=best_config.get('lf_mean_early_stopping_min_delta', 0.0),
                   flow_log_every=best_config.get('flow_log_every', 500),
                   flow_num_samples=best_config.get('flow_num_samples', 512))
    else:
        run_it(args.ind_test, z=args.z, train_subdir=base_config['train_subdir'],
               machine=args.machine,
               norm_type=base_config.get('norm_type', 'subtract_mean'),
               noise_floor=base_config.get('noise_floor', 0.0),
               flow_max_iters=base_config.get('flow_max_iters', 20000),
               flow_initial_lr=base_config.get('flow_initial_lr', 1e-3),
               flow_scheduler_type=base_config.get('flow_scheduler_type', 'plateau'),
               flow_scheduler_gamma=base_config.get('flow_scheduler_gamma', 0.5),
               flow_scheduler_patience=base_config.get('flow_scheduler_patience', 150),
               flow_scheduler_min_lr=base_config.get('flow_scheduler_min_lr', 1e-6),
               flow_early_stopping_patience=base_config.get('flow_early_stopping_patience', 400),
               flow_early_stopping_min_delta=base_config.get('flow_early_stopping_min_delta', 1e-4),
               flow_batch_size=base_config.get('flow_batch_size', 128),
               flow_num_bijectors=base_config.get('flow_num_bijectors', 6),
               flow_hidden_units=tuple(base_config.get('flow_hidden_units', (256, 256))),
               mean_hidden_units=tuple(base_config.get('mean_hidden_units', (256, 256))),
               mean_loss_weight=base_config.get('mean_loss_weight', 0.0),
               lf_mean_hidden_units=tuple(base_config.get('lf_mean_hidden_units', (128, 128))),
               lf_mean_max_iters=base_config.get('lf_mean_max_iters', 2000),
               lf_mean_lr=base_config.get('lf_mean_lr', 1e-3),
               lf_mean_batch_size=base_config.get('lf_mean_batch_size', 128),
               lf_mean_dropout=base_config.get('lf_mean_dropout', 0.0),
               lf_mean_scheduler_type=base_config.get('lf_mean_scheduler_type', 'plateau'),
               lf_mean_scheduler_gamma=base_config.get('lf_mean_scheduler_gamma', 0.5),
               lf_mean_scheduler_patience=base_config.get('lf_mean_scheduler_patience', 200),
               lf_mean_scheduler_min_lr=base_config.get('lf_mean_scheduler_min_lr', 1e-6),
               lf_mean_early_stopping_patience=base_config.get('lf_mean_early_stopping_patience'),
               lf_mean_early_stopping_min_delta=base_config.get('lf_mean_early_stopping_min_delta', 0.0),
               flow_log_every=base_config.get('flow_log_every', 500),
               flow_num_samples=base_config.get('flow_num_samples', 512))
