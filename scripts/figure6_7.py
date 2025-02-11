"""Script for generating Figure 6 and 7.

- Axial slice of 12, 6, and 4pc parameter estimation for T1, T2
- SVNN, PINN, and MIRACLE.
- Merge the script for application of standard (magnitude-based) DNNs and MIRACLE with script from 2b (complex-based DNNs)

Manuscript: Fig 6: T1 maps
Manuscript: Fig 7: T2 maps

Author: florian.birk@tuebingen.mpg.de (Florian Birk), December 2024
"""
# %% Import
import torch
import numpy as np
import matplotlib.pyplot as plt
from tueplots import figsizes, fonts, fontsizes, axes
from fcns_mb import calc_Fn, norm2_vox, getparams, return_without_f0_imag_2d
from fcns_dataprep import train_val_test_split, apply_transform, revert_min_max_transform
from fcns_fig_plot import plot_fig6, plot_fig7
from miracle import calc_t1_t2_ima_gss_B1_reg
from model import Qbssfp
import paths
from pathlib import Path
import yaml

# %% Define paths
path_in_vivo_pred = paths.data / 'in_vivo' / 'pred'
path_in_vivo_pred.mkdir(parents=True, exist_ok=True)

pred_svnn_fp = path_in_vivo_pred / 'pred_svnn.npy'
pred_pinn_fp = path_in_vivo_pred / 'pred_pinn.npy'
pred_miracle_fp = path_in_vivo_pred / 'pred_miracle.npy'
if pred_svnn_fp.exists():
    pred_svnn = np.load(pred_svnn_fp)
    pred_pinn = np.load(pred_pinn_fp)
    pred_miracle = np.load(pred_miracle_fp)
    npc_trained = [12, 6, 4]
else:
    # load the training data and estimate the sig_mean and sig_std
    path_in_silico = paths.data / 'in_silico' / 'train'
    path_in_vivo = paths.data / 'in_vivo' / 'test'
    cfg = yaml.safe_load(
        open(paths.cfgs / '1.nn_train' / 'train_in_silico.yml'))
    cfg['model'] = 'sv'
    cfg['snr'] = 'inf'
    cfg['distr'] = 'uniform'
    cfg['b0fit'] = False
    cfg['realimag'] = False
    cfg['param_min'] = cfg['param_min'][:-1]
    cfg['param_max'] = cfg['param_max'][:-1]
    cfg['nneurons'] = [4, 64, 64, 2]
    cfg['mb'] = False
    cfg_pinn = cfg.copy()
    cfg_pinn['mb'] = True
    # define the DNNs without B0 fitting for all three pc and both models (mag + B1 as input)
    # for 12pc, use standard magnitude-based DNNs from initial submission
    dnn_mag_fop = paths.dnn_models / 'magnitude-based'
    svnn_fps = [dnn_mag_fop / '1_sv_inf_uniform_blooming-universe.pt',
                dnn_mag_fop / '2_sv_inf_uniform_polar-microwave.pt',
                dnn_mag_fop / '3_sv_inf_uniform_drawn-cherry.pt']
    pinn_fps = [dnn_mag_fop / '4_mb_inf_uniform_wandering-paper.pt',
                dnn_mag_fop / '5_mb_inf_uniform_sage-smoke.pt',
                dnn_mag_fop / '6_mb_inf_uniform_sage-deluge.pt']
    npc_trained = [12, 6, 4]

    for i, npc_train in enumerate(npc_trained):
        npc_name = f'{npc_train}pc'
        cfg['npc'] = npc_train
        cfg_pinn['npc'] = npc_train
        # init models
        model_svnn = Qbssfp(cfg).to(cfg['device'])
        model_pinn = Qbssfp(cfg_pinn).to(cfg['device'])
        # load the models
        model_svnn.load_state_dict(torch.load(svnn_fps[i]))
        model_pinn.load_state_dict(torch.load(pinn_fps[i]))

        # load the training data and estimate mean and std
        sig = torch.load(path_in_silico /
                         f'bssfp_complex_nob0_uniform_{npc_name}.pt').cfloat()
        par = torch.load(path_in_silico /
                         f't1t2b0b1_nob0_uniform_{npc_name}.pt').float()
        sig_train, sig_val, sig_test, par_train, par_val, par_test = train_val_test_split(
            sig, par, cfg['split'])
        # calculate mean and std
        fn_complex = calc_Fn(sig_train, model_svnn.phi_nom_rad,
                             Np=model_svnn.Np, device=cfg['device'])
        fn_complex = norm2_vox(fn_complex)  # normalize using euclidean norm
        sig = torch.cat(
            (torch.abs(fn_complex), par_train[:, -1].unsqueeze(-1)), dim=1)
        sig_mean = torch.mean(sig, dim=0)
        sig_std = torch.std(sig, dim=0)

        # load the in vivo data (already fn_mag_norm, fn_pha, b1 in 2D)
        sig_in_vivo = torch.load(path_in_vivo /
                                 f'sig_all_nn.{npc_name}.pt').float()
        # remove the phase columns from the input
        sig_in_vivo = torch.cat(
            (sig_in_vivo[:, :3], sig_in_vivo[:, -1].unsqueeze(-1)), dim=1)
        mask_3d = torch.load(
            path_in_vivo / f'mask_3d.{npc_name}.pt').float()

        # apply transform
        sig_in_vivo_z = apply_transform(
            sig_in_vivo, sig_mean, sig_std, 'cpu')

        # DNN inference
        par_pred_svnn = revert_min_max_transform(
            model_svnn(sig_in_vivo_z), model_svnn.param_min, model_svnn.param_max).detach().numpy()
        par_pred_pinn = getparams(model_pinn(sig_in_vivo_z), model_pinn.param_min,
                                  model_pinn.param_max, 'cpu').detach().numpy()
        # reshape
        par_pred_svnn = np.reshape(
            par_pred_svnn, (mask_3d.shape[0], mask_3d.shape[1], mask_3d.shape[2], 2))
        par_pred_pinn = np.reshape(
            par_pred_pinn, (mask_3d.shape[0], mask_3d.shape[1], mask_3d.shape[2], 2))
        for j in range(par_pred_svnn.shape[-1]):
            par_pred_svnn[:, :, :, j] = np.where(
                mask_3d.numpy() != 0, par_pred_svnn[:, :, :, j], np.nan)
            par_pred_pinn[:, :, :, j] = np.where(
                mask_3d.numpy() != 0, par_pred_pinn[:, :, :, j], np.nan)

        # MIRACLE
        # prepare input
        fn_mag_in_vivo = torch.reshape(
            sig_in_vivo[:, :3], (mask_3d.shape[0], mask_3d.shape[1], mask_3d.shape[2], 3))
        b1_in_vivo = torch.reshape(
            sig_in_vivo[:, -1], (mask_3d.shape[0], mask_3d.shape[1], mask_3d.shape[2])).numpy()
        fm1 = fn_mag_in_vivo[..., 0].numpy()
        f0 = fn_mag_in_vivo[..., 1].numpy()
        f1 = fn_mag_in_vivo[..., 2].numpy()
        tr = 4.8  # ms
        te = 2.4  # ms
        fa_nom = 15  # deg
        t1_est = 1000

        # estimate T1 and T2
        t1_miracle, t2_miracle, _ = calc_t1_t2_ima_gss_B1_reg(
            f1=f1,
            f0=f0,
            fm1=fm1,
            b1=b1_in_vivo,
            tr=tr,
            te=te,
            flipangle=fa_nom,
            t1_est=t1_est,
            mask=mask_3d.numpy().astype(bool))

        # set miracle predictions to nan where mask is zero
        t1_miracle = np.where(mask_3d.numpy() != 0, t1_miracle, np.nan)
        t2_miracle = np.where(mask_3d.numpy() != 0, t2_miracle, np.nan)

        # save the predictions
        if i == 0:
            pred_svnn = np.zeros(par_pred_svnn.shape + (len(npc_trained),))
            pred_pinn = np.zeros(par_pred_pinn.shape + (len(npc_trained),))
            pred_miracle = np.zeros(par_pred_svnn.shape + (len(npc_trained),))
        pred_svnn[..., i] = par_pred_svnn
        pred_pinn[..., i] = par_pred_pinn
        pred_miracle[..., i] = np.stack((t1_miracle, t2_miracle), axis=-1)

    # save the predictions
    np.save(pred_svnn_fp, pred_svnn)
    np.save(pred_pinn_fp, pred_pinn)
    np.save(pred_miracle_fp, pred_miracle)

# %% Apply the same for complex-based DNNs
pred_svnn_complex_fp = path_in_vivo_pred / 'pred_svnn_complex.npy'
pred_pinn_complex_fp = path_in_vivo_pred / 'pred_pinn_complex.npy'
if pred_svnn_complex_fp.exists():
    pred_svnn_complex = np.load(pred_svnn_complex_fp)
    pred_pinn_complex = np.load(pred_pinn_complex_fp)
    npc_trained = [12, 6, 4]
else:
    # load the training data and estimate the sig_mean and sig_std
    path_in_silico = paths.data / 'in_silico' / 'train'
    path_in_vivo = paths.data / 'in_vivo' / 'test'
    cfg = yaml.safe_load(
        open(paths.cfgs / '1.nn_train' / 'train_in_silico.yml'))
    cfg['model'] = 'sv'
    cfg['snr'] = 'inf'
    cfg['distr'] = 'uniform'
    cfg['b0fit'] = True
    cfg['realimag'] = True
    cfg['param_min'][-1] = cfg['param_min'][-1] * np.pi
    cfg['param_max'][-1] = cfg['param_max'][-1] * np.pi
    cfg['mb'] = False
    cfg_pinn = cfg.copy()
    cfg_pinn['mb'] = True
    # define the DNNs with B0 fitting for all three pc and both models (real/imag + B1 as input)
    dnn_complex_fop = paths.dnn_models / 'complex-based'
    svnn_fps = [dnn_complex_fop / '7_sv_inf_uniform_pretty-gorge.pt',
                dnn_complex_fop / '8_sv_inf_uniform_revived-breeze.pt',
                dnn_complex_fop / '9_sv_inf_uniform_sweet-dream.pt']
    pinn_fps = [dnn_complex_fop / '10_mb_inf_uniform_leafy-paper.pt',
                dnn_complex_fop / '11_mb_inf_uniform_splendid-energy.pt',
                dnn_complex_fop / '12_mb_inf_uniform_stellar-mountain.pt']

    npc_trained = [12, 6, 4]

    for i, npc_train in enumerate(npc_trained):
        npc_name = f'{npc_train}pc'
        cfg['npc'] = npc_train
        cfg_pinn['npc'] = npc_train
        # init models
        model_svnn = Qbssfp(cfg).to(cfg['device'])
        model_pinn = Qbssfp(cfg_pinn).to(cfg['device'])
        # load the models
        model_svnn.load_state_dict(torch.load(svnn_fps[i]))
        model_pinn.load_state_dict(torch.load(pinn_fps[i]))

        # load the training data and estimate mean and std
        sig = torch.load(path_in_silico /
                         f'bssfp_complex_uniform_{npc_name}.pt').cfloat()
        par = torch.load(path_in_silico /
                         f't1t2b0b1_uniform_{npc_name}.pt').float()
        sig_train, sig_val, sig_test, par_train, par_val, par_test = train_val_test_split(
            sig, par, cfg['split'])
        # calculate mean and std
        fn_complex = calc_Fn(sig_train, model_svnn.phi_nom_rad,
                             Np=model_svnn.Np, device=cfg['device'])
        fn_complex = norm2_vox(fn_complex)  # normalize using euclidean norm
        sig = torch.cat(
            (torch.real(fn_complex), return_without_f0_imag_2d(fn_complex, Np=1), par_train[:, -1].unsqueeze(-1)), dim=1)
        sig_mean = torch.mean(sig, dim=0)
        sig_std = torch.std(sig, dim=0)

        # load the in vivo data (already fn_mag_norm, fn_pha, b1 in 2D)
        sig_in_vivo = torch.load(path_in_vivo /
                                 f'sig_all_realimag_nn.{npc_name}.pt').float()
        # remove imag of F0 from input.
        # Multiply Real F-1 with -1 because add_pifm1 function also has an impact on real part of F-1
        sig_in_vivo = torch.cat(
            (sig_in_vivo[:, :1]*-1, sig_in_vivo[:, 1:4], sig_in_vivo[:, 5].unsqueeze(-1), sig_in_vivo[:, -1].unsqueeze(-1)), dim=1)
        mask_3d = torch.load(
            path_in_vivo / f'mask_3d.{npc_name}.pt').float()

        # apply transform
        sig_in_vivo_z = apply_transform(
            sig_in_vivo, sig_mean, sig_std, 'cpu')

        # DNN inference
        par_pred_svnn_complex = revert_min_max_transform(
            model_svnn(sig_in_vivo_z), model_svnn.param_min, model_svnn.param_max).detach().numpy()
        par_pred_pinn_complex = getparams(model_pinn(sig_in_vivo_z), model_pinn.param_min,
                                          model_pinn.param_max, 'cpu').detach().numpy()
        # reshape
        par_pred_svnn_complex = np.reshape(
            par_pred_svnn_complex, (mask_3d.shape[0], mask_3d.shape[1], mask_3d.shape[2], 3))
        par_pred_pinn_complex = np.reshape(
            par_pred_pinn_complex, (mask_3d.shape[0], mask_3d.shape[1], mask_3d.shape[2], 3))
        for j in range(par_pred_svnn_complex.shape[-1]):
            par_pred_svnn_complex[:, :, :, j] = np.where(
                mask_3d.numpy() != 0, par_pred_svnn_complex[:, :, :, j], np.nan)
            par_pred_pinn_complex[:, :, :, j] = np.where(
                mask_3d.numpy() != 0, par_pred_pinn_complex[:, :, :, j], np.nan)

        # save the predictions
        if i == 0:
            pred_svnn_complex = np.zeros(
                par_pred_svnn_complex.shape + (len(npc_trained),))
            pred_pinn_complex = np.zeros(
                par_pred_pinn_complex.shape + (len(npc_trained),))
        pred_svnn_complex[..., i] = par_pred_svnn_complex
        pred_pinn_complex[..., i] = par_pred_pinn_complex

    # save the predictions
    np.save(pred_svnn_complex_fp, pred_svnn_complex)
    np.save(pred_pinn_complex_fp, pred_pinn_complex)


# %% calculate the difference maps between 6 and 4pc and respective 12pc predictions
diff_miracle = np.zeros(pred_miracle.shape)[..., :-1]
diff_svnn = np.zeros(pred_svnn.shape)[..., :-1]
diff_pinn = np.zeros(pred_pinn.shape)[..., :-1]
diff_svnn_complex = np.zeros(pred_svnn_complex.shape)[..., :-1]
diff_pinn_complex = np.zeros(pred_pinn_complex.shape)[..., :-1]

npc_diff = [6, 4]
for i, npc in enumerate(npc_diff):
    diff_miracle[..., i] = pred_miracle[..., i+1] - \
        pred_miracle[..., 0]
    diff_svnn[..., i] = pred_svnn[..., i+1] - \
        pred_svnn[..., 0]
    diff_pinn[..., i] = pred_pinn[..., i+1] - \
        pred_pinn[..., 0]
    diff_svnn_complex[..., i] = pred_svnn_complex[..., i+1] - \
        pred_svnn_complex[..., 0]
    diff_pinn_complex[..., i] = pred_pinn_complex[..., i+1] - \
        pred_pinn_complex[..., 0]


# %% Plotting
# Figure 6 (T1 maps)
plot_fig6(
    pred_miracle=pred_miracle,
    pred_svnn=pred_svnn,
    pred_svnn_complex=pred_svnn_complex,
    pred_pinn=pred_pinn,
    pred_pinn_complex=pred_pinn_complex,
    diff_miracle=diff_miracle,
    diff_svnn=diff_svnn,
    diff_svnn_complex=diff_svnn_complex,
    diff_pinn=diff_pinn,
    diff_pinn_complex=diff_pinn_complex,
    npc_diff=npc_diff,
    path_save=paths.figures / 'fig6.svg'
)

# %% Figure 7 (T2 maps)
plot_fig7(
    pred_miracle=pred_miracle,
    pred_svnn=pred_svnn,
    pred_svnn_complex=pred_svnn_complex,
    pred_pinn=pred_pinn,
    pred_pinn_complex=pred_pinn_complex,
    diff_miracle=diff_miracle,
    diff_svnn=diff_svnn,
    diff_svnn_complex=diff_svnn_complex,
    diff_pinn=diff_pinn,
    diff_pinn_complex=diff_pinn_complex,
    npc_diff=npc_diff,
    path_save=paths.figures / 'fig7.svg'
)
