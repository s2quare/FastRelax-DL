"""Script for generating Figure 5.

- Simulate effect of off-resonances (delta B0) on T1, T2, B0 estimation
- Use Standard DNNs (magnitude-base) and DNNs trained with additional B0 estimation (complex-based), and compare to MIRACLE

Author: florian.birk@tuebingen.mpg.de (Florian Birk), December 2024
"""
# %%
import torch
import numpy as np
import matplotlib.pyplot as plt
from tueplots import figsizes, fonts, fontsizes, axes
from miracle import calc_t1_t2_ima_gss_B1_reg
from fcns_mb import *
from fcns_dataprep import train_val_test_split, apply_transform, revert_min_max_transform
from fcns_fig_plot import plot_fig5
from model import Qbssfp
import paths
from pathlib import Path
import yaml

# manual seed
torch.manual_seed(2308)

path_dat = paths.data / 'in_silico' / 'pred'
path_dat.mkdir(parents=True, exist_ok=True)
pred_fp = path_dat / 'pred.npy'
pred_nob0_fp = path_dat / 'pred_nob0.npy'
pred_miracle_fp = path_dat / 'pred_miracle.npy'
if pred_fp.exists():
    pred = np.load(pred_fp)
    pred_nob0 = np.load(pred_nob0_fp)
    pred_miracle = np.load(pred_miracle_fp)
    shift = torch.from_numpy(np.arange(-0.9, 0.9, 0.01) * np.pi)  # rad
    npc_trained = [12, 6, 4]
    t1 = torch.Tensor([939.0])    # ms
    t2 = torch.Tensor([62.0])     # ms
else:
    # 1a. Simulate the Fn mode variation dependent on B0
    # config params 3T (White Matter) - see Zu et al. (Relaxation Measurements in Brain tissue at field strengths between 0.35T and 9.4T)
    tr = torch.Tensor([4.8])     # ms
    te = torch.Tensor([2.4])      # ms
    flip = torch.Tensor([15.0])   # deg
    t1 = torch.Tensor([939.0])    # ms
    t2 = torch.Tensor([62.0])     # ms
    shift = torch.from_numpy(np.arange(-0.9, 0.9, 0.01) * np.pi)  # rad
    b1 = torch.ones((len(shift), 1))  # u.a.
    npc_trained = [12, 6, 4]  # number of phase cycles
    bssfp_width = 1/(tr*1e-3)    # Hz

    # number of pc values, two models, three/two target parameters
    pred = np.zeros((len(shift), len(npc_trained), 2, 3))
    pred_nob0 = np.zeros((len(shift), len(npc_trained), 2, 2))
    pred_miracle = np.zeros((len(shift), len(npc_trained), 2))

    # load the training data and estimate the sig_mean and sig_std
    path_in_silico = paths.data / 'in_silico' / 'train'
    dnn_mag_fop = paths.dnn_models / 'magnitude-based'
    dnn_complex_fop = paths.dnn_models / 'complex-based'
    # define the DNNs with B0 fitting for all three pc and both models (real/imag + B1 as input)
    svnn_fps = [dnn_complex_fop / '7_sv_inf_uniform_pretty-gorge.pt',
                dnn_complex_fop / '8_sv_inf_uniform_revived-breeze.pt',
                dnn_complex_fop / '9_sv_inf_uniform_sweet-dream.pt']
    pinn_fps = [dnn_complex_fop / '10_mb_inf_uniform_leafy-paper.pt',
                dnn_complex_fop / '11_mb_inf_uniform_splendid-energy.pt',
                dnn_complex_fop / '12_mb_inf_uniform_stellar-mountain.pt']
    # dnns trained without additional B0 fitting (Mag + B1 as input)
    # svnn_nob0_fps = [dnn_standard_fop / '1_sv_inf_uniform_ancient-cherry.pt',
    svnn_nob0_fps = [dnn_mag_fop / '1_sv_inf_uniform_blooming-universe.pt',
                     dnn_mag_fop / '2_sv_inf_uniform_polar-microwave.pt',
                     dnn_mag_fop / '3_sv_inf_uniform_drawn-cherry.pt']
    # pinn_nob0_fps = [dnn_standard_fop / '13_mb_inf_uniform_ethereal-snowball.pt',
    pinn_nob0_fps = [dnn_mag_fop / '4_mb_inf_uniform_wandering-paper.pt',
                     dnn_mag_fop / '5_mb_inf_uniform_sage-smoke.pt',
                     dnn_mag_fop / '6_mb_inf_uniform_sage-deluge.pt']
    svnn = []
    pinn = []
    sig_mean_list = []
    sig_std_list = []

    svnn_nob0 = []
    pinn_nob0 = []
    sig_mean_nob0_list = []
    sig_std_nob0_list = []

    # npc_trained = [12]
    # npc_trained = [12, 6, 4]
    cfg = yaml.safe_load(
        open(paths.cfgs / '1.nn_train' / 'train_in_silico.yml'))
    cfg_nob0 = yaml.safe_load(
        open(paths.cfgs / '1.nn_train' / 'train_in_silico.yml'))
    cfg['model'] = 'sv'
    cfg['snr'] = 'inf'
    cfg['distr'] = 'uniform'
    cfg['b0fit'] = True
    cfg['realimag'] = True
    cfg['param_min'][-1] = cfg['param_min'][-1] * np.pi
    cfg['param_max'][-1] = cfg['param_max'][-1] * np.pi
    cfg_nob0['b0fit'] = False
    cfg_nob0['realimag'] = False
    cfg_nob0['param_min'] = cfg['param_min'][:-1]
    cfg_nob0['param_max'] = cfg['param_max'][:-1]
    cfg_nob0['nneurons'] = [4, 64, 64, 2]

    for i, npc_train in enumerate(npc_trained):
        npc_name = f'{npc_train}pc'
        cfg['npc'] = npc_train
        cfg_nob0['npc'] = npc_train
        sig = torch.load(path_in_silico /
                         f'bssfp_complex_uniform_{npc_name}.pt').cfloat()
        sig_nob0 = torch.load(path_in_silico /
                              f'bssfp_complex_nob0_uniform_{npc_name}.pt').cfloat()
        par = torch.load(path_in_silico /
                         f't1t2b0b1_uniform_{npc_name}.pt').float()
        par_nob0 = torch.load(path_in_silico /
                              f't1t2b0b1_nob0_uniform_{npc_name}.pt').float()
        sig_train, sig_val, sig_test, par_train, par_val, par_test = train_val_test_split(
            sig, par, cfg['split'])
        sig_train_nob0, sig_val_nob0, sig_test_nob0, par_train_nob0, par_val_nob0, par_test_nob0 = train_val_test_split(
            sig_nob0, par_nob0, cfg['split'])

        # calculate the mean and std of the training data for b0 fitting
        cfg['mb'] = False
        model_svnn = Qbssfp(cfg).to('cpu')
        cfg['mb'] = True
        model_pinn = Qbssfp(cfg).to('cpu')
        # load the models
        model_svnn.load_state_dict(torch.load(svnn_fps[i]))
        model_pinn.load_state_dict(torch.load(pinn_fps[i]))
        svnn.append(model_svnn)
        pinn.append(model_pinn)
        # calculate mean and std
        fn_complex = calc_Fn(sig_train, model_svnn.phi_nom_rad,
                             Np=model_svnn.Np, device='cpu')
        fn_complex = norm2_vox(fn_complex)  # normalize using euclidean norm
        sig = torch.cat(
            (torch.real(fn_complex), return_without_f0_imag_2d(fn_complex, Np=1), par_train[:, -1].unsqueeze(-1)), dim=1
        )
        sig_mean_list.append(torch.mean(sig, dim=0))
        sig_std_list.append(torch.std(sig, dim=0))

        # %% calculate the mean and std of the training data for no b0 fitting
        cfg_nob0['mb'] = False
        model_svnn_nob0 = Qbssfp(cfg_nob0).to('cpu')
        cfg_nob0['mb'] = True
        model_pinn_nob0 = Qbssfp(cfg_nob0).to('cpu')
        # load the models
        model_svnn_nob0.load_state_dict(torch.load(svnn_nob0_fps[i]))
        model_pinn_nob0.load_state_dict(torch.load(pinn_nob0_fps[i]))
        svnn_nob0.append(model_svnn_nob0)
        pinn_nob0.append(model_pinn_nob0)
        # calculate mean and std
        fn_complex = calc_Fn(sig_train_nob0, model_svnn_nob0.phi_nom_rad,
                             Np=model_svnn_nob0.Np, device='cpu')
        fn_complex = norm2_vox(fn_complex)  # normalize using euclidean norm
        sig_nob0 = torch.cat(
            (torch.abs(fn_complex), par_train_nob0[:, -1].unsqueeze(-1)), dim=1)
        sig_mean_nob0_list.append(torch.mean(sig_nob0, dim=0))
        sig_std_nob0_list.append(torch.std(sig_nob0, dim=0))

    # simulate bssfp data and predict
    for i, npc in enumerate(npc_trained):
        # phase cycle vector
        phi_pc_hz = torch.from_numpy(np.arange(bssfp_width/(2*npc), bssfp_width,
                                               bssfp_width/npc))  # [Hz]
        phi_pc_rad = phi_pc_hz*tr*1e-3*2*torch.pi
        sim_bssfp_temp = torch.zeros(
            (len(shift), len(phi_pc_hz)), dtype=torch.cfloat)
        # sim_bssfp_temp_nophasecorr = torch.zeros(
        #     (len(shift), len(phi_pc_hz)), dtype=torch.cfloat)
        fn_complex_temp = torch.zeros((len(shift), 3), dtype=torch.cfloat)
        # fn_complex_temp_nophasecorr = torch.zeros(
        #     (len(shift), 3), dtype=torch.cfloat)
        for j, b0 in enumerate(shift):
            sim_bssfp_temp[j, :] = sim_bssfp(
                phi_nom_hz=phi_pc_hz,
                tr=tr,
                te=te,
                M0=1,
                fa_nom=flip,
                b1=b1[0, 0],
                t1=t1,
                t2=t2,
                b0=b0)
        # perform phase correction on the complex bssfp signal
        sim_bssfp_temp = phase_correction_2d(sim_bssfp_temp)
        fn_complex_temp = calc_Fn(
            sim_bssfp_temp, phi_pc_rad, Np=1)
        fn_complex_temp = norm2_vox(fn_complex_temp)

        # sig = torch.cat((fn_mag_norm, fn_pha, b1), dim=1)
        sig = torch.cat(
            (torch.real(fn_complex_temp), return_without_f0_imag_2d(fn_complex_temp, Np=1), b1), dim=1)
        sig_nob0 = torch.cat(
            (torch.abs(fn_complex_temp), b1), dim=1)

        sig_z = apply_transform(
            sig, sig_mean_list[i], sig_std_list[i], 'cpu')
        sig_nob0_z = apply_transform(
            sig_nob0, sig_mean_nob0_list[i], sig_std_nob0_list[i], 'cpu')

        # get parameters in correct units
        pred_pinn = getparams(pinn[i](sig_z).detach(), pinn[i].param_min,
                              pinn[i].param_max, 'cpu').numpy()
        pred_pinn_nob0 = getparams(pinn_nob0[i](sig_nob0_z).detach(), pinn_nob0[i].param_min,
                                   pinn_nob0[i].param_max, 'cpu').numpy()
        pred_svnn = revert_min_max_transform(svnn[i](sig_z).detach(), svnn[i].param_min,
                                             svnn[i].param_max).numpy()
        pred_svnn_nob0 = revert_min_max_transform(svnn_nob0[i](sig_nob0_z).detach(), svnn_nob0[i].param_min,
                                                  svnn_nob0[i].param_max).numpy()

        # # MIRACLE
        # prepare input
        f_abs = torch.reshape(sig_nob0[:, :3], (len(shift)//2, 2, 3))
        fm1 = f_abs[..., 0].unsqueeze(-1).numpy()
        f0 = f_abs[..., 1].unsqueeze(-1).numpy()
        f1 = f_abs[..., 2].unsqueeze(-1).numpy()
        b1_miracle = np.ones_like(f0)
        mask = np.full_like(f0, True)
        tr = 4.8  # ms
        te = 2.4  # ms
        fa_nom = 15  # deg
        t1_est = 1000

        # estimate T1 and T2
        t1_miracle, t2_miracle, _ = calc_t1_t2_ima_gss_B1_reg(
            f1=f1,
            f0=f0,
            fm1=fm1,
            b1=b1_miracle,
            tr=tr,
            te=te,
            flipangle=fa_nom,
            t1_est=t1_est,
            mask=mask)

        t1_miracle = np.reshape(t1_miracle, (len(shift)))
        t2_miracle = np.reshape(t2_miracle, (len(shift)))

        # Assign predictions to pred array
        # number of pc values, two models, three target parameters, b0 fit or not
        pred[:, i, 0, :] = pred_svnn
        pred[:, i, 1, :] = pred_pinn
        pred_nob0[:, i, 0, :] = pred_svnn_nob0
        pred_nob0[:, i, 1, :] = pred_pinn_nob0
        pred_miracle[:, i, 0] = t1_miracle
        pred_miracle[:, i, 1] = t2_miracle
    print('Saving the predictions')
    np.save(pred_fp, pred)
    np.save(pred_nob0_fp, pred_nob0)
    np.save(pred_miracle_fp, pred_miracle)


# %% Plot relative errors
plot_fig5(
    pred=pred,
    pred_nob0=pred_nob0,
    pred_miracle=pred_miracle,
    npc_trained=npc_trained,
    shift=shift,
    path_save=paths.figures / 'fig5.png'
)
