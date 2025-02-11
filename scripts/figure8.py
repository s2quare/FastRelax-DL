"""Script for generating Figure 8.

- Train magnitude-based DNNs (SVNN, PINN) with inf SNR and uniform distribution. 
a)
- Compute and plot the CoD between in vivo prediction from final epoch vs. x epoch
- Plot the validation loss along number of epochs for SVNN and PINN
b)
- Compare single epoch to final epoch
- Show the whole-brain (axial, coronal, sagittal) parameter maps for T1 and T2 for SVNN, PINN (single epoch and final epoch)

Author: florian.birk@tuebingen.mpg.de (Florian Birk), Februar 2024
"""
# %% Import
import paths
from model import Qbssfp
from fcns_mb import getparams, calc_Fn, norm2_vox
from fcns_model import train, test, exec_mb
from fcns_dataprep import train_val_test_split, apply_transform, min_max_transform, revert_min_max_transform, revert_z_transform
from fcns_performance import cod, rmse
from fcns_fig_plot import plot_fig8
from datasets import UnsupervisedDataset, SupervisedDataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import nibabel as nib
import wandb
import yaml


if __name__ == '__main__':
    # Set random seed
    torch.manual_seed(2308)
    # cfgs
    framework_list = ['sv', 'mb']
    snr = 'inf'
    distr = 'uniform'
    # paths
    path_maps = paths.data / 'in_vivo' / 'test' / 'maps'
    path_metrics = paths.data / 'in_vivo' / 'metrics'
    path_models = paths.dnn_models / 'fig8'
    path_in_vivo = paths.data / 'in_vivo' / 'test'
    path_models.mkdir(parents=True, exist_ok=True)
    path_metrics.mkdir(parents=True, exist_ok=True)

    nepochs_save = 10
    path_cod = path_metrics / f'fig8-cod-{distr}.npy'
    path_loss_train = path_metrics / f'fig8-loss_train-{distr}.npy'
    path_loss_val = path_metrics / f'fig8-loss_val-{distr}.npy'
    path_time = path_metrics / f'fig8-time-{distr}.npy'
    path_pred_array = path_maps / f'fig8-pred_array-{distr}.npy'
    path_pred_array_init = path_maps / f'fig8-pred_array_init-{distr}.npy'

    if path_pred_array.exists() and path_pred_array_init.exists() and path_cod.exists() and path_loss_train.exists() and path_loss_val.exists() and path_time.exists():
        print('Data already exists. Loading npy files...')
        cods = np.load(path_cod)
        loss_train = np.load(path_loss_train)
        loss_val = np.load(path_loss_val)
        time = np.load(path_time)
        pred_array = np.load(path_pred_array)
    else:
        # initialize CoD array
        # (framework, epochs, T1/T2, wm/gm/csf/brain)
        cods = np.zeros((len(framework_list), 300, 2, 3))
        loss_train = np.zeros((len(framework_list), 300))
        loss_val = np.zeros((len(framework_list), 300))
        time = np.zeros((len(framework_list), 300))

        sig_in_vivo_all = torch.load(
            path_in_vivo / 'sig_all_nn.12pc.sub1.pt').float()
        sig_in_vivo_all = torch.cat(
            (sig_in_vivo_all[:, :3], sig_in_vivo_all[:, -1:]), dim=-1)
        mask_wm_3d = nib.load(
            path_in_vivo / 't1_mprage.hd-bet.reg_pve_2.nii.gz').get_fdata()
        mask_gm_3d = nib.load(
            path_in_vivo / 't1_mprage.hd-bet.reg_pve_1.nii.gz').get_fdata()
        mask_csf_3d = nib.load(
            path_in_vivo / 't1_mprage.hd-bet.reg_pve_0.nii.gz').get_fdata()
        mask_brain_3d = np.zeros(mask_csf_3d.shape)
        mask_brain_3d[mask_wm_3d > 0] = 1
        mask_brain_3d[mask_gm_3d > 0] = 1

        # reshape masks to 1d
        mask_csf = mask_csf_3d.reshape(-1).astype(bool)
        mask_gm = mask_gm_3d.reshape(-1).astype(bool)
        mask_wm = mask_wm_3d.reshape(-1).astype(bool)
        mask_brain = mask_brain_3d.reshape(-1).astype(bool)

        # init pred array
        pred_array = np.zeros(
            (len(framework_list), mask_brain_3d.shape[0], mask_brain_3d.shape[1], mask_brain_3d.shape[2], 2, nepochs_save+1))

        # iterate over all frameworks
        for m, model_type in enumerate(framework_list):
            print(f'Run training for {model_type} model')
            # load cfg
            cfg = yaml.safe_load(
                open(paths.cfgs / '1.nn_train' / 'train_in_silico.yml'))
            cfg['model'] = model_type
            cfg['snr'] = snr
            cfg['distr'] = distr
            cfg['b0fit'] = False
            cfg['realimag'] = False
            cfg['npc'] = 12
            cfg['nneurons'] = [4, 64, 64, 2]
            cfg['param_min'] = cfg['param_min'][:-1]
            cfg['param_max'] = cfg['param_max'][:-1]
            if distr == 'uniform_ext':
                cfg['param_max'] = cfg['param_max_ext']
            if model_type == 'mb':
                cfg['mb'] = True
            else:
                cfg['mb'] = False

            print('++++++ CONFIG ++++++')
            if model_type == 'mb':
                print('Model: Semi-Supervised (model-based) training selected')
            elif model_type == 'sv':
                print('Model: Supervised training selected')
            else:
                print('Wrong model selection')
            print(f'Selected SNR level: {snr}')
            print(f'Target parameter distribution: {distr}')
            print('++++++++++++++++++++\n')

            # %% Load data (in silico)
            path_in_silico = paths.data / 'in_silico' / 'train'
            sig = torch.load(path_in_silico /
                             f'bssfp_complex_nob0_{distr}_12pc.pt').cfloat()
            par = torch.load(path_in_silico /
                             f't1t2b0b1_nob0_{distr}_12pc.pt').float()
            b1 = par[:, -1]
            # take the number of phase cycles based on config
            if cfg['npc'] == 4:
                print('Take every third phase cycles from in silico data')
                sig = sig[:, ::3]   # every third column
            elif cfg['npc'] == 6:
                print('Take every second phase cycles from in silico data')
                sig = sig[:, ::2]   # every second column
            else:
                print('Take all phase cycles from in silico data')

            # %% Split data
            sig_train, sig_val, sig_test, par_train, par_val, par_test = train_val_test_split(
                sig, par, cfg['split'])
            # take only T1 and T2
            b1_train = par_train[:, -1].unsqueeze(-1)
            b1_val = par_val[:, -1].unsqueeze(-1)
            b1_test = par_test[:, -1].unsqueeze(-1)
            b0_train = par_train[:, 2].unsqueeze(-1)
            b0_val = par_val[:, 2].unsqueeze(-1)
            b0_test = par_test[:, 2].unsqueeze(-1)
            par_train = par_train[:, :2]
            par_val = par_val[:, :2]
            par_test = par_test[:, :2]

            # %% Initialize model
            # set wandb disabled
            wandb.disabled = True
            wandb.init(
                project=cfg['project'],
                entity=cfg['entity'],
                notes=cfg['notes'],
                config=cfg)
            model = Qbssfp(cfg).to(cfg['device'])
            loss = nn.MSELoss().to(cfg['device'])
            optimizer = torch.optim.Adam(
                model.parameters(), lr=cfg['lr'])
            print(f'Model Learning rate: {cfg["lr"]}\n')

            # calculate the mean and std of the training data
            # calculate the modes and norm2vox for simulation data
            fn = calc_Fn(sig, model.phi_nom_rad,
                         Np=model.Np, device=cfg['device'])
            fn_norm = torch.abs(norm2_vox(fn))
            fn_mean, fn_std = torch.mean(
                fn_norm, dim=0), torch.std(fn_norm, dim=0)
            b1_mean = torch.mean(b1)
            b1_std = torch.std(b1)
            sig_mean = torch.cat((fn_mean, b1_mean.unsqueeze(-1)))
            sig_std = torch.cat((fn_std, b1_std.unsqueeze(-1)))

            # %% get noise level based on snr
            if cfg['snr'] == 'inf':
                noise_level = 0
            else:
                # sig_sim_mag: Average simulated signal of the magnitude of the complex sum
                # snr: SNR of the magnitude of the complex sum
                noise_level_mag = torch.tensor(
                    cfg['sig_sim_mag']/int(cfg['snr']))
                noise_level = noise_level_mag / torch.sqrt(torch.tensor(2.0))
            print(
                f'Noise level added to pc-bSSFP (with SNR {cfg["snr"]}): {noise_level}\n')

            # %% Apply transform to in vivo data
            # apply transform to in vivo data
            sig_in_vivo_all_z = apply_transform(
                sig_in_vivo_all, sig_mean, sig_std, cfg['device'])

            # %% Prepare dataloader
            if model_type == 'mb':
                trainloader = DataLoader(UnsupervisedDataset(
                    sig_train, b1_train, b0_train), batch_size=cfg['mbatch'], shuffle=True)
                valloader = DataLoader(UnsupervisedDataset(
                    sig_val, b1_val, b0_train), batch_size=cfg['mbatch'], shuffle=True)
            else:
                # apply transform to target parameter
                par_train_transformed = min_max_transform(
                    par_train, cfg['param_min'], cfg['param_max'])
                par_val_transformed = min_max_transform(
                    par_val, cfg['param_min'], cfg['param_max'])
                trainloader = DataLoader(SupervisedDataset(
                    sig_train, par_train_transformed, b1_train, b0_train), batch_size=cfg['mbatch'], shuffle=True, pin_memory=True)
                valloader = DataLoader(SupervisedDataset(
                    sig_val, par_val_transformed, b1_val, b0_train), batch_size=cfg['mbatch'], shuffle=True, pin_memory=True)

            # %% infer params for initialized model (for SVNN model only)
            if model_type == 'sv':
                model_init = Qbssfp(cfg).to(cfg['device'])
                model_init.eval()
                par_pred = model_init(sig_in_vivo_all_z)
                par_pred_init = revert_min_max_transform(
                    par_pred.detach(), cfg['param_min'], cfg['param_max']).numpy()
                # reshape
                pred_array_init = np.reshape(par_pred_init, (
                    mask_brain_3d.shape[0], mask_brain_3d.shape[1], mask_brain_3d.shape[2], 2))

                # save par_pred_init
                np.save(path_pred_array_init, pred_array_init)

            # %% Model training
            best_model, time_list, train_loss_list, val_loss_list = train(
                model=model,
                trainloader=trainloader,
                valloader=valloader,
                sig_mean=sig_mean,
                sig_std=sig_std,
                noise_level=noise_level,
                loss=loss,
                optimizer=optimizer,
                epochs=cfg['nepochs'],
                patience=cfg['patience'],
                mb=cfg['mb'],    # False refers to supervised training
                device=cfg['device'],
                wandb=wandb,
                path_save=path_models,
                return_epochs=True
            )

            # List all model paths
            models_list = list(path_models.glob(f'*{model_type}.pt'))
            models_list.sort(key=lambda f: int(f.stem.split('-')[0]))

            # infere parameters with final epoch model
            if best_model.mb:
                # Get NN output MRI signal --> tissue parameter
                x = best_model(sig_in_vivo_all_z)
                # get parameters from model (in 2d shape)
                par_pred = getparams(x, best_model.param_min,
                                     best_model.param_max, best_model.device)
                par_pred_final = par_pred.detach().numpy()
            else:   # supervised training
                par_pred = best_model(sig_in_vivo_all_z)
                par_pred_final = revert_min_max_transform(
                    par_pred.detach(), cfg['param_min'], cfg['param_max']).numpy()

            # iterate over all epochs
            j = 0
            for i, model_path in enumerate(models_list):
                # load model weights
                # load model
                model_epoch = Qbssfp(cfg).to(cfg['device'])
                model_epoch.load_state_dict(torch.load(model_path))
                model_epoch.eval()
                with torch.no_grad():
                    if model_epoch.mb:
                        # Get NN output MRI signal --> tissue parameter
                        x = model_epoch(sig_in_vivo_all_z)
                        # get parameters from model (in 2d shape)
                        par_pred = getparams(x, model_epoch.param_min,
                                             model_epoch.param_max, model_epoch.device)
                        par_pred_epoch = par_pred.detach().numpy()
                    else:   # supervised training
                        par_pred = model_epoch(sig_in_vivo_all_z)
                        par_pred_epoch = revert_min_max_transform(
                            par_pred.detach(), cfg['param_min'], cfg['param_max']).numpy()

                # save pred array if below nepochs_save or final epoch
                if i < nepochs_save or i == len(models_list)-1:
                    pred_array[m, :, :, :, :, j] = np.reshape(par_pred_epoch, (
                        mask_brain_3d.shape[0], mask_brain_3d.shape[1], mask_brain_3d.shape[2], 2))
                    j += 1

                # 2. Compute evaluation metrics (CoD, Diff) for WM, GM, CSF and Brian masks
                par_pred_epoch_2d_wm, par_pred_final_2d_wm = par_pred_epoch[mask_wm,
                                                                            :], par_pred_final[mask_wm, :]
                par_pred_epoch_2d_gm, par_pred_final_2d_gm = par_pred_epoch[mask_gm,
                                                                            :], par_pred_final[mask_gm, :]
                par_pred_epoch_2d_csf, par_pred_final_2d_csf = par_pred_epoch[
                    mask_csf, :], par_pred_final[mask_csf, :]
                par_pred_epoch_2d_brain, par_pred_final_2d_brain = par_pred_epoch[
                    mask_brain, :], par_pred_final[mask_brain, :]

                # calculate cod
                cod_wm = cod(par_pred_final_2d_wm, par_pred_epoch_2d_wm)
                cod_gm = cod(par_pred_final_2d_gm, par_pred_epoch_2d_gm)
                cod_csf = cod(par_pred_final_2d_csf, par_pred_epoch_2d_csf)
                cod_brain = cod(par_pred_final_2d_brain,
                                par_pred_epoch_2d_brain)
                cods[m, i, :, 0] = cod_wm
                cods[m, i, :, 1] = cod_brain
                cods[m, i, :, 2] = cod_gm
                # add loss to array
                len_loss = len(train_loss_list)
                loss_train[m, :len_loss] = train_loss_list
                loss_val[m, :len_loss] = val_loss_list
                # add time list to array
                len_time = len(time_list)
                time[m, :len_time] = time_list

        # save arrays
        print('Saving metrics and pred_array...')
        np.save(path_cod, cods)
        np.save(path_loss_train, loss_train)
        np.save(path_loss_val, loss_val)
        np.save(path_time, time)
        np.save(path_pred_array, pred_array)

    # create slices for plotting
    # Get Axial, Coronal and Sagittal slices
    par_4d_dnns_single = pred_array[..., 0]
    par_4d_dnns_final = pred_array[..., -1]

    # Apply mask to parameter maps
    # load mask
    mask_3d = nib.load(
        path_in_vivo / 't1_mprage.hd-bet_mask.reg.nii.gz').get_fdata()
    mask_3d = mask_3d.astype(bool)
    for i in range(par_4d_dnns_single.shape[-1]):
        for m in range(len(framework_list)):
            par_4d_dnns_single[m, ...,
                               i] = par_4d_dnns_single[m, ..., i] * mask_3d
            par_4d_dnns_final[m, ...,
                              i] = par_4d_dnns_final[m, ..., i] * mask_3d

    slices = [102, 60, 69]
    axi_r = [25, 162, 5, 123]  # axial range, [h1, h2, w1, w2]
    cor_r = [32, 135, 8, 118]  # coronal range, [h1, h2, w1, w2]
    sag_r = [28, 136, 12, 151]  # sagittal range, [h1, h2, w1, w2]
    # Axial
    par_sv_single_axi = np.rot90(par_4d_dnns_single[0, ...], axes=(0, 1))[
        axi_r[0]:axi_r[1], axi_r[2]:axi_r[3], slices[0], :]
    par_mb_single_axi = np.rot90(par_4d_dnns_single[1, ...], axes=(0, 1))[
        axi_r[0]:axi_r[1], axi_r[2]:axi_r[3], slices[0], :]
    par_sv_final_axi = np.rot90(par_4d_dnns_final[0, ...], axes=(0, 1))[
        axi_r[0]:axi_r[1], axi_r[2]:axi_r[3], slices[0], :]
    par_mb_final_axi = np.rot90(par_4d_dnns_final[1, ...], axes=(0, 1))[
        axi_r[0]:axi_r[1], axi_r[2]:axi_r[3], slices[0], :]
    # Coronal
    par_sv_single_cor = np.rot90(par_4d_dnns_single[0, ...], axes=(0, 2))[
        cor_r[0]:cor_r[1], slices[1], cor_r[2]:cor_r[3], :]
    par_mb_single_cor = np.rot90(par_4d_dnns_single[1, ...], axes=(0, 2))[
        cor_r[0]:cor_r[1], slices[1], cor_r[2]:cor_r[3], :]
    par_sv_final_cor = np.rot90(par_4d_dnns_final[0, ...], axes=(0, 2))[
        cor_r[0]:cor_r[1], slices[1], cor_r[2]:cor_r[3], :]
    par_mb_final_cor = np.rot90(par_4d_dnns_final[1, ...], axes=(0, 2))[
        cor_r[0]:cor_r[1], slices[1], cor_r[2]:cor_r[3], :]
    # Sagittal
    par_sv_single_sag = np.rot90(par_4d_dnns_single[0, ...], axes=(1, 2))[
        slices[2], sag_r[0]:sag_r[1], sag_r[2]:sag_r[3], :]
    par_mb_single_sag = np.rot90(par_4d_dnns_single[1, ...], axes=(1, 2))[
        slices[2], sag_r[0]:sag_r[1], sag_r[2]:sag_r[3], :]
    par_sv_final_sag = np.rot90(par_4d_dnns_final[0, ...], axes=(1, 2))[
        slices[2], sag_r[0]:sag_r[1], sag_r[2]:sag_r[3], :]
    par_mb_final_sag = np.rot90(par_4d_dnns_final[1, ...], axes=(1, 2))[
        slices[2], sag_r[0]:sag_r[1], sag_r[2]:sag_r[3], :]

    par_sv_single_slices = [par_sv_single_axi,
                            par_sv_single_cor, par_sv_single_sag]
    par_mb_single_slices = [par_mb_single_axi,
                            par_mb_single_cor, par_mb_single_sag]
    par_sv_final_slices = [par_sv_final_axi,
                           par_sv_final_cor, par_sv_final_sag]
    par_mb_final_slices = [par_mb_final_axi,
                           par_mb_final_cor, par_mb_final_sag]

    # %% set all values to nan where 0
    par_sv_single_slices = [np.where(par_sv_single_slices[i] == 0, np.nan,
                                     par_sv_single_slices[i]) for i in range(len(par_sv_single_slices))]
    par_mb_single_slices = [np.where(par_mb_single_slices[i] == 0, np.nan,
                                     par_mb_single_slices[i]) for i in range(len(par_mb_single_slices))]
    par_sv_final_slices = [np.where(par_sv_final_slices[i] == 0, np.nan,
                                    par_sv_final_slices[i]) for i in range(len(par_sv_final_slices))]
    par_mb_final_slices = [np.where(par_mb_final_slices[i] == 0, np.nan,
                                    par_mb_final_slices[i]) for i in range(len(par_mb_final_slices))]

    # %% Plot fig 7
    plot_fig8(
        cods=cods,
        loss_val=loss_val,
        nepochs=300,
        par_sv_single=par_sv_single_slices,
        par_mb_single=par_mb_single_slices,
        par_sv_final=par_sv_final_slices,
        par_mb_final=par_mb_final_slices,
        vmin=[0, 0],
        vmax=[1500, 120],
        cmaps=['inferno', 'viridis'],
        min_cod=0.925,
        path_save=paths.figures / f'fig8.svg'
    )
