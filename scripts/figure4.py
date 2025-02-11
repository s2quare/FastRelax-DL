"""Script for generating Figure 4.

- Comparison between SVNN vs. PINN vs. MIRACLE estimation of T1 and T2 on one unseen test subject
- SVNN and PINN trained with all distributions
- Compare to MIRACLE estimates with difference maps
- Train SNR DNNs: inf

Author: florian.birk@tuebingen.mpg.de (Florian Birk), February 2024
"""
# %% Import
# own modules
import paths
from miracle import calc_t1_t2_ima_gss_B1_reg
from model import Qbssfp
from fcns_mb import getparams, calc_Fn, norm2_vox
from fcns_model import apply_transform
from fcns_dataprep import revert_min_max_transform
from fcns_fig_plot import plot_fig4
# external modules
import torch
import numpy as np
import yaml


def prepare_in_silico(path_in_silico, distr, model):
    sig = torch.load(path_in_silico /
                     f'bssfp_complex_nob0_{distr}_12pc.pt').cfloat()
    par = torch.load(path_in_silico /
                     f't1t2b0b1_nob0_{distr}_12pc.pt').float()
    b1 = par[:, -1]
    fn = calc_Fn(sig, model.phi_nom_rad,
                 Np=model.Np, device='cpu')
    fn_norm = torch.abs(norm2_vox(fn))
    fn_mean, fn_std = torch.mean(fn_norm, dim=0), torch.std(fn_norm, dim=0)
    b1_mean = torch.mean(b1)
    b1_std = torch.std(b1)
    sig_input = torch.cat((fn_norm, b1.unsqueeze(-1)), dim=1)
    sig_mean = torch.cat((fn_mean, b1_mean.unsqueeze(-1)))
    sig_std = torch.cat((fn_std, b1_std.unsqueeze(-1)))
    return sig_input, par, sig_mean, sig_std


def mask_predictions(par_pred, mask):
    for i in range(par_pred.shape[-1]):
        par_pred[:, :, :, i] = par_pred[:, :, :, i] * mask
    return par_pred


def estimate_in_vivo_distr(model, cfg, sig_sub_nn_z, mask):
    """Estimate parameter for model"""
    # get parameters from models (in 2d shape)
    if cfg['mb']:
        par_pred = getparams(model(sig_sub_nn_z), model.param_min,
                             model.param_max, 'cpu').detach().numpy()
    else:
        par_pred = revert_min_max_transform(
            model(sig_sub_nn_z), cfg['param_min'], cfg['param_max']).detach().numpy()
    # reshape model predictions to mask shape
    par_pred = np.rot90(np.reshape(
        par_pred, (mask.shape[0], mask.shape[1], mask.shape[2], 2)), axes=(0, 1))
    # mask predictions
    par_pred_masked = mask_predictions(
        par_pred, np.rot90(mask, axes=(0, 1)))
    return par_pred_masked


if __name__ == '__main__':
    # %% Set random seed
    torch.manual_seed(2308)
    # cfgs
    path_in_silico = paths.data / 'in_silico' / 'train'
    path_in_vivo = paths.data / 'in_vivo' / 'test'
    path_maps = paths.data / 'in_vivo' / 'test' / 'maps'
    path_models = paths.dnn_models / 'magnitude-based' / 'trained_snr'
    snr_train_idx = 0
    # get list of all distr models with snr = inf trained on
    model_list = list(path_models.glob('*_inf_*.pt'))
    # sort list by numbers split by '_' in filename
    model_list.sort(key=lambda f: int(''.join(filter(str.isdigit, f.name))))
    framework_list = ['sv', 'mb']
    distr_list = ['in_vivo', 'uniform', 'uniform_ext']
    distr_title_list = ['in vivo', 'uniform', 'extended']

    # Load in vivo test data
    sig_sub = torch.load(path_in_vivo / 'sig_all_nn.12pc.sub1.pt').float()
    b1_sub = sig_sub[:, -1]
    sig_sub_input = torch.cat(
        (sig_sub[:, :3], sig_sub[:, -1].unsqueeze(-1)), dim=1)
    mask = torch.load(path_in_vivo / 'mask_3d.sub1.pt').float().numpy()

    # Load data if available or run inference otherwise
    path_dnn_sub_pred = path_maps / 'fig4-dnn_sub_pred.npy'
    path_miracle_sub_pred = path_maps / 'fig4-miracle_sub_pred.npy'
    if path_dnn_sub_pred.exists() and path_miracle_sub_pred.exists():
        print('DNN and MIRACLE output already exists. Loading it...')
        par_pred_dnn = np.load(path_dnn_sub_pred)
        par_pred_miracle = np.load(path_miracle_sub_pred)
    else:
        if not path_dnn_sub_pred.exists():
            print('DNN output does not exist. Running inference...')
            path_dnn_sub_pred.parent.mkdir(parents=True, exist_ok=True)
            # initialize empty array for DNN whole-brain parameter predictions
            par_pred_dnn = np.zeros(
                (len(framework_list), len(distr_list), mask.shape[1], mask.shape[0], mask.shape[2], 2))

            # iterate over models and estimate parameter predictions for all 3 distributions and 2 frameworks
            for m, model_type in enumerate(framework_list):
                for d, distr in enumerate(distr_list):
                    # take the path from model_list where the path matches model_type and distr
                    path_model = [
                        path for path in model_list if model_type in path.name and distr in path.name][0]
                    print(f'Using Model: {path_model.name}')
                    print(
                        f'Estimate parameter predictions for {model_type} model trained on {distr} distribution')
                    # cfgs
                    cfg = yaml.safe_load(
                        open(paths.cfgs / '1.nn_train' / 'train_in_silico.yml'))
                    if model_type == 'mb':
                        cfg['mb'] = True
                    else:
                        cfg['mb'] = False
                    if distr == 'uniform_ext':
                        cfg['param_max'] = cfg['param_max_ext']
                    cfg['b0fit'] = False
                    cfg['realimag'] = False
                    cfg['npc'] = 12
                    cfg['nneurons'] = [4, 64, 64, 2]
                    cfg['param_min'] = cfg['param_min'][:-1]
                    cfg['param_max'] = cfg['param_max'][:-1]

                    # initialize model
                    model = Qbssfp(cfg)
                    model.load_state_dict(torch.load(path_model))

                    # get sig_mean and sig_std dependent on distribution
                    sig, par, sig_mean, sig_std = prepare_in_silico(
                        path_in_silico, distr, model)

                    # Apply transforms from different distributions to same in vivo data
                    sig_sub_uniform_z = apply_transform(
                        sig_sub_input, sig_mean, sig_std, 'cpu')

                    # change model to eval mode
                    model.eval()

                    # Perform parameter estimation
                    par_pred_dnn[m, d, :, :, :, :] = estimate_in_vivo_distr(
                        model, cfg, sig_sub_uniform_z, mask)

                    del model, cfg

            # save DNN output
            np.save(path_dnn_sub_pred, par_pred_dnn)

        if not path_miracle_sub_pred.exists():
            print('MIRACLE output does not exist. Running inference...')
            path_miracle_sub_pred.parent.mkdir(parents=True, exist_ok=True)
            # cfg
            cfg_miracle = yaml.safe_load(
                open(paths.cfgs / '2.miracle_est' / 'miracle.yml'))

            sig_sub_4d = torch.reshape(
                sig_sub_input, (mask.shape[0], mask.shape[1], mask.shape[2], 4))
            fm1 = sig_sub_4d[:, :, :, 0].numpy()
            f0 = sig_sub_4d[:, :, :, 1].numpy()
            f1 = sig_sub_4d[:, :, :, 2].numpy()
            b1 = sig_sub_4d[:, :, :, 3].numpy()
            # run miracle
            t1_miracle, t2_miracle, iter_map = calc_t1_t2_ima_gss_B1_reg(
                f1=f1,
                f0=f0,
                fm1=fm1,
                b1=b1,
                tr=cfg_miracle['tr'],
                te=cfg_miracle['te'],
                flipangle=cfg_miracle['fa_nom'],
                t1_est=cfg_miracle['t1_est'],
                mask=mask
            )
            # reshape miracle output to mask shape
            t1_miracle = np.reshape(
                t1_miracle, (mask.shape[0], mask.shape[1], mask.shape[2]))
            t2_miracle = np.reshape(
                t2_miracle, (mask.shape[0], mask.shape[1], mask.shape[2]))
            # rotate the miracle output to match the DNN output
            t1_miracle = np.rot90(t1_miracle, axes=(0, 1))
            t2_miracle = np.rot90(t2_miracle, axes=(0, 1))
            par_pred_miracle = np.stack((t1_miracle, t2_miracle), axis=-1)
            # save miracle output
            np.save(path_miracle_sub_pred, par_pred_miracle)

    # calculate the absolute difference between the miracle predictions and each distribution / framework predictions
    path_diff_dnn_sub_pred = path_maps / 'fig4-diff_dnn_sub_pred.npy'
    if path_diff_dnn_sub_pred.exists():
        print('DNN difference output already exists. Loading it...')
        par_diff_dnn = np.load(path_diff_dnn_sub_pred)
    else:
        print('DNN difference output does not exist. Calculating it...')
        par_diff_dnn = np.zeros_like(par_pred_dnn)
        for m, model_type in enumerate(framework_list):
            for d, distr in enumerate(distr_list):
                par_diff_dnn[m, d, :, :, :, :] = par_pred_dnn[m,
                                                              d, :, :, :, :] - par_pred_miracle
        # save DNN difference output
        np.save(path_diff_dnn_sub_pred, par_diff_dnn)

    # set all values outside the mask to nan
    mask = np.rot90(mask, axes=(0, 1))
    for m in range(par_diff_dnn.shape[0]):
        for d in range(par_diff_dnn.shape[1]):
            for t in range(par_diff_dnn.shape[-1]):
                par_diff_dnn[m, d, :, :, :, t] = np.where(
                    mask != 0, par_diff_dnn[m, d, :, :, :, t], np.nan)
                par_pred_dnn[m, d, :, :, :, t] = np.where(
                    mask != 0, par_pred_dnn[m, d, :, :, :, t], np.nan)
    for t in range(par_pred_miracle.shape[-1]):
        par_pred_miracle[:, :, :, t] = np.where(
            mask != 0, par_pred_miracle[:, :, :, t], np.nan)

    # %% Plot figure
    plot_fig4(
        par_pred_dnn=par_pred_dnn,
        par_pred_miracle=par_pred_miracle,
        par_diff_dnn=par_diff_dnn,
        distr_list=distr_title_list,
        slice=105,
        cmap_list=['inferno', 'viridis', 'seismic'],
        vmin_t1t2=[360, 20],
        vmax_t1t2=[1500, 120],
        vmax_diff=[40, 10],
        path_save=paths.figures / 'fig4.svg')
