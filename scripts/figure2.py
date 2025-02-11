"""Script for generating Figure 2.

- Relative error (Accuracy) in T1/T2 estimation of DNNs trained without noise (SNR = inf) and in vivo noise (SNR =25) on test data with SNR = 25. 
- DNNs trained with different frameworks models (SVNN, PINN) and different distributions (in_vivo, uniform, uniform extended)
- Use Mean of MC simulations for accuracy calculation

Author: florian.birk@tuebingen.mpg.de (Florian Birk), February 2024
"""
# %% Import
import paths
from fcns_fig_plot import plot_fig2
import torch
import numpy as np
import yaml

if __name__ == '__main__':
    # Set random seed
    torch.manual_seed(2308)

    # cfgs
    cfg = yaml.safe_load(
        open(paths.cfgs / '1.nn_train' / 'train_in_silico.yml'))
    par_min = cfg['param_min'][:2]
    par_max = cfg['param_max'][:2]
    path_in_silico = paths.data / 'in_silico' / 'test'
    path_mc = paths.data / 'in_silico' / 'mc'
    snr_test_list = ['inf', 50, 45, 40, 35, 30, 25, 20, 15, 10]
    snr_train_list = ['inf', 50, 25, 10]
    distr_title_list = ['in vivo', 'uniform', 'extended']
    mc_sample = 5000

    # flip t1 and t2 mesh and mc results along T1
    # load t1 and t2 mesh again
    t1_mesh = torch.load(path_in_silico /
                         't1_mesh_uniform_test.pt').float().numpy()
    t2_mesh = torch.load(path_in_silico /
                         't2_mesh_uniform_test.pt').float().numpy()
    t1_mesh_flip = np.flip(t1_mesh, axis=0)
    t2_mesh_flip = np.flip(t2_mesh, axis=0)
    t1t2_mesh = np.stack((t1_mesh_flip, t2_mesh_flip), axis=-1)

    # calculate the relative error/accuracy in percentage ((mc_mean - gt)/gt)*100
    i_test = 6  # 6: snr = 25, 9: snr = 10
    i_train = 2  # 2: snr = 25, training SNR for Fig. 2b
    snr = snr_test_list[i_test]
    print(f'Compute Accuracy on test data with SNR {snr}')
    dnn_acc = np.zeros((2, len(snr_train_list), len(
        distr_title_list), t1_mesh.shape[0], t1_mesh.shape[1], 2))
    miracle_acc = np.zeros((t1_mesh.shape[0], t1_mesh.shape[1], 2))
    path_dnn_mc = path_mc / \
        f'dnn_{mc_sample}-samples_grid.npy'
    path_miracle_mc = path_mc / \
        f'miracle_{mc_sample}-samples_grid.npy'
    dnn_mc = np.load(path_dnn_mc)[i_test, ...]
    miracle_mc = np.load(path_miracle_mc)[i_test, ...]

    # flip along T1 axis
    dnn_mc = np.flip(dnn_mc, axis=3)
    miracle_mc = np.flip(miracle_mc, axis=0)

    # Calculate accuracy for DNNs
    for m, model_type in enumerate(['sv', 'mb']):
        for k, snr_model in enumerate(snr_train_list):
            for j, distr in enumerate(distr_title_list):
                for t, par in enumerate(['t1', 't2']):
                    dnn_acc[m, k, j, :, :, t] = (
                        (dnn_mc[m, k, j, :, :, 0, t] - t1t2_mesh[:, :, t])/t1t2_mesh[:, :, t])*100

    # Calculate accuracy for MIRACLE
    for t, par in enumerate(['t1', 't2']):
        miracle_acc[:, :, t] = (
            (miracle_mc[:, :, 0, t] - t1t2_mesh[:, :, t])/t1t2_mesh[:, :, t])*100

    plot_fig2(
        snr_train_idx=i_train,
        distr_list=distr_title_list,
        t1_mesh=t1_mesh_flip,
        t2_mesh=t2_mesh_flip,
        mc_acc_dnn=dnn_acc,
        path_save=paths.figures / 'fig2.svg'
    )
