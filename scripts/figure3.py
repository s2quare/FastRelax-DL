"""Script for generating Figure 3.

- CoD of DNNs and MIRACLE for different SNR levels and distributions on in silico data.
- Two distributions to calculate the CoD: 
    - uniform
    - in vivo
- Use Mean of MC simulations

Author: florian.birk@tuebingen.mpg.de (Florian Birk), February 2024
"""
# %% Import
import paths
from fcns_fig_plot import plot_fig3
from fcns_performance import cod
# external modules
import torch
import numpy as np
import yaml

if __name__ == '__main__':
    # Set random seed
    torch.manual_seed(2308)

    # cfgs
    cfg = yaml.safe_load(
        open(paths.cfgs / '1.nn_train' / 'train_in_silico.yml'))
    path_in_silico = paths.data / 'in_silico' / 'test'
    path_mc = paths.data / 'in_silico' / 'mc'
    snr_test_list = ['inf', 50, 45, 40, 35, 30, 25, 20, 15, 10]
    snr_train_list = ['inf', 50, 25, 10]
    distr_title_list = ['in vivo', 'uniform', 'extended']
    mc_sample = 5000
    snr_start_idx = 1  # 0 = inf, 1 = 50, 2 = 45,...

    # Load ground truth for uniform grid
    # flip t1 and t2 mesh and mc results along T1
    t1_mesh = torch.load(path_in_silico /
                         't1_mesh_uniform_test.pt').float().numpy()
    t2_mesh = torch.load(path_in_silico /
                         't2_mesh_uniform_test.pt').float().numpy()
    t1_mesh_flip = np.flip(t1_mesh, axis=0)
    t2_mesh_flip = np.flip(t2_mesh, axis=0)
    t1t2_mesh = np.stack((t1_mesh_flip, t2_mesh_flip), axis=-1)
    t1t2_1d = np.reshape(t1t2_mesh, (t1t2_mesh.shape[0]*t1t2_mesh.shape[1], 2))

    # Load ground truth for in vivo distribution
    t1t2_in_vivo = torch.load(path_in_silico /
                              't1t2b0b1_in_vivo_test.pt').float().numpy()[:, :2]

    # Full CoD array for all fixed SNRs
    cod_dnn_grid = np.zeros(
        (len(snr_test_list), 2, len(snr_train_list), len(distr_title_list), 2))
    cod_dnn_invivo = np.zeros(
        (len(snr_test_list), 2, len(snr_train_list), len(distr_title_list), 2))
    cod_miracle_grid = np.zeros((len(snr_test_list), 2))
    cod_miracle_invivo = np.zeros((len(snr_test_list), 2))
    # Load data if available
    path_dnn_mc_grid = path_mc / \
        f'dnn_{mc_sample}-samples_grid.npy'
    path_dnn_mc_invivo = path_mc / \
        f'dnn_{mc_sample}-samples_invivo.npy'
    path_miracle_mc_grid = path_mc / \
        f'miracle_{mc_sample}-samples_grid.npy'
    path_miracle_mc_invivo = path_mc / \
        f'miracle_{mc_sample}-samples_invivo.npy'
    mc_dnn_grid = np.load(path_dnn_mc_grid)
    mc_dnn_invivo = np.load(path_dnn_mc_invivo)
    mc_miracle_grid = np.load(path_miracle_mc_grid)
    mc_miracle_invivo = np.load(path_miracle_mc_invivo)
    # iterate over all four SNR levels and distributions
    # flip along T1 axis for grid mc results
    mc_dnn_grid = np.flip(mc_dnn_grid, axis=4)
    mc_miracle_grid = np.flip(mc_miracle_grid, axis=1)
    # calculate the CoD from MC mean vs. gt on a certain test data SNR:
    # For all DNNs trained with different frameworks, SNR levels and distributions and MIRACLE
    for i, snr in enumerate(snr_test_list):
        # just take the mean of the MC simulations
        mc_dnn_grid_mean = mc_dnn_grid[:, :, :, :, :, :, 0, :]
        mc_dnn_invivo_mean = mc_dnn_invivo[:, :, :, :, :, 0, :]
        mc_miracle_grid_mean = mc_miracle_grid[:, :, :, 0, :]
        mc_miracle_invivo_mean = mc_miracle_invivo[:, :, 0, :]
        # get shapes
        s_dnn_grid = mc_dnn_grid_mean.shape
        s_miracle_grid = mc_miracle_grid_mean.shape

        # reshape uniform grid values to 1d
        mc_dnn_grid_mean_1d = np.reshape(
            mc_dnn_grid_mean, (s_dnn_grid[0], s_dnn_grid[1], s_dnn_grid[2], s_dnn_grid[3], s_dnn_grid[4]*s_dnn_grid[5], 2))
        mc_miracle_grid_mean_1d = np.reshape(
            mc_miracle_grid_mean, (s_miracle_grid[0], s_miracle_grid[1]*s_miracle_grid[2], 2))

        # CoD for MIRACLE
        cod_miracle_grid[i, :] = cod(
            t1t2_1d, mc_miracle_grid_mean_1d[i, :, :])
        cod_miracle_invivo[i, :] = cod(
            t1t2_in_vivo, mc_miracle_invivo_mean[i, :, :])
        # CoD for DNNs
        for m, model_type in enumerate(['sv', 'mb']):
            for k, snr_model in enumerate(snr_train_list):
                for j, distr in enumerate(distr_title_list):
                    cod_dnn_grid[i, m, k, j, :] = cod(
                        t1t2_1d, mc_dnn_grid_mean_1d[i, m, k, j, :, :])
                    cod_dnn_invivo[i, m, k, j, :] = cod(
                        t1t2_in_vivo, mc_dnn_invivo_mean[i, m, k, j, :, :])

    # convert snr_test_list elements to string
    snr_test_list = [str(snr) for snr in snr_test_list]

    plot_fig3(
        cod_dnn_grid=cod_dnn_grid[snr_start_idx:, ...],
        cod_dnn_invivo=cod_dnn_invivo[snr_start_idx:, ...],
        cod_miracle_grid=cod_miracle_grid[snr_start_idx:, ...],
        cod_miracle_invivo=cod_miracle_invivo[snr_start_idx:, ...],
        snr_test_list=snr_test_list[snr_start_idx:],
        snr_train_idx=0,  # 0 = inf, 1 = 50, 2 = 25, 3 = 10
        distr_list=distr_title_list,
        path_save=paths.figures / 'fig3.svg'
    )
