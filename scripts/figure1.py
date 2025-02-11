"""Script for generating the distributions of Figure 1.

- Uniform, uniform extended and in vivo distributions

Author: florian.birk@tuebingen.mpg.de (Florian Birk), February 2024
"""
# %% Import
import paths
from model import Qbssfp
from fcns_mb import calc_Fn, norm2_vox
from fcns_fig_plot import plot_fig1
import torch
import numpy as np
import yaml


def prepare_in_silico(path_nn_in_silico, distr):
    par = torch.load(path_nn_in_silico /
                     f't1t2b0b1_nob0_{distr}_12pc.pt').float()
    return par


if __name__ == '__main__':
    paths.figures.mkdir(parents=True, exist_ok=True)
    distr = ['uniform', 'uniform_ext', 'in_vivo']

    # get the different sig_mean and sig_std for each distribution
    path_nn_in_silico = paths.data / 'in_silico' / 'train'
    par_uniform = prepare_in_silico(path_nn_in_silico, distr[0])
    par_uniform_ext = prepare_in_silico(path_nn_in_silico, distr[1])
    par_in_vivo = prepare_in_silico(path_nn_in_silico, distr[2])

    # %% Plot figure
    plot_fig1(
        par_gt_list=[par_uniform.numpy(), par_uniform_ext.numpy(),
                     par_in_vivo.numpy()],
        distr_list=distr,
        v_list=[0, 15],
        path_save=paths.figures)
