"""Functions used to create plots

Author: florian.birk@tuebingen.mpg.de, February 2024
"""
import matplotlib.pyplot as plt
import numpy as np
import wandb

########################################
# Plots for conditional mean predictions
########################################


def plot_mean_std_t1_t2(gt, par_mean, par_std, t1_range, t2_range):
    """Plot mean and std of T1 and T2 values. Compare to GT (black line)"""
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].plot(t1_range, par_mean[:, 0], c='b', label='pred_mean')
    axs[0].fill_between(t1_range, par_mean[:, 0]-par_std[:, 0],
                        par_mean[:, 0]+par_std[:, 1], color='b', alpha=0.2)
    axs[0].plot(gt[:, 0], gt[:, 0], c='k', label='gt')
    axs[0].set_title('T1 [ms]')
    axs[0].legend()
    axs[1].plot(t2_range, par_mean[:, 1], c='b', label='pred_mean')
    axs[1].fill_between(t2_range, par_mean[:, 1]-par_std[:, 1],
                        par_mean[:, 1]+par_std[:, 1], color='b', alpha=0.2)
    axs[1].plot(gt[:, 1], gt[:, 1], c='k', label='gt')
    axs[1].set_title('T2 [ms]')
    axs[1].legend()
    plt.suptitle('T1 and T2 mean and std')
    # push to wandb
    wandb.log({'t1_t2_mean_std': wandb.Image(fig)})


def plot_mean_std_t1_t2_b0(gt, par_mean, par_std, t1_range, t2_range, b0_range):
    """Plot mean and std of T1, T2 and B0 values. Compare to GT (black line)"""
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].plot(t1_range, par_mean[:, 0], c='b', label='pred_mean')
    axs[0].fill_between(t1_range, par_mean[:, 0]-par_std[:, 0],
                        par_mean[:, 0]+par_std[:, 1], color='b', alpha=0.2)
    axs[0].plot(gt[:, 0], gt[:, 0], c='k', label='gt')
    axs[0].set_title('T1 [ms]')
    axs[0].legend()
    axs[1].plot(t2_range, par_mean[:, 1], c='b', label='pred_mean')
    axs[1].fill_between(t2_range, par_mean[:, 1]-par_std[:, 1],
                        par_mean[:, 1]+par_std[:, 1], color='b', alpha=0.2)
    axs[1].plot(gt[:, 1], gt[:, 1], c='k', label='gt')
    axs[1].set_title('T2 [ms]')
    axs[1].legend()
    axs[2].plot(b0_range, par_mean[:, 2], c='b', label='pred_mean')
    axs[2].fill_between(b0_range, par_mean[:, 2]-par_std[:, 2],
                        par_mean[:, 2]+par_std[:, 2], color='b', alpha=0.2)
    axs[2].plot(gt[:, 2], gt[:, 2], c='k', label='gt')
    axs[2].set_title('B0 [Hz]')
    axs[2].legend()
    plt.suptitle('T1, T2 and B0 mean and std')
    # push to wandb
    wandb.log({'t1_t2_b0_mean_std': wandb.Image(fig)})


def plot_scatter_t1_t2_mean(gt, par_mean):
    """Plot the point predictions for the mean vs. gt"""
    fig, axs = plt.subplots(1, 2)
    axs[0].scatter(gt[:, 0], par_mean[:, 0], s=.6, alpha=.5)
    axs[1].scatter(gt[:, 1], par_mean[:, 1], s=.6, alpha=.5)
    axs[0].set_title('T1 [ms]')
    axs[1].set_title('T2 [ms]')
    plt.suptitle('GT vs. Mean')
    # push to wandb
    wandb.log({'scatter_t1_t2_mean': wandb.Image(fig)})


def plot_scatter_t1_t2_b0_mean(gt, par_mean):
    """Plot the point predictions for the mean vs. gt"""
    fig, axs = plt.subplots(1, 3)
    axs[0].scatter(gt[:, 0], par_mean[:, 0], s=.6, alpha=.5)
    axs[1].scatter(gt[:, 1], par_mean[:, 1], s=.6, alpha=.5)
    axs[2].scatter(gt[:, 2], par_mean[:, 2], s=.6, alpha=.5)
    axs[0].set_title('T1 [ms]')
    axs[1].set_title('T2 [ms]')
    axs[2].set_title('B0 [Hz]')
    plt.suptitle('GT vs. Mean')
    # push to wandb
    wandb.log({'scatter_t1_t2_b0_mean': wandb.Image(fig)})
