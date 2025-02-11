"""Functions used to create the manuscript figures.

Author: florian.birk@tuebingen.mpg.de (Florian Birk), February 2024
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
from matplotlib.gridspec import GridSpec
import seaborn as sns  # needed to use the colormap 'mako'
import numpy as np
from tueplots import figsizes, fonts, fontsizes, axes


def plot_fig1(par_gt_list, distr_list, v_list, path_save):
    """Plot the distributions used for data simulation"""
    plt.rcParams.update(figsizes.neurips2023(
        nrows=1, ncols=1, height_to_width_ratio=1))
    plt.rcParams.update(fonts.neurips2023())
    for i in range(3):
        par_gt_temp = par_gt_list[i]
        distr_temp = distr_list[i]
        fig, axs = plt.subplots(1, 1)
        axs.hist2d(x=par_gt_temp[:, 1], y=par_gt_temp[:, 0], bins=200, range=[[par_gt_temp[:, 1].min(), par_gt_temp[:, 1].max()], [par_gt_temp[:, 0].min(), par_gt_temp[:, 0].max()]],
                   cmap='mako', vmin=v_list[0], vmax=v_list[1])
        axs.axis('off')
        plt.savefig(path_save / f'fig1_{distr_temp}.png', dpi=300)
        plt.close()


def plot_fig2(snr_train_idx, distr_list, t1_mesh, t2_mesh, mc_acc_dnn, path_save):
    """Plot relative error in percentage of Monte Carlo sampling for 2D in silico grid T1/T2 estimation. Compare DNNS trained with three different distributions and two different frameworks with each other.
    Test data with SNR = 25.
    Train data with SNR = inf and SNR = 25."""
    # matplotlib.use('PS')
    plt.rcParams.update(figsizes.neurips2023(
        nrows=4, ncols=6, height_to_width_ratio=1))
    plt.rcParams.update(fonts.neurips2023())
    plt.rcParams.update({'xtick.labelsize': 14})
    plt.rcParams.update({'ytick.labelsize': 14})
    plt.rcParams.update({'font.size': 14})
    plt.rcParams['ytick.right'] = plt.rcParams['ytick.labelright'] = True
    plt.rcParams['ytick.left'] = plt.rcParams['ytick.labelleft'] = False

    t1_vec = t1_mesh[:, 0]
    t2_vec = t2_mesh[0, :]
    vmin = -15
    vmax = 15
    # Plot DNNs
    fig, axs = plt.subplots(4, 6, sharex=True, sharey=True, figsize=(12, 8))
    for i in range(3):
        # plot SVNN accuracy
        if i == 0:
            axs_svnn_iv_t1 = axs[0, i].imshow(mc_acc_dnn[0, 0, i, :, :, 0],
                                              cmap='bwr', vmin=vmin, vmax=vmax)  # [SVNN, Inf, Distr, grid, T1]
        else:
            axs[0, i].imshow(mc_acc_dnn[0, 0, i, :, :, 0],
                             cmap='bwr', vmin=vmin, vmax=vmax)  # [SVNN, Inf, Distr, grid, T1]
        axs[1, i].imshow(mc_acc_dnn[0, 0, i, :, :, 1],
                         cmap='bwr', vmin=vmin, vmax=vmax)  # [SVNN, Inf, Distr, grid, T2]
        axs[2, i].imshow(mc_acc_dnn[0, snr_train_idx, i, :, :, 0],
                         cmap='bwr', vmin=vmin, vmax=vmax)  # [SVNN, snr_train, Distr, grid, T1]
        axs[3, i].imshow(mc_acc_dnn[0, snr_train_idx, i, :, :, 1],
                         cmap='bwr', vmin=vmin, vmax=vmax)  # [SVNN, snr_train, Distr, grid, T2]
        # plot PINN accuracy
        axs[0, i+3].imshow(mc_acc_dnn[1, 0, i, :, :, 0],
                           cmap='bwr', vmin=vmin, vmax=vmax)  # [PINN, Inf, Distr, grid, T1]
        axs[1, i+3].imshow(mc_acc_dnn[1, 0, i, :, :, 1],
                           cmap='bwr', vmin=vmin, vmax=vmax)  # [PINN, Inf, Distr, grid, T2]
        axs[2, i+3].imshow(mc_acc_dnn[1, snr_train_idx, i, :, :, 0],
                           cmap='bwr', vmin=vmin, vmax=vmax)  # [PINN, snr_train, Distr, grid, T1]
        axs[3, i+3].imshow(mc_acc_dnn[1, snr_train_idx, i, :, :, 1],
                           cmap='bwr', vmin=vmin, vmax=vmax)  # [PINN, snr_train, Distr, grid, T2]
    axs[0, 0].set_title(f"{distr_list[0]}")
    axs[0, 1].set_title(r"$\bf{SVNN}$"+f"\n{distr_list[1]}")
    axs[0, 2].set_title(f"{distr_list[2]}")
    axs[0, 3].set_title(f"{distr_list[0]}")
    axs[0, 4].set_title(r"$\bf{PINN}$"+f"\n{distr_list[1]}")
    axs[0, 5].set_title(f"{distr_list[2]}")
    # add yaxis label in the first column
    axs[0, 0].set_ylabel(r"$\epsilon_{rel} (T_1)$")
    axs[1, 0].set_ylabel(r"$\epsilon_{rel} (T_2)$")
    axs[2, 0].set_ylabel(r"$\epsilon_{rel} (T_1)$")
    axs[3, 0].set_ylabel(r"$\epsilon_{rel} (T_2)$")
    ax_cbar = fig.add_axes([1.01, 0.28, 0.015, 0.4])
    cbar = fig.colorbar(axs_svnn_iv_t1, cax=ax_cbar)
    cbar.set_label(r"$\epsilon_{rel}$" + " [%]")
    for ax in axs.flat:
        ax.set_xticks(np.linspace(0, t2_vec.shape[0], 3).astype(int))
        ax.set_yticks(np.linspace(0, t1_vec.shape[0], 3).astype(int))
        ax.set_xticklabels(
            np.linspace(t2_vec.min(), t2_vec.max(), 3).astype(int))
        ax.set_yticklabels(
            np.linspace(t1_vec.max(), t1_vec.min(), 3).astype(int))
    plt.savefig(path_save, format='svg')


def plot_fig3(cod_dnn_grid, cod_dnn_invivo, cod_miracle_grid, cod_miracle_invivo, snr_test_list, snr_train_idx, distr_list, path_save):
    """Plot the CoD for all DNNs trained on certain SNR level (snr_train_idx). Test Uniform and in vivo distribution. 
    Plot the CoD for MIRACLE and the absolute difference for each distribution and dnn to MIRACEL on the second yaxis.
    """
    plt.rcParams.update(figsizes.neurips2023(
        nrows=3, ncols=2, height_to_width_ratio=0.9))
    plt.rcParams.update(fonts.neurips2023())
    plt.rcParams.update({'xtick.labelsize': 10})
    plt.rcParams.update({'ytick.labelsize': 10})
    plt.rcParams.update({'font.size': 12})

    relax_color = ['r', 'b']
    miracle_marker = 's'
    distr_marker = ['^', 'o', '+']
    marker_sz = 8
    marker_sz_miracle = 6
    line_style_grid = ''
    line_style_invivo = ''
    line_style_miracle = 'dashed'
    line_width = 1
    method_list = ['SVNN', 'PINN']

    fig, axs = plt.subplots(3, 2, sharex=True)
    # PINN
    for d, distr in enumerate(distr_list):
        cod_dnn_grid_diff = cod_dnn_grid[:, 1,
                                         snr_train_idx, d, :] - cod_miracle_grid
        cod_dnn_invivo_diff = cod_dnn_invivo[:, 1,
                                             snr_train_idx, d, :] - cod_miracle_invivo
        # Uniform grid
        axs[1, 0].plot(snr_test_list, cod_dnn_grid_diff[:, 0], color=relax_color[0], marker=distr_marker[d], markersize=marker_sz, fillstyle='none',
                       linestyle=line_style_grid, linewidth=line_width, label=rf'$T_1$ {distr}')
        axs[1, 0].plot(snr_test_list, cod_dnn_grid_diff[:, 1], color=relax_color[1], marker=distr_marker[d], markersize=marker_sz, fillstyle='none',
                       linestyle=line_style_grid, linewidth=line_width, label=rf'$T_2$ {distr}')
        # In vivo
        axs[1, 1].plot(snr_test_list, cod_dnn_invivo_diff[:, 0], color=relax_color[0], marker=distr_marker[d], markersize=marker_sz, fillstyle='none',
                       linestyle=line_style_invivo, linewidth=line_width, label=rf'$T_1$ {distr}')
        axs[1, 1].plot(snr_test_list, cod_dnn_invivo_diff[:, 1], color=relax_color[1], marker=distr_marker[d], markersize=marker_sz, fillstyle='none',
                       linestyle=line_style_invivo, linewidth=line_width, label=rf'$T_2$ {distr}')
    # SVNN
    for d, distr in enumerate(distr_list):
        cod_dnn_grid_diff = cod_dnn_grid[:, 0,
                                         snr_train_idx, d, :] - cod_miracle_grid
        cod_dnn_invivo_diff = cod_dnn_invivo[:, 0,
                                             snr_train_idx, d, :] - cod_miracle_invivo
        # Uniform grid
        axs[2, 0].plot(snr_test_list, cod_dnn_grid_diff[:, 0], color=relax_color[0], marker=distr_marker[d], markersize=marker_sz, fillstyle='none',
                       linestyle=line_style_grid, linewidth=line_width, label=rf'$T_1$ {distr}')
        axs[2, 0].plot(snr_test_list, cod_dnn_grid_diff[:, 1], color=relax_color[1], marker=distr_marker[d], markersize=marker_sz, fillstyle='none',
                       linestyle=line_style_grid, linewidth=line_width, label=rf'$T_2$ {distr}')
        # In vivo
        axs[2, 1].plot(snr_test_list, cod_dnn_invivo_diff[:, 0], color=relax_color[0], marker=distr_marker[d], markersize=marker_sz, fillstyle='none',
                       linestyle=line_style_invivo, linewidth=line_width, label=rf'$T_1$ {distr}')
        axs[2, 1].plot(snr_test_list, cod_dnn_invivo_diff[:, 1], color=relax_color[1], marker=distr_marker[d], markersize=marker_sz, fillstyle='none',
                       linestyle=line_style_invivo, linewidth=line_width, label=rf'$T_2$ {distr}')
    axs[1, 0].set_ylabel(r'$\Delta$CoD '+'PINN')
    axs[2, 0].set_ylabel(r'$\Delta$CoD '+'SVNN')
    axs[0, 0].set_title('Uniform Grid', fontweight='bold')
    axs[0, 1].set_title('In Vivo', fontweight='bold')
    axs[2, 0].set_xlabel('Test SNR')
    axs[2, 1].set_xlabel('Test SNR')
    # set ylim for each row
    axs[0, 0].set_ylim(0.63, 1.02)
    axs[0, 1].set_ylim(0.63, 1.02)
    axs[1, 0].set_ylim(-0.02, 0.19)
    axs[1, 1].set_ylim(-0.02, 0.19)
    axs[2, 0].set_ylim(-0.19, 0.19)
    axs[2, 1].set_ylim(-0.19, 0.19)

    # Plot MIRACLE reference line
    axs[0, 0].set_ylabel('CoD MIRACLE')
    # Uniform grid T1
    axs[0, 0].plot(snr_test_list, cod_miracle_grid[:, 0], color=relax_color[0], marker=miracle_marker, markersize=marker_sz_miracle, fillstyle='none',
                   linestyle=line_style_miracle, linewidth=line_width, label=r'$T_1$')
    # in vivo T1
    axs[0, 1].plot(snr_test_list, cod_miracle_invivo[:, 0], color=relax_color[0], marker=miracle_marker, markersize=marker_sz_miracle, fillstyle='none',
                   linestyle=line_style_miracle, linewidth=line_width, label=r'$T_1$')
    # Uniform grid T2
    axs[0, 0].plot(snr_test_list, cod_miracle_grid[:, 1], color=relax_color[1], marker=miracle_marker, markersize=marker_sz_miracle, fillstyle='none',
                   linestyle=line_style_miracle, linewidth=line_width, label=r'$T_2$')
    # in vivo T2
    axs[0, 1].plot(snr_test_list, cod_miracle_invivo[:, 1], color=relax_color[1], marker=miracle_marker, markersize=marker_sz_miracle, fillstyle='none',
                   linestyle=line_style_miracle, linewidth=line_width, label=r'$T_2$')
    # plot legend outside of the plot
    axs[0, 0].legend(loc='center', fontsize=8)
    axs[1, 0].legend(loc='center', fontsize=8)
    plt.figtext(0, 0.97, 'a', ha='center',
                va='center', fontweight='bold', fontsize=22)
    plt.figtext(0, 0.65, 'b', ha='center',
                va='center', fontweight='bold', fontsize=22)
    plt.figtext(0, 0.35, 'c', ha='center',
                va='center', fontweight='bold', fontsize=22)
    plt.savefig(path_save, dpi=300)


def plot_fig4(par_pred_dnn, par_pred_miracle, par_diff_dnn, distr_list, slice, cmap_list, vmin_t1t2, vmax_t1t2, vmax_diff, path_save):
    """Plot an exemplary axial slice for T1/T2 parametere estimations and difference map to MIRACLE for one subject"""
    plt.rcParams.update(fonts.neurips2023())
    plt.rcParams.update({'xtick.labelsize': 6})
    plt.rcParams.update({'ytick.labelsize': 6})
    plt.rcParams.update({'font.size': 6})

    fig, axs = plt.subplots(4, 7, layout='constrained')
    # MIRACLE plot
    # T1
    axs[0, 0].imshow(par_pred_miracle[:, :, slice, 0],
                     cmap=cmap_list[0], vmin=vmin_t1t2[0], vmax=vmax_t1t2[0])
    # T2
    axs[2, 0].imshow(par_pred_miracle[:, :, slice, 1],
                     cmap=cmap_list[1], vmin=vmin_t1t2[1], vmax=vmax_t1t2[1])
    # DNNs plot
    for m in range(2):
        for d in range(3):
            if m == 0 and d == 0:
                # T1
                # Parameter prediction
                ax_t1_pred = axs[0, m*3+d+1].imshow(par_pred_dnn[m, d, :, :, slice, 0],
                                                    cmap=cmap_list[0], vmin=vmin_t1t2[0], vmax=vmax_t1t2[0])
                # Diff to MIRACLE
                ax_t1_diff = axs[1, m*3+d+1].imshow(par_diff_dnn[m, d, :, :, slice, 0],
                                                    cmap=cmap_list[-1], vmin=-vmax_diff[0], vmax=vmax_diff[0])
                # T2
                # Parameter prediction
                ax_t2_pred = axs[2, m*3+d+1].imshow(par_pred_dnn[m, d, :, :, slice, 1],
                                                    cmap=cmap_list[1], vmin=vmin_t1t2[1], vmax=vmax_t1t2[1])
                # Diff to MIRACLE
                ax_t2_diff = axs[3, m*3+d+1].imshow(par_diff_dnn[m, d, :, :, slice, 1],
                                                    cmap=cmap_list[-1], vmin=-vmax_diff[1], vmax=vmax_diff[1])
            else:
                # T1
                axs[0, m*3+d+1].imshow(par_pred_dnn[m, d, :, :, slice, 0],
                                       cmap=cmap_list[0], vmin=vmin_t1t2[0], vmax=vmax_t1t2[0])
                axs[1, m*3+d+1].imshow(par_diff_dnn[m, d, :, :, slice, 0],
                                       cmap=cmap_list[-1], vmin=-vmax_diff[0], vmax=vmax_diff[0])
                # T2
                axs[2, m*3+d+1].imshow(par_pred_dnn[m, d, :, :, slice, 1],
                                       cmap=cmap_list[1], vmin=vmin_t1t2[1], vmax=vmax_t1t2[1])
                axs[3, m*3+d+1].imshow(par_diff_dnn[m, d, :, :, slice, 1],
                                       cmap=cmap_list[-1], vmin=-vmax_diff[1], vmax=vmax_diff[1])
            if d == 1:
                if m == 0:
                    axs[0, m*3 +
                        d+1].set_title(r'$\bf{SVNN}$'+f'\n{distr_list[d]}')
                elif m == 1:
                    axs[0, m*3 +
                        d+1].set_title(r'$\bf{PINN}$'+f'\n{distr_list[d]}')
            else:
                axs[0, m*3+d+1].set_title(f'{distr_list[d]}')
    # set titles
    axs[0, 0].set_title(r'$\bf{MIRACLE}$')
    # set ylabels (bold font)
    axs[0, 0].set_ylabel(r'$\bf{T_1}$' + ' [ms]', weight='bold')
    axs[1, 1].set_ylabel(r'$\bf{\Delta T_1}$' + ' [ms]', weight='bold')
    axs[2, 0].set_ylabel(r'$\bf{T_2}$' + ' [ms]', weight='bold')
    axs[3, 1].set_ylabel(r'$\bf{\Delta T_2}$' + ' [ms]', weight='bold')
    # remove all x and y ticks and labels
    for ax in axs.flat:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')
    fig.colorbar(ax_t1_pred, ax=axs[0, 6])
    fig.colorbar(ax_t1_diff, ax=axs[1, 6])
    fig.colorbar(ax_t2_pred, ax=axs[2, 6])
    fig.colorbar(ax_t2_diff, ax=axs[3, 6])
    # remove the entire axs[1, 0] and axs[3, 0] axis
    axs[1, 0].axis('off')
    axs[3, 0].axis('off')
    plt.savefig(path_save, dpi=300)


def plot_fig5(pred, pred_nob0, pred_miracle, npc_trained, shift, path_save):
    """Plot the relative error in percentage of T1/T2 parameter estimations for different phase cycles and models. Compare the predictions of DNNs trained with and without B0 fitting and MIRACLE for a different number of pc.
    """
    # param_values = ['[ms]', '[ms]', '[rad]']
    t1 = 939.0
    t2 = 62.0
    ref_relax_values = [t1, t2]
    color_list_svnn = ['b', 'g', 'r']
    color_list_pinn = ['c', 'm', 'y']
    alphas = [1, 0.6, 0.4]

    pred_rel_t1 = (pred[:, :, :, 0] - t1)/t1*100
    pred_rel_t2 = (pred[:, :, :, 1] - t2)/t2*100
    pred_rel = np.stack((pred_rel_t1, pred_rel_t2), axis=-1)
    pred_nob0_rel_t1 = (pred_nob0[:, :, :, 0] - t1)/t1*100
    pred_nob0_rel_t2 = (pred_nob0[:, :, :, 1] - t2)/t2*100
    pred_nob0_rel = np.stack((pred_nob0_rel_t1, pred_nob0_rel_t2), axis=-1)
    pred_miracle_rel_t1 = (pred_miracle[:, :, 0] - t1)/t1*100
    pred_miracle_rel_t2 = (pred_miracle[:, :, 1] - t2)/t2*100
    pred_miracle_rel = np.stack(
        (pred_miracle_rel_t1, pred_miracle_rel_t2), axis=-1)

    for i, npc in enumerate(npc_trained):
        print(
            f"Maximum relative error of T1 for {npc} phase cycles: {np.max(pred_rel_t1[:, i, :])}%")
        print(
            f"Maximum relative error of T2 for {npc} phase cycles: {np.max(pred_rel_t2[:, i, :])}%")

    # plot
    plt.rcParams.update({'xtick.labelsize': 14})
    plt.rcParams.update({'ytick.labelsize': 14})
    plt.rcParams.update({'font.size': 18})
    plt.rcParams.update(figsizes.neurips2023(
        nrows=3, ncols=1))
    plt.rcParams.update(fonts.neurips2023())
    fig, axs = plt.subplots(2, 2, figsize=(12, 12), sharex=True)

    for k, param in enumerate(['T1', 'T2']):
        for i, npc in enumerate(npc_trained):
            for j, model in enumerate(['SVNN', 'PINN']):
                if model == 'SVNN':
                    color_list = color_list_svnn
                else:
                    color_list = color_list_pinn
                # plot the predictions when trained without B0 fitting
                axs[k, 0].plot(shift, pred_nob0_rel[:, i, j, k],
                               label=rf'${npc}pc$'+' '+fr'${model}$', color=color_list[i], alpha=alphas[i])
                # plot the predictions when trained with B0 fitting
                axs[k, 1].plot(shift, pred_rel[:, i, j, k],
                               label=rf'${npc}pc$'+' ' + fr'${model}_{{complex}}$', color=color_list[i], alpha=alphas[i])
            # plot the MIRACLE predictions (left column)
            axs[k, 0].plot(shift, pred_miracle_rel[:, i, k],
                           label=rf'${npc}pc$'+' ' + r'$MIRACLE$', color='k', alpha=alphas[i])
    axs[1, 0].axhline(y=0, color='k', linestyle='--',
                      label=r'$Ref$'+' '+r'$T_1$' + ' ' + ': ' + rf'${int(ref_relax_values[0])}$' + r'$ms$')
    axs[1, 1].axhline(y=0, color='k', linestyle='--',
                      label=r'$Ref$'+' '+r'$T_1$' + ' ' + ': ' + rf'${int(ref_relax_values[0])}$' + r'$ms$')
    for m, param in enumerate(['$T_1$', '$T_2$']):
        axs[m, 0].axhline(y=0, color='k',
                          linestyle='--', label=r'$Ref$'+' '+rf'{param}' + ' ' + ': ' + rf'${int(ref_relax_values[m])}$' + r'$ms$')
        axs[m, 1].axhline(y=0, color='k',
                          linestyle='--', label=r'$Ref$'+' '+rf'{param}' + ' ' + ': ' + rf'${int(ref_relax_values[m])}$' + r'$ms$')
    for ax in axs[1, :]:
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, loc='upper center',
                  bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=16)
    axs[1, 0].set_xlabel(r'$\mathbf{\theta}$' + ' ' + r'$\mathbf{[rad]}$')
    axs[1, 1].set_xlabel(r'$\mathbf{\theta}$' + ' ' + r'$\mathbf{[rad]}$')
    axs[0, 0].set_ylabel(
        r'$\mathbf{\epsilon_{rel}(T_1)}$' + ' ' + r'$\bf{[\%]}$')
    axs[1, 0].set_ylabel(
        r'$\mathbf{\epsilon_{rel}(T_2)}$' + ' ' + r'$\bf{[\%]}$')
    axs[0, 0].set_title(
        r'$\mathbf{DNN}$' + ' ' + r'$(\mathbf{standard}$' + ' ' + r'$\mathbf{magnitude}$'+'-'+r'$\mathbf{based})$' + ' | ' + r'$\mathbf{MIRACLE}$', fontsize=14)
    axs[0, 1].set_title(r'$\mathbf{DNN}$' + ' ' +
                        r'$(\mathbf{complex}$'+'-'+r'$\mathbf{based})$', fontsize=14)
    rel_min = -50
    rel_max = 60
    axs[0, 0].set_ylim([rel_min, rel_max])
    axs[0, 1].set_ylim([rel_min, rel_max])
    axs[1, 0].set_ylim([rel_min, rel_max])
    axs[1, 1].set_ylim([rel_min, rel_max])
    plt.savefig(path_save, dpi=300)


def plot_fig6(pred_miracle, pred_svnn, pred_svnn_complex, pred_pinn, pred_pinn_complex, diff_miracle, diff_svnn, diff_svnn_complex, diff_pinn, diff_pinn_complex, npc_diff, path_save):
    """Plot the T1 predictions for all frameworks based on 12pc data and the predictions + difference maps for 6 and 4pc data.
    """
    plt.rcParams.update(fonts.neurips2023())
    plt.rcParams.update({'xtick.labelsize': 14})
    plt.rcParams.update({'ytick.labelsize': 14})
    plt.rcParams.update({'font.size': 14})
    fig, axs = plt.subplots(5, 5, layout='constrained', figsize=(10, 10))
    slice = 100
    cmap_list = ['inferno', 'viridis', 'seismic']
    vmin_t1t2 = [360, 20]
    vmax_t1t2 = [1500, 120]
    vmax_diff = [500, 50]

    # plot the predictions (T1)
    npc_trained = [12, 6, 4]
    for i, npc in enumerate(npc_trained):
        img_t1_miracle = axs[0, i].imshow(np.flip(pred_miracle[..., 0, i][:, :, slice].T, axis=0),
                                          cmap=cmap_list[0], vmin=vmin_t1t2[0], vmax=vmax_t1t2[0])
        axs[1, i].imshow(np.flip(pred_svnn[..., 0, i][:, :, slice].T, axis=0),
                         cmap=cmap_list[0], vmin=vmin_t1t2[0], vmax=vmax_t1t2[0])
        axs[2, i].imshow(np.flip(pred_svnn_complex[..., 0, i][:, :, slice].T, axis=0),
                         cmap=cmap_list[0], vmin=vmin_t1t2[0], vmax=vmax_t1t2[0])
        axs[3, i].imshow(np.flip(pred_pinn[..., 0, i][:, :, slice].T, axis=0),
                         cmap=cmap_list[0], vmin=vmin_t1t2[0], vmax=vmax_t1t2[0])
        axs[4, i].imshow(np.flip(pred_pinn_complex[..., 0, i][:, :, slice].T, axis=0),
                         cmap=cmap_list[0], vmin=vmin_t1t2[0], vmax=vmax_t1t2[0])
        if i == 1:
            axs[0, i].set_title(r'$\bf{T_1}$' +
                                r' $\bf{[ms]}$' + '\n' + fr'$\bf{{{npc} pc}}$')
        else:
            axs[0, i].set_title(fr'$\bf{{{npc} pc}}$')

    # plot the difference maps (T1)
    for i, npc in enumerate(npc_diff):
        img_diff_miracle = axs[0, i+3].imshow(np.flip(diff_miracle[..., 0, i][:, :, slice].T, axis=0),
                                              cmap=cmap_list[2], vmin=-vmax_diff[0], vmax=vmax_diff[0])
        axs[1, i+3].imshow(np.flip(diff_svnn[..., 0, i][:, :, slice].T, axis=0),
                           cmap=cmap_list[2], vmin=-vmax_diff[0], vmax=vmax_diff[0])
        axs[2, i+3].imshow(np.flip(diff_svnn_complex[..., 0, i][:, :, slice].T, axis=0),
                           cmap=cmap_list[2], vmin=-vmax_diff[0], vmax=vmax_diff[0])
        axs[3, i+3].imshow(np.flip(diff_pinn[..., 0, i][:, :, slice].T, axis=0),
                           cmap=cmap_list[2], vmin=-vmax_diff[0], vmax=vmax_diff[0])
        axs[4, i+3].imshow(np.flip(diff_pinn_complex[..., 0, i][:, :, slice].T, axis=0),
                           cmap=cmap_list[2], vmin=-vmax_diff[0], vmax=vmax_diff[0])
        axs[0, i+3].set_title(r'$\bf{\Delta T_1}$' +
                              r' $\bf{[ms]}$' + '\n' + fr'$\bf{{{npc} pc}}$')

    axs[0, 0].set_ylabel(r'$\mathbf{MIRACLE}$', weight='bold')
    axs[1, 0].set_ylabel(r'$\mathbf{SVNN}$', weight='bold')
    axs[2, 0].set_ylabel(r'$\mathbf{SVNN}$', weight='bold')
    axs[3, 0].set_ylabel(r'$\mathbf{PINN}$', weight='bold')
    axs[4, 0].set_ylabel(r'$\mathbf{PINN}$', weight='bold')
    for ax in axs.flat:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    # [left, bottom, width, height]
    cbar_t1 = fig.add_axes([0.1, 0.03, 0.3, 0.02])
    fig.colorbar(img_t1_miracle, cax=cbar_t1, orientation='horizontal')
    cbar_diff = fig.add_axes([0.6, 0.03, 0.3, 0.02])
    fig.colorbar(img_diff_miracle, cax=cbar_diff, orientation='horizontal')
    # figure name of revised main manuscript
    plt.savefig(path_save, dpi=300)


def plot_fig7(pred_miracle, pred_svnn, pred_svnn_complex, pred_pinn, pred_pinn_complex, diff_miracle, diff_svnn, diff_svnn_complex, diff_pinn, diff_pinn_complex, npc_diff, path_save):
    """Plot the T2 predictions for all frameworks based on 12pc data and the predictions + difference maps for 6 and 4pc data."""
    plt.rcParams.update(fonts.neurips2023())
    plt.rcParams.update({'xtick.labelsize': 14})
    plt.rcParams.update({'ytick.labelsize': 14})
    plt.rcParams.update({'font.size': 14})
    fig, axs = plt.subplots(5, 5, layout='constrained', figsize=(10, 10))
    slice = 100
    cmap_list = ['inferno', 'viridis', 'seismic']
    vmin_t1t2 = [360, 20]
    vmax_t1t2 = [1500, 120]
    vmax_diff = [500, 50]

    # plot the predictions (T2)
    npc_trained = [12, 6, 4]
    for i, npc in enumerate(npc_trained):
        img_t2_miracle = axs[0, i].imshow(np.flip(pred_miracle[..., 1, i][:, :, slice].T, axis=0),
                                          cmap=cmap_list[1], vmin=vmin_t1t2[1], vmax=vmax_t1t2[1])
        axs[1, i].imshow(np.flip(pred_svnn[..., 1, i][:, :, slice].T, axis=0),
                         cmap=cmap_list[1], vmin=vmin_t1t2[1], vmax=vmax_t1t2[1])
        axs[2, i].imshow(np.flip(pred_svnn_complex[..., 1, i][:, :, slice].T, axis=0),
                         cmap=cmap_list[1], vmin=vmin_t1t2[1], vmax=vmax_t1t2[1])
        axs[3, i].imshow(np.flip(pred_pinn[..., 1, i][:, :, slice].T, axis=0),
                         cmap=cmap_list[1], vmin=vmin_t1t2[1], vmax=vmax_t1t2[1])
        axs[4, i].imshow(np.flip(pred_pinn_complex[..., 1, i][:, :, slice].T, axis=0),
                         cmap=cmap_list[1], vmin=vmin_t1t2[1], vmax=vmax_t1t2[1])
        if i == 1:
            axs[0, i].set_title(r'$\bf{T_2}$' +
                                r' $\bf{[ms]}$' + '\n' + fr'$\bf{{{npc} pc}}$')
        else:
            axs[0, i].set_title(fr'$\bf{{{npc} pc}}$')

    # plot the difference maps (T2)
    for i, npc in enumerate(npc_diff):
        img_diff_miracle = axs[0, i+3].imshow(np.flip(diff_miracle[..., 1, i][:, :, slice].T, axis=0),
                                              cmap=cmap_list[2], vmin=-vmax_diff[1], vmax=vmax_diff[1])
        axs[1, i+3].imshow(np.flip(diff_svnn[..., 1, i][:, :, slice].T, axis=0),
                           cmap=cmap_list[2], vmin=-vmax_diff[1], vmax=vmax_diff[1])
        axs[2, i+3].imshow(np.flip(diff_svnn_complex[..., 1, i][:, :, slice].T, axis=0),
                           cmap=cmap_list[2], vmin=-vmax_diff[1], vmax=vmax_diff[1])
        axs[3, i+3].imshow(np.flip(diff_pinn[..., 1, i][:, :, slice].T, axis=0),
                           cmap=cmap_list[2], vmin=-vmax_diff[1], vmax=vmax_diff[1])
        axs[4, i+3].imshow(np.flip(diff_pinn_complex[..., 1, i][:, :, slice].T, axis=0),
                           cmap=cmap_list[2], vmin=-vmax_diff[1], vmax=vmax_diff[1])
        axs[0, i+3].set_title(r'$\bf{\Delta T_2}$' +
                              r' $\bf{[ms]}$' + '\n' + fr'$\bf{{{npc} pc}}$')

    axs[0, 0].set_ylabel(r'$\mathbf{MIRACLE}$')
    axs[1, 0].set_ylabel(r'$\mathbf{SVNN}$')
    axs[2, 0].set_ylabel(r'$\mathbf{SVNN}$')
    axs[3, 0].set_ylabel(r'$\mathbf{PINN}$')
    axs[4, 0].set_ylabel(r'$\mathbf{PINN}$')
    for ax in axs.flat:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    # [left, bottom, width, height]
    cbar_t2 = fig.add_axes([0.1, 0.03, 0.3, 0.02])
    fig.colorbar(img_t2_miracle, cax=cbar_t2, orientation='horizontal')
    cbar_diff = fig.add_axes([0.6, 0.03, 0.3, 0.02])
    fig.colorbar(img_diff_miracle, cax=cbar_diff, orientation='horizontal')
    # figure name of revised main manuscript
    plt.savefig(path_save, dpi=300)


def plot_fig8(cods, loss_val, nepochs, par_sv_single, par_mb_single, par_sv_final, par_mb_final, vmin, vmax, cmaps, min_cod, path_save):
    """Plot the CoD of epoch X vs. epoch final for SVNN and PINN on in vivo data. Both frameworks were trained on uniform distr with snr=inf.
    Accumulated time for each epoch is plotted as well (second y-axis).
    Plot axial, coronal and sagittal slice for dnn (single and final epoch) predictions in vivo whole-brain.
    Plot single epoch vs. final epoch
    """
    plt.rcParams.update(fonts.neurips2023())
    plt.rcParams['font.size'] = 12
    # plt.rcParams.update(axes.color(face="k"))

    # create gridspec with the first two rows for the CoD and loss plots
    fig = plt.figure(constrained_layout=True, figsize=(10, 16))
    gs = GridSpec(8, 4, figure=fig)
    # axs for cod and loss plots
    axs_cod_svnn = fig.add_subplot(gs[:2, :2])
    axs_cod_pinn = fig.add_subplot(gs[:2, 2:])
    # set the background color for cod plots and legends to white
    axs_cod_svnn.set_facecolor('w')
    axs_cod_pinn.set_facecolor('w')

    # axs for loss

    # axs for slices
    # T1
    # SVNN T1 (single epoch)
    axs_single_svnn_t1_axi = fig.add_subplot(gs[2, 0])
    axs_single_svnn_t1_cor = fig.add_subplot(gs[3, 0])
    axs_single_svnn_t1_sag = fig.add_subplot(gs[4, 0])
    # SVNN T1 (final epoch)
    axs_final_svnn_t1_axi = fig.add_subplot(gs[2, 1])
    axs_final_svnn_t1_cor = fig.add_subplot(gs[3, 1])
    axs_final_svnn_t1_sag = fig.add_subplot(gs[4, 1])
    # PINN T1 (single epoch)
    axs_single_pinn_t1_axi = fig.add_subplot(gs[2, 2])
    axs_single_pinn_t1_cor = fig.add_subplot(gs[3, 2])
    axs_single_pinn_t1_sag = fig.add_subplot(gs[4, 2])
    # PINN T1 (final epoch)
    axs_final_pinn_t1_axi = fig.add_subplot(gs[2, 3])
    axs_final_pinn_t1_cor = fig.add_subplot(gs[3, 3])
    axs_final_pinn_t1_sag = fig.add_subplot(gs[4, 3])
    # T2
    # SVNN T2 (single epoch)
    axs_single_svnn_t2_axi = fig.add_subplot(gs[5, 0])
    axs_single_svnn_t2_cor = fig.add_subplot(gs[6, 0])
    axs_single_svnn_t2_sag = fig.add_subplot(gs[7, 0])
    # SVNN T2 (final epoch)
    axs_final_svnn_t2_axi = fig.add_subplot(gs[5, 1])
    axs_final_svnn_t2_cor = fig.add_subplot(gs[6, 1])
    axs_final_svnn_t2_sag = fig.add_subplot(gs[7, 1])
    # PINN T2 (single epoch)
    axs_single_pinn_t2_axi = fig.add_subplot(gs[5, 2])
    axs_single_pinn_t2_cor = fig.add_subplot(gs[6, 2])
    axs_single_pinn_t2_sag = fig.add_subplot(gs[7, 2])
    # PINN T2 (final epoch)
    axs_final_pinn_t2_axi = fig.add_subplot(gs[5, 3])
    axs_final_pinn_t2_cor = fig.add_subplot(gs[6, 3])
    axs_final_pinn_t2_sag = fig.add_subplot(gs[7, 3])

    # plot the CoD and loss
    method_list = ['SVNN', 'PINN']
    mask_labels = ['WM', 'WM+GM', 'GM']
    mask_color = ['r', 'b']
    color_alphas = [1, 0.6, 0.3]
    max_cod = 1.005
    epoch_vec = np.arange(nepochs)+1

    # plot validation loss
    # SVNN
    axs_svnn_log_val = axs_cod_svnn.twinx()
    axs_svnn_log_val.set_yscale('log')
    axs_svnn_log_val.plot(epoch_vec, loss_val[0, :nepochs], color='k',
                          linestyle='solid', label=r'$\mathbf{Validation\ loss}$')
    log_val_ylim = axs_svnn_log_val.get_ylim()
    # PINN
    axs_pinn_log_val = axs_cod_pinn.twinx()
    axs_pinn_log_val.set_yscale('log')
    axs_pinn_log_val.plot(epoch_vec, loss_val[1, :nepochs], color='k',
                          linestyle='solid', label=r'$\mathbf{Validation\ loss}$')
    axs_pinn_log_val.set_ylim(log_val_ylim)

    # plot CoD
    # SVNN
    axs_cod_svnn.set_title(r'$\mathbf{SVNN}$')
    axs_cod_svnn.set_xlabel(r'$\mathbf{Epoch}$')
    axs_cod_svnn.set_ylabel(r'$\mathbf{CoD}$')
    axs_svnn_log_val.set_ylabel(r'$\mathbf{Validation\ loss}$')
    axs_pinn_log_val.set_ylabel(r'$\mathbf{Validation\ loss}$')
    for i, mask_label in enumerate(mask_labels):
        # T1
        axs_cod_svnn.plot(epoch_vec, cods[0, :nepochs, 0, i], color=mask_color[0],
                          alpha=color_alphas[i], label=r'$\mathbf{T_1}$' + rf'${mask_label}$')
        axs_cod_svnn.set_ylim(min_cod, max_cod)
    for i, mask_label in enumerate(mask_labels):
        # T2
        axs_cod_svnn.plot(epoch_vec, cods[0, :nepochs, 1, i], color=mask_color[1],
                          alpha=color_alphas[i], label=r'$\mathbf{T_2}$' + rf'${mask_label}$')
        axs_cod_svnn.set_ylim(min_cod, max_cod)
    # add legend
    axs_svnn_log_val.legend(fontsize=10, loc='center', facecolor='w')
    axs_cod_svnn.legend(fontsize=10, loc='center right', facecolor='w')

    # PINN
    axs_cod_pinn.set_title(r'$\mathbf{PINN}$')
    axs_cod_pinn.set_xlabel(r'$\mathbf{Epoch}$')
    axs_cod_pinn.set_ylabel(r'$\mathbf{CoD}$')
    axs_cod_pinn.set_ylim(min_cod, max_cod)
    for i, mask_label in enumerate(mask_labels):
        # T1
        axs_cod_pinn.plot(epoch_vec, cods[1, :nepochs, 0, i], color=mask_color[0],
                          alpha=color_alphas[i], label=r'$\mathbf{T_1}$' + rf'${mask_label}$')
        axs_cod_pinn.set_ylim(min_cod, max_cod)
    for i, mask_label in enumerate(mask_labels):
        # T2
        axs_cod_pinn.plot(epoch_vec, cods[1, :nepochs, 1, i], color=mask_color[1],
                          alpha=color_alphas[i], label=r'$\mathbf{T_2}$' + rf'${mask_label}$')
        axs_cod_pinn.set_ylim(min_cod, max_cod)

    # plot whole-brain slices
    # T1
    axs_single_svnn_t1_axi.imshow(par_sv_single[0][:, :, 0],
                                  cmap=cmaps[0], vmin=vmin[0], vmax=vmax[0])
    axs_single_svnn_t1_cor.imshow(par_sv_single[1][:, :, 0],
                                  cmap=cmaps[0], vmin=vmin[0], vmax=vmax[0])
    axs_single_svnn_t1_sag.imshow(par_sv_single[2][:, :, 0],
                                  cmap=cmaps[0], vmin=vmin[0], vmax=vmax[0])
    axs_final_svnn_t1_axi.imshow(par_sv_final[0][:, :, 0],
                                 cmap=cmaps[0], vmin=vmin[0], vmax=vmax[0])
    axs_final_svnn_t1_cor.imshow(par_sv_final[1][:, :, 0],
                                 cmap=cmaps[0], vmin=vmin[0], vmax=vmax[0])
    axs_final_svnn_t1_sag.imshow(par_sv_final[2][:, :, 0],
                                 cmap=cmaps[0], vmin=vmin[0], vmax=vmax[0])
    axs_single_pinn_t1_axi.imshow(par_mb_single[0][:, :, 0],
                                  cmap=cmaps[0], vmin=vmin[0], vmax=vmax[0])
    axs_single_pinn_t1_cor.imshow(par_mb_single[1][:, :, 0],
                                  cmap=cmaps[0], vmin=vmin[0], vmax=vmax[0])
    axs_single_pinn_t1_sag.imshow(par_mb_single[2][:, :, 0],
                                  cmap=cmaps[0], vmin=vmin[0], vmax=vmax[0])
    axs_final_pinn_t1_axi.imshow(par_mb_final[0][:, :, 0],
                                 cmap=cmaps[0], vmin=vmin[0], vmax=vmax[0])
    axs_final_pinn_t1_cor.imshow(par_mb_final[1][:, :, 0],
                                 cmap=cmaps[0], vmin=vmin[0], vmax=vmax[0])
    axs_cmap_t1 = axs_final_pinn_t1_sag.imshow(par_mb_final[2][:, :, 0],
                                               cmap=cmaps[0], vmin=vmin[0], vmax=vmax[0])
    # T2
    axs_single_svnn_t2_axi.imshow(par_sv_single[0][:, :, 1],
                                  cmap=cmaps[1], vmin=vmin[1], vmax=vmax[1])
    axs_single_svnn_t2_cor.imshow(par_sv_single[1][:, :, 1],
                                  cmap=cmaps[1], vmin=vmin[1], vmax=vmax[1])
    axs_single_svnn_t2_sag.imshow(par_sv_single[2][:, :, 1],
                                  cmap=cmaps[1], vmin=vmin[1], vmax=vmax[1])
    axs_final_svnn_t2_axi.imshow(par_sv_final[0][:, :, 1],
                                 cmap=cmaps[1], vmin=vmin[1], vmax=vmax[1])
    axs_final_svnn_t2_cor.imshow(par_sv_final[1][:, :, 1],
                                 cmap=cmaps[1], vmin=vmin[1], vmax=vmax[1])
    axs_final_svnn_t2_sag.imshow(par_sv_final[2][:, :, 1],
                                 cmap=cmaps[1], vmin=vmin[1], vmax=vmax[1])
    axs_single_pinn_t2_axi.imshow(par_mb_single[0][:, :, 1],
                                  cmap=cmaps[1], vmin=vmin[1], vmax=vmax[1])
    axs_single_pinn_t2_cor.imshow(par_mb_single[1][:, :, 1],
                                  cmap=cmaps[1], vmin=vmin[1], vmax=vmax[1])
    axs_single_pinn_t2_sag.imshow(par_mb_single[2][:, :, 1],
                                  cmap=cmaps[1], vmin=vmin[1], vmax=vmax[1])
    axs_final_pinn_t2_axi.imshow(par_mb_final[0][:, :, 1],
                                 cmap=cmaps[1], vmin=vmin[1], vmax=vmax[1])
    axs_final_pinn_t2_cor.imshow(par_mb_final[1][:, :, 1],
                                 cmap=cmaps[1], vmin=vmin[1], vmax=vmax[1])
    axs_cmap_t2 = axs_final_pinn_t2_sag.imshow(par_mb_final[2][:, :, 1],
                                               cmap=cmaps[1], vmin=vmin[1], vmax=vmax[1])
    # remove all ticks for the whole-brain slices
    for ax in [axs_single_svnn_t1_axi, axs_single_svnn_t1_cor, axs_single_svnn_t1_sag, axs_final_svnn_t1_axi, axs_final_svnn_t1_cor, axs_final_svnn_t1_sag, axs_single_pinn_t1_axi, axs_single_pinn_t1_cor, axs_single_pinn_t1_sag, axs_final_pinn_t1_axi, axs_final_pinn_t1_cor, axs_final_pinn_t1_sag, axs_single_svnn_t2_axi, axs_single_svnn_t2_cor, axs_single_svnn_t2_sag, axs_final_svnn_t2_axi, axs_final_svnn_t2_cor, axs_final_svnn_t2_sag, axs_single_pinn_t2_axi, axs_single_pinn_t2_cor, axs_single_pinn_t2_sag, axs_final_pinn_t2_axi, axs_final_pinn_t2_cor, axs_final_pinn_t2_sag]:
        ax.set_xticks([])
        ax.set_yticks([])
        # remove frame around the slices
        ax.axis('off')

    # add colorbars below subplots
    # create colorbar axes
    # [left, bottom, width, height]
    axs_cbar_t1 = fig.add_axes([0.95, 0.45, 0.015, 0.25])
    axs_cbar_t2 = fig.add_axes([0.95, 0.05, 0.015, 0.25])
    fig.colorbar(axs_cmap_t1, cax=axs_cbar_t1, orientation='vertical')
    fig.colorbar(axs_cmap_t2, cax=axs_cbar_t2, orientation='vertical')
    # add titles
    axs_single_svnn_t1_axi.set_title(
        r'$\mathbf{T1}$'+' ' + r'$\mathbf{SVNN}$'+'\n'+r'$\mathbf{(Single Epoch)}$')
    axs_final_svnn_t1_axi.set_title(
        r'$\mathbf{T1}$'+' ' + r'$\mathbf{SVNN}$'+'\n'+r'$\mathbf{(Final Epoch)}$')
    axs_single_pinn_t1_axi.set_title(
        r'$\mathbf{T1}$'+' ' + r'$\mathbf{PINN}$'+'\n'+r'$\mathbf{(Single Epoch)}$')
    axs_final_pinn_t1_axi.set_title(
        r'$\mathbf{T1}$'+' ' + r'$\mathbf{PINN}$'+'\n'+r'$\mathbf{(Final Epoch)}$')
    axs_single_svnn_t2_axi.set_title(
        r'$\mathbf{T2}$'+' ' + r'$\mathbf{SVNN}$'+'\n'+r'$\mathbf{(Single Epoch)}$')
    axs_final_svnn_t2_axi.set_title(
        r'$\mathbf{T2}$'+' ' + r'$\mathbf{SVNN}$'+'\n'+r'$\mathbf{(Final Epoch)}$')
    axs_single_pinn_t2_axi.set_title(
        r'$\mathbf{T2}$'+' ' + r'$\mathbf{PINN}$'+'\n'+r'$\mathbf{(Single Epoch)}$')
    axs_final_pinn_t2_axi.set_title(
        r'$\mathbf{T2}$'+' ' + r'$\mathbf{PINN}$'+'\n'+r'$\mathbf{(Final Epoch)}$')

    # add text a, b and c
    fig.text(0.01, 0.985, 'a', fontweight='bold', fontsize=30)
    fig.text(0.01, 0.713, 'b', fontweight='bold', fontsize=30)
    fig.text(0.01, 0.345, 'c', fontweight='bold', fontsize=30)

    plt.savefig(path_save, dpi=300)
