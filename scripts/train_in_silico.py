"""Train model with in silico data.

- Supervised OR semi-supervised (model-based/physics-informed) training
- 2-3 Target parameter T1, T2 (Delta B0)

Author, florian.birk@tuebingen.mpg.de (Florian Birk), February 2024
"""
# %% Import
# own modules
import paths
from model import Qbssfp
from fcns_mb import *
from fcns_model import train, test, exec_mb
from fcns_dataprep import train_val_test_split, apply_transform, min_max_transform, revert_min_max_transform, revert_z_transform
from fcns_performance import cod, rmse, mean_std_conditional_mean_interval
from fcns_plot import *
from datasets import UnsupervisedDataset, SupervisedDataset
# external modules
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb
import argparse
import yaml


if __name__ == '__main__':
    # %% Parse arguments
    parser = argparse.ArgumentParser()
    # number of hidden layers an    d neurons (nneurons) and learning rate can be parsed
    parser.add_argument('--model', '-m', type=str, default='mb',
                        help='Which model type to use (default: mb)')
    parser.add_argument('--snr', '-s', type=str,
                        default='inf', help='SNR level (default: inf)')
    parser.add_argument('--distr', '-d', type=str, default='uniform',
                        help='Choose between uniform, uniform_ext, and in_vivo (default: uniform)')
    parser.add_argument('--npc', '-n', type=int, default=12,
                        help='Number of phase cycles (default: 12)')
    parser.add_argument('--b0fit', '-b', action='store_true', default=False,
                        help='Fit B0 in model-based training')
    parser.add_argument('--realimag', '-ri', action='store_true', default=False,
                        help='Use the real and imaginary parts of the Fn modes as input')
    args = parser.parse_args()

    # load cfg
    cfg = yaml.safe_load(
        open(paths.cfgs / '1.nn_train' / 'train_in_silico.yml'))
    model_type = args.model
    snr = args.snr
    distr = args.distr
    npc = args.npc
    cfg['model'] = model_type
    cfg['snr'] = snr
    cfg['distr'] = distr
    cfg['npc'] = npc
    # args.b0fit, args.realimag = True, True
    cfg['b0fit'] = args.b0fit
    cfg['realimag'] = args.realimag
    # if args.realimag:
    #     cfg['b0fit'] = True
    if distr == 'uniform_ext':
        cfg['param_max'] = cfg['param_max_ext']
    if model_type == 'mb':
        cfg['mb'] = True
    else:
        cfg['mb'] = False

    if args.b0fit:
        # multiply max and min of b0 with torch.pi
        cfg['param_min'][-1] = cfg['param_min'][-1] * np.pi
        cfg['param_max'][-1] = cfg['param_max'][-1] * np.pi
        # cfg['param_max_ext'][-1] = cfg['param_max_ext'][-1] * np.pi
        ntar = 3
    else:
        # remove b0 from the target parameters (bo = 0)
        cfg['param_min'] = cfg['param_min'][:-1]
        cfg['param_max'] = cfg['param_max'][:-1]
        ntar = 2
        cfg['nneurons'] = [4, 64, 64, 2]

    # %% Initialize wandb
    wandb.init(
        project=cfg['project'],
        entity=cfg['entity'],
        notes=cfg['notes'],
        config=cfg)

    print('++++++ CONFIG ++++++')
    if model_type == 'mb':
        print('Model: Semi-Supervised (physics-informed PINN) training selected')
    elif model_type == 'sv':
        print('Model: Supervised (SVNN) training selected')
    else:
        print('Wrong model selection')
    print(f'Selected SNR level: {snr}')
    print(f'Target parameter distribution: {distr}')
    print(f'Number of phase cycles: {npc}')
    print(f'Fit B0: {args.b0fit}')
    print(f'Use real and imaginary parts of the Fn modes: {args.realimag}')
    print(cfg)
    print('++++++++++++++++++++\n')
    # %% Set random seed
    torch.manual_seed(2308)

    # %% Load data
    path_nn_in_silico = paths.data / 'in_silico' / 'train'
    path_nn_in_vivo = paths.data / 'in_vivo' / 'test'
    npc_name = f'{npc}pc'
    # in silico
    if args.b0fit:
        sig = torch.load(path_nn_in_silico /
                         f'bssfp_complex_{distr}_{npc_name}.pt').cfloat()
        par = torch.load(path_nn_in_silico /
                         f't1t2b0b1_{distr}_{npc_name}.pt').float()
    else:
        sig = torch.load(path_nn_in_silico /
                         f'bssfp_complex_nob0_{distr}_{npc_name}.pt').cfloat()
        par = torch.load(path_nn_in_silico /
                         f't1t2b0b1_nob0_{distr}_{npc_name}.pt').float()

    # in vivo (1p25 mm³)
    if args.realimag:
        sig_in_vivo_all = torch.load(
            path_nn_in_vivo / f'sig_all_realimag_nn.{npc_name}.pt').float()
        sig_in_vivo_all = torch.cat(
            (sig_in_vivo_all[:, :3], sig_in_vivo_all[:, 3].unsqueeze(-1), sig_in_vivo_all[:, 5].unsqueeze(-1), sig_in_vivo_all[:, -1].unsqueeze(-1)), dim=1)
    else:
        sig_in_vivo_all = torch.load(
            path_nn_in_vivo / f'sig_all_nn.{npc_name}.pt').float()
    mask_3d = torch.load(
        path_nn_in_vivo / f'mask_3d.{npc_name}.pt').float().numpy()
    if not args.b0fit:  # remove Fn phase from input if B0 is not fitted
        sig_in_vivo_all = torch.cat(
            (sig_in_vivo_all[:, :3], sig_in_vivo_all[:, -1].unsqueeze(-1)), dim=1)

    # %% Split data
    sig_train, sig_val, sig_test, par_train, par_val, par_test = train_val_test_split(
        sig, par, cfg['split'])
    # take T1, T2, and B0 as target parameters
    b1_train = par_train[:, -1].unsqueeze(-1)
    b1_val = par_val[:, -1].unsqueeze(-1)
    b1_test = par_test[:, -1].unsqueeze(-1)
    b0_train = par_train[:, 2].unsqueeze(-1)
    b0_val = par_val[:, 2].unsqueeze(-1)
    b0_test = par_test[:, 2].unsqueeze(-1)
    par_train = par_train[:, :ntar]
    par_val = par_val[:, :ntar]
    par_test = par_test[:, :ntar]

    # %% Initialize model
    model = Qbssfp(cfg).to(cfg['device'])
    loss = nn.MSELoss().to(cfg['device'])
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg['lr'])  # SGD did not help
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    print(f'Model Learning rate: {cfg["lr"]}\n')

    # calculate the mean and std of the training data
    fn_complex = calc_Fn(sig_train, model.phi_nom_rad,
                         Np=model.Np, device=cfg['device'])
    fn_complex = norm2_vox(fn_complex)  # normalize using euclidean norm
    if args.b0fit:
        if args.realimag:
            sig = torch.cat(
                (torch.real(fn_complex), return_without_f0_imag_2d(fn_complex, Np=1), b1_train), dim=1)
        else:
            fn_mag = torch.abs(fn_complex)
            fn_pha = mod_fn_train(
                fn_complex, Np=model.Np, b0=b0_train.squeeze())  # modulate phase of Fn modes (except F0), like phase unwrapping
            sig = torch.cat((fn_mag, fn_pha, b1_train), dim=1)
    else:
        if args.realimag:
            sig = torch.cat((torch.real(fn_complex), b1_train), dim=1)
        else:
            fn_mag = torch.abs(fn_complex)
            sig = torch.cat((fn_mag, b1_train), dim=1)
    sig_mean = torch.mean(sig, dim=0)
    sig_std = torch.std(sig, dim=0)

    # %% get noise level based on snr
    if cfg['snr'] == 'inf':
        noise_level = 0
    else:
        # sig_sim_mag: Average simulated signal of the magnitude of the complex sum
        # snr: SNR of the magnitude of the complex sum
        noise_level_mag = torch.tensor(cfg['sig_sim_mag']/int(cfg['snr']))
        noise_level = noise_level_mag / torch.sqrt(torch.tensor(2.0))
    print(
        f'Noise level added to pc-bSSFP (with SNR {cfg["snr"]}): {noise_level}\n')

    # Add noise to input test data if selected
    sig_test = (torch.real(sig_test) + torch.randn(sig_test.shape) * noise_level) + \
        1j * (torch.imag(sig_test) + torch.randn(sig_test.shape) * noise_level)
    # calculate the input for test data
    # calculate the mean and std of the training data
    fn_complex_test = calc_Fn(sig_test, model.phi_nom_rad,
                              Np=model.Np, device=cfg['device'])
    fn_complex_test = norm2_vox(fn_complex_test)
    # concat fn_mag, fn_pha, and b1
    if args.b0fit:
        if args.realimag:
            sig_test_input = torch.cat(
                (torch.real(fn_complex_test), return_without_f0_imag_2d(fn_complex_test, Np=1), b1_test), dim=1)
        else:
            fn_mag_test = torch.abs(fn_complex_test)
            fn_pha_test = mod_fn_train(
                fn_complex_test, Np=model.Np, b0=b0_test.squeeze())
            sig_test_input = torch.cat(
                (fn_mag_test, fn_pha_test, b1_test), dim=1)
    else:
        if args.realimag:
            sig_test_input = torch.cat(
                (torch.real(fn_complex_test), b1_test), dim=1)
        else:
            fn_mag_test = torch.abs(fn_complex_test)
            sig_test_input = torch.cat((fn_mag_test, b1_test), dim=1)

    # %% Apply transform to test data
    sig_test_z = apply_transform(
        sig_test_input, sig_mean, sig_std, cfg['device'])
    sig_in_vivo_all_z = apply_transform(
        sig_in_vivo_all, sig_mean, sig_std, cfg['device'])
    # sig_test_z = sig_test_input

    # %% Prepare dataloader
    if model_type == 'mb':
        trainloader = DataLoader(UnsupervisedDataset(
            sig_train, b1_train, b0_train), batch_size=cfg['mbatch'], shuffle=True)
        valloader = DataLoader(UnsupervisedDataset(
            sig_val, b1_val, b0_val), batch_size=cfg['mbatch'], shuffle=True)
        testloader = DataLoader(UnsupervisedDataset(
            sig_test, b1_test, b0_test), batch_size=cfg['mbatch'], shuffle=True)
    else:
        # apply transform to target parameter
        par_train_transformed = min_max_transform(
            par_train, cfg['param_min'], cfg['param_max'])
        par_val_transformed = min_max_transform(
            par_val, cfg['param_min'], cfg['param_max'])
        par_test_transformed = min_max_transform(
            par_test, cfg['param_min'], cfg['param_max'])
        trainloader = DataLoader(SupervisedDataset(
            sig_train, par_train_transformed, b1_train, b0_train), batch_size=cfg['mbatch'], shuffle=True, pin_memory=True)
        valloader = DataLoader(SupervisedDataset(
            sig_val, par_val_transformed, b1_val, b0_val), batch_size=cfg['mbatch'], shuffle=True, pin_memory=True)
        testloader = DataLoader(SupervisedDataset(
            sig_test, par_test_transformed, b1_test, b0_test), batch_size=cfg['mbatch'], shuffle=True, pin_memory=True)
    # %% Model training
    best_model = train(
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
        wandb=wandb
    )
    # get the run name from wandb
    run_name_split = (wandb.run.name).split('-')
    run_name = run_name_split[0] + '-' + run_name_split[1]
    run_id = run_name_split[-1]
    path_model_save = paths.dnn_models / wandb.run.project
    path_model_save.mkdir(parents=True, exist_ok=True)
    # save best model to ssd
    torch.save(best_model.state_dict(), path_model_save /
               f'{run_id}_{model_type}_{snr}_{distr}_{run_name}.pt')

    # %% Test model
    # test loss
    avg_test_loss = test(
        best_model,
        testloader,
        loss,
        sig_mean,
        sig_std,
        noise_level,
        cfg['mb'],
        cfg['device']
    )
    wandb.log({'loss_test': avg_test_loss})
    print(f'Average test loss: {avg_test_loss:.4f} \n')

    # %%
    ###### Evaluate model performance - IN SILICO #######
    # 1. Parameter estimation
    par_test = par_test.numpy()
    best_model.eval()
    with torch.no_grad():
        if best_model.mb:
            # Get NN output MRI signal --> tissue parameter
            x = best_model(sig_test_z)
            # get parameters from model (in 2d shape)
            par_pred = getparams(x, best_model.param_min,
                                 best_model.param_max, best_model.device)
            sig_pred = exec_mb(
                model=best_model,
                x=sig_test_z,
                sig_mean=sig_mean,
                sig_std=sig_std,
                b1=b1_test.squeeze()).detach()
        else:   # supervised training
            par_pred = best_model(sig_test_z)
    # get parameters (and signal) estimates
    par_mean = par_pred
    if best_model.mb:
        sig_mean = revert_z_transform(
            sig_pred, sig_mean[:-1], sig_std[:-1]).numpy()
        # sig_mean = sig_pred.numpy()
        par_mean = par_mean.numpy()
    else:
        par_mean = revert_min_max_transform(
            par_mean, cfg['param_min'], cfg['param_max']).numpy()

    # 2. Compute evaluation metrics (CoD, RMSE)
    cod_par = cod(par_test, par_mean)
    rmse_par = rmse(par_test, par_mean)
    print(f'COD (PAR): {cod_par}\nRMSE (PAR): {rmse_par}\n')

    if best_model.mb:
        cod_sig = cod(sig_test_input[:, :-1].numpy(), sig_mean)
        rmse_sig = rmse(sig_test_input[:, :-1].numpy(), sig_mean)
        print(f'COD (SIG): {cod_sig}\nRMSE: {rmse_sig}\n')

    # 3. Create plots - IN SILICO
    steps = 200

    t1_range = np.linspace(cfg['param_min']
                           [0], cfg['param_max'][0], steps)
    t2_range = np.linspace(cfg['param_min']
                           [1], cfg['param_max'][1], steps)
    if par_test.shape[-1] == 3:
        b0_range = np.linspace(
            cfg['param_min'][2], cfg['param_max'][2], steps)

    # calculate the mean and std for certain intervals
    par_mean_steps, par_std_steps = mean_std_conditional_mean_interval(
        par_test, par_mean, cfg['param_min'], cfg['param_max'], steps)
    if par_test.shape[-1] == 2:
        plot_scatter_t1_t2_mean(par_test, par_mean)
        plot_mean_std_t1_t2(par_test, par_mean_steps,
                            par_std_steps, t1_range, t2_range)
    elif par_test.shape[-1] == 3:
        plot_scatter_t1_t2_b0_mean(par_test, par_mean)
        plot_mean_std_t1_t2_b0(par_test, par_mean_steps,
                               par_std_steps, t1_range, t2_range, b0_range)
