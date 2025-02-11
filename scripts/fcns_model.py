"""Functions used for model training and testing.

Author: florian.birk@tuebingen.mpg.de (Florian Birk), February 2024
"""
# %% Import
import copy
import torch
from tqdm import tqdm
import time
from fcns_mb import *
from fcns_dataprep import apply_transform


def train(model, trainloader, valloader, sig_mean, sig_std, noise_level, loss, optimizer, epochs, patience, mb, device, wandb, path_save=None, return_epochs=False):

    # init
    early_stop = True if patience > 0 else False
    best_loss = float('inf')
    counter = 0
    if return_epochs:
        time_list = []
        train_loss_list = []
        val_loss_list = []
    # train
    with tqdm((range(epochs)), unit='epoch') as tepoch:
        for epoch in tepoch:
            start_epoch = time.time()
            tepoch.set_description(f'Epoch {epoch+1}/{epochs}')
            # train
            train_loss = 0.0
            model.train()
            if mb:
                for sig, b1, b0 in trainloader:
                    sig, b1, b0 = sig.to(device), b1.to(device), b0.to(device)
                    # add noise level to real and imaginary parts
                    sig = (torch.real(sig) + torch.randn(sig.shape)*noise_level) + \
                        1j*(torch.imag(sig) + torch.randn(sig.shape)*noise_level)
                    # compute fn modes from pc-bssfp signal
                    fn_complex = calc_Fn(sig, model.phi_nom_rad, Np=model.Np)
                    fn_complex = norm2_vox(fn_complex)
                    # fn_complex[:, model.Np-1] = fn_complex[:, model.Np-1] * \
                    #     torch.exp(torch.tensor(1j*torch.pi))
                    if model.b0fit:
                        if model.realimag:
                            sig = torch.cat(
                                (torch.real(fn_complex), return_without_f0_imag_2d(fn_complex, Np=1), b1), dim=1)
                        else:
                            fn_mag = torch.abs(fn_complex)
                            fn_pha = mod_fn_train(
                                fn_complex, Np=model.Np, b0=b0.squeeze())
                            sig = torch.cat(
                                (fn_mag, fn_pha, b1), dim=1)
                    else:
                        if model.realimag:
                            sig = torch.cat(
                                (torch.real(fn_complex), b1), dim=1)
                        else:
                            fn_mag = torch.abs(fn_complex)
                            sig = torch.cat(
                                (fn_mag, b1), dim=1)
                    # normalize
                    sig = (sig - sig_mean)/sig_std
                    # set b1 for model
                    b1 = b1.squeeze()
                    # zero the parameter gradients
                    optimizer.zero_grad()
                    # forward
                    sig_pred = exec_mb(model, sig, sig_mean,
                                       sig_std, b1)
                    # loss
                    loss_train = loss(sig_pred, sig[:, :-1])
                    # backward
                    loss_train.backward()
                    # optimize
                    optimizer.step()
                    # loss
                    train_loss += loss_train.item()
            else:  # supervised
                for sig, par, b1, b0 in trainloader:
                    sig, par, b1, b0 = sig.to(device), par.to(
                        device), b1.to(device), b0.to(device)
                    # add noise level to real and imaginary parts
                    sig = (torch.real(sig) + torch.randn(sig.shape)*noise_level) + \
                        1j*(torch.imag(sig) + torch.randn(sig.shape)*noise_level)
                    # compute fn modes from pc-bssfp signal
                    fn_complex = calc_Fn(sig, model.phi_nom_rad, Np=model.Np)
                    fn_complex = norm2_vox(fn_complex)
                    # fn_complex[:, model.Np-1] = fn_complex[:, model.Np-1] * \
                    #     torch.exp(torch.tensor(1j*torch.pi))
                    if model.b0fit:
                        if model.realimag:
                            sig = torch.cat(
                                (torch.real(fn_complex), return_without_f0_imag_2d(fn_complex, Np=1), b1), dim=1)
                        else:
                            fn_mag = torch.abs(fn_complex)
                            fn_pha = mod_fn_train(
                                fn_complex, Np=model.Np, b0=b0.squeeze())
                            sig = torch.cat(
                                (fn_mag, fn_pha, b1), dim=1)
                    else:
                        if model.realimag:
                            sig = torch.cat(
                                (torch.real(fn_complex), b1), dim=1)
                        else:
                            fn_mag = torch.abs(fn_complex)
                            sig = torch.cat(
                                (fn_mag, b1), dim=1)
                    # normalize
                    sig = (sig - sig_mean)/sig_std
                    # zero the parameter gradients
                    optimizer.zero_grad()
                    # forward
                    par_pred = model(sig)
                    # loss
                    loss_train = loss(par_pred, par)
                    # backward
                    loss_train.backward()
                    # optimize
                    optimizer.step()
                    # loss
                    train_loss += loss_train.item()
            avg_train_loss = train_loss/len(trainloader)
            tepoch.set_postfix(loss=avg_train_loss)
            # Log the loss logaritmically and normal
            wandb.log({'loss_train': avg_train_loss}, step=epoch+1)
            # val
            val_loss = 0.0
            model.eval()
            with torch.no_grad():
                if mb:
                    for sig, b1, b0 in valloader:
                        sig, b1, b0 = sig.to(device), b1.to(
                            device), b0.to(device)
                        # add noise level to real and imaginary parts
                        sig = (torch.real(sig) + torch.randn(sig.shape)*noise_level) + \
                            1j*(torch.imag(sig) +
                                torch.randn(sig.shape)*noise_level)
                        # compute fn modes from pc-bssfp signal
                        fn_complex = calc_Fn(
                            sig, model.phi_nom_rad, Np=model.Np)
                        fn_complex = norm2_vox(fn_complex)
                        # fn_complex[:, model.Np-1] = fn_complex[:, model.Np-1] * \
                        #     torch.exp(torch.tensor(1j*torch.pi))
                        if model.b0fit:
                            if model.realimag:
                                sig = torch.cat(
                                    (torch.real(fn_complex), return_without_f0_imag_2d(fn_complex, Np=1), b1), dim=1)
                            else:
                                fn_mag = torch.abs(fn_complex)
                                fn_pha = mod_fn_train(
                                    fn_complex, Np=model.Np, b0=b0.squeeze())
                                sig = torch.cat(
                                    (fn_mag, fn_pha, b1), dim=1)
                        else:
                            if model.realimag:
                                sig = torch.cat(
                                    (torch.real(fn_complex), b1), dim=1)
                            else:
                                fn_mag = torch.abs(fn_complex)
                                sig = torch.cat(
                                    (fn_mag, b1), dim=1)
                        # normalize
                        sig = (sig - sig_mean)/sig_std
                        # set b1 for model
                        b1 = b1.squeeze()
                        # forward
                        sig_pred = exec_mb(model, sig, sig_mean,
                                           sig_std, b1)
                        # loss
                        loss_val = loss(sig_pred, sig[:, :-1])
                        # loss
                        val_loss += loss_val.item()
                else:
                    for sig, par, b1, b0 in valloader:
                        sig, par, b1, b0 = sig.to(device), par.to(
                            device), b1.to(device), b0.to(device)
                        # add noise level to real and imaginary parts
                        sig = (torch.real(sig) + torch.randn(sig.shape)*noise_level) + \
                            1j*(torch.imag(sig) +
                                torch.randn(sig.shape)*noise_level)
                        # compute fn modes from pc-bssfp signal
                        fn_complex = calc_Fn(
                            sig, model.phi_nom_rad, Np=model.Np)
                        fn_complex = norm2_vox(fn_complex)
                        # fn_complex[:, model.Np-1] = fn_complex[:, model.Np-1] * \
                        #     torch.exp(torch.tensor(1j*torch.pi))
                        if model.b0fit:
                            if model.realimag:
                                sig = torch.cat(
                                    (torch.real(fn_complex), return_without_f0_imag_2d(fn_complex, Np=1), b1), dim=1)
                            else:
                                fn_mag = torch.abs(fn_complex)
                                fn_pha = mod_fn_train(
                                    fn_complex, Np=model.Np, b0=b0.squeeze())
                                sig = torch.cat(
                                    (fn_mag, fn_pha, b1), dim=1)
                        else:
                            if model.realimag:
                                sig = torch.cat(
                                    (torch.real(fn_complex), b1), dim=1)
                            else:
                                fn_mag = torch.abs(fn_complex)
                                sig = torch.cat(
                                    (fn_mag, b1), dim=1)
                        # normalize
                        sig = (sig - sig_mean)/sig_std
                        # forward
                        par_pred = model(sig)
                        # loss
                        loss_val = loss(par_pred, par)
                        # loss
                        val_loss += loss_val.item()
            avg_val_loss = val_loss/len(valloader)
            wandb.log({'loss_val': avg_val_loss}, step=epoch+1)
            # scheduler.step(avg_val_loss)
            # scheduler.step()
            # save best model
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                best_model = copy.deepcopy(model)
                counter = 0
            else:
                counter += 1
            if early_stop and counter >= patience:
                print(f'Early stopping after {epoch+1} epochs')
                break
            if return_epochs:
                return_epoch = epoch + 1
                end_epoch = time.time()
                time_list.append(end_epoch - start_epoch)
                train_loss_list.append(avg_train_loss)
                val_loss_list.append(avg_val_loss)
                if path_save is not None:
                    if mb:
                        torch.save(model.state_dict(), path_save /
                                   f'{return_epoch}-epoch-mb.pt')
                    else:
                        torch.save(model.state_dict(), path_save /
                                   f'{return_epoch}-epoch-sv.pt')
        if return_epochs:
            return best_model, time_list, train_loss_list, val_loss_list
        else:
            return best_model


def test(model, testloader, loss, sig_mean, sig_std, noise_level, mb, device):
    test_loss = 0.0
    model.eval()
    with torch.no_grad():
        if mb:
            for sig, b1, b0 in testloader:
                sig, b1, b0 = sig.to(device), b1.to(device), b0.to(device)
                # add noise level to real and imaginary parts
                sig = (torch.real(sig) + torch.randn(sig.shape)*noise_level) + \
                    1j*(torch.imag(sig) + torch.randn(sig.shape)*noise_level)
                # compute fn modes from pc-bssfp signal
                fn_complex = calc_Fn(sig, model.phi_nom_rad, Np=model.Np)
                fn_complex = norm2_vox(fn_complex)
                # fn_complex[:, model.Np-1] = fn_complex[:, model.Np-1] * \
                #     torch.exp(torch.tensor(1j*torch.pi))
                if model.b0fit:
                    if model.realimag:
                        sig = torch.cat(
                            (torch.real(fn_complex), return_without_f0_imag_2d(fn_complex, Np=1), b1), dim=1)
                    else:
                        fn_mag = torch.abs(fn_complex)
                        fn_pha = mod_fn_train(
                            fn_complex, Np=model.Np, b0=b0.squeeze())
                        sig = torch.cat(
                            (fn_mag, fn_pha, b1), dim=1)
                else:
                    if model.realimag:
                        sig = torch.cat((torch.real(fn_complex), b1), dim=1)
                    else:
                        fn_mag = torch.abs(fn_complex)
                        sig = torch.cat(
                            (fn_mag, b1), dim=1)
                # normalize
                sig = (sig - sig_mean)/sig_std
                # set b1 for model
                b1 = b1.squeeze()
                # forward
                sig_pred = exec_mb(model, sig, sig_mean,
                                   sig_std, b1)
                # loss
                loss_test = loss(sig_pred, sig[:, :-1])
                # loss
                test_loss += loss_test.item()
        else:
            for sig, par, b1, b0 in testloader:
                sig, par, b1, b0 = sig.to(device), par.to(
                    device), b1.to(device), b0.to(device)
                # add noise level to real and imaginary parts
                sig = (torch.real(sig) + torch.randn(sig.shape)*noise_level) + \
                    1j*(torch.imag(sig) + torch.randn(sig.shape)*noise_level)
                # compute fn modes from pc-bssfp signal
                fn_complex = calc_Fn(sig, model.phi_nom_rad, Np=model.Np)
                fn_complex = norm2_vox(fn_complex)
                # fn_complex[:, model.Np-1] = fn_complex[:, model.Np-1] * \
                #     torch.exp(torch.tensor(1j*torch.pi))
                if model.b0fit:
                    if model.realimag:
                        sig = torch.cat(
                            (torch.real(fn_complex), return_without_f0_imag_2d(fn_complex, Np=1), b1), dim=1)
                    else:
                        fn_mag = torch.abs(fn_complex)
                        fn_pha = mod_fn_train(
                            fn_complex, Np=model.Np, b0=b0.squeeze())
                        sig = torch.cat(
                            (fn_mag, fn_pha, b1), dim=1)
                else:
                    if model.realimag:
                        sig = torch.cat((torch.real(fn_complex), b1), dim=1)
                    else:
                        fn_mag = torch.abs(fn_complex)
                        sig = torch.cat(
                            (fn_mag, b1), dim=1)
                # normalize
                sig = (sig - sig_mean)/sig_std
                # forward
                par_pred = model(sig)
                # loss
                loss_test = loss(par_pred, par)
                # loss
                test_loss += loss_test.item()
    avg_test_loss = test_loss/len(testloader)
    return avg_test_loss


def exec_mb(model, x, sig_mean, sig_std, b1):
    """Run analytical path of model-based model."""
    # Get NN output MRI signal --> tissue parameter
    x = model(x)

    # Map the NN output to tissue parameter range
    x = getparams(x, model.param_min, model.param_max, model.device)

    if x.shape[-1] > 2:
        b0 = x[:, 2]
    else:
        b0 = torch.Tensor([0])
    # simulate the pc-bssfp signal from the tissue parameters
    x = sim_bssfp(
        phi_nom_hz=model.phi_nom_hz,
        tr=model.tr,
        te=model.te,
        M0=model.M0,
        fa_nom=model.fa_nom,
        b1=b1,
        t1=x[:, 0],
        t2=x[:, 1],
        b0=b0,
    )

    # perform phase correction on the complex bssfp signal
    x = phase_correction_2d(x)

    # calulate the configuration modes from the pc-bssfp signal
    x = calc_Fn(x, model.phi_nom_rad, model.Np, model.device)
    x = norm2_vox(x)
    # x[:, model.Np-1] = x[:, model.Np-1] * torch.exp(torch.tensor(1j*torch.pi))
    if model.b0fit:
        if model.realimag:
            x = torch.cat(
                (torch.real(x), return_without_f0_imag_2d(x, Np=1)), dim=1)
        else:
            x_mag = torch.abs(x)
            x_pha = mod_fn_train(x, Np=model.Np, b0=b0)
            x = torch.cat((x_mag, x_pha), dim=1)
    else:
        if model.realimag:
            x = torch.real(x)
        else:
            x = torch.abs(x)

    # apply transformation on signal prediction
    x = apply_transform(x, sig_mean[:-1], sig_std[:-1], device=model.device)
    return x
