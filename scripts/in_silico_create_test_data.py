"""Prepare the raw in silico pc-bSSFP test data (grid or in vivo distribution)

- Test grid with 200 steps along each parameter dimension. Uniform distribution range
- In vivo distribution test data with 40.000 samples

Author: florian.birk@tuebingen.mpg.de (Florian Birk), February 2024
"""
import torch
import yaml
import argparse
import numpy as np
import paths
from model import Qbssfp
from fcns_mb import sim_bssfp, phase_correction_2d


if __name__ == '__main__':
    torch.manual_seed(2311)
    # Parse the config file
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, default='uniform')
    args = parser.parse_args()
    path_density = paths.data / 'in_vivo' / 'density_map'
    path_density.mkdir(parents=True, exist_ok=True)
    path_in_silico = paths.data / 'in_silico' / 'test'
    path_in_silico.mkdir(parents=True, exist_ok=True)
    # %% Uniform test grid (200 x 200 steps)
    cfg = yaml.safe_load(
        open(paths.cfgs / '0.simulation' / f'cfg_{args.config}.yml'))
    cfg['nsteps'] = 200
    cfg['b0fit'] = False
    cfg['realimag'] = False

    if args.config == 'in_vivo':
        cfg['nsamples_init'] = 48500  # Used
        t1t2_density = torch.from_numpy(
            np.load(path_density / f't1t2_density.npy'))
        t1_edges = torch.from_numpy(
            np.load(path_density / f't1_edges.npy'))
        t2_edges = torch.from_numpy(
            np.load(path_density / f't2_edges.npy'))

        # create target parameters
        par = torch.zeros((0, 2))
        t1t2_hist = torch.round(
            t1t2_density * cfg['nsamples_init']).to(torch.int)
        shape = t1t2_density.shape
        for i in range(shape[0]):
            for j in range(shape[1]):
                if t1t2_hist[i, j] > 0:
                    par_temp = torch.zeros((t1t2_hist[i, j], 2))
                    par_temp[:, 0] = torch.rand(t1t2_hist[i, j]) * \
                        (t1_edges[i+1] - t1_edges[i]) + t1_edges[i]
                    par_temp[:, 1] = torch.rand(t1t2_hist[i, j]) * \
                        (t2_edges[j+1] - t2_edges[j]) + t2_edges[j]
                    par = torch.vstack((par, par_temp))
                else:
                    continue
        # For each distribution the same
        print(f'Shape of par (before removal): {par.shape}')
        # remove all rows where T1 is smaller than T2
        par = par[torch.greater_equal(par[:, 0], par[:, 1]), :]
        print(f'Shape of par (after removal): {par.shape}')
        # check if t1 is always greater than t2
        if torch.all(torch.greater_equal(par[:, 0], par[:, 1])):
            print('All T1 values are greater than T2 values. Good!')

        # shuffle the parameters
        idx = torch.randperm(par.shape[0])
        par = par[idx, :]

        # take only cfg['nsamples'] samples
        nsamples_final = 40000
        par = par[:nsamples_final, :]

        # create 1d arrays and concatenate
        t1_1d = par[:, 0]
        t2_1d = par[:, 1]
        b0_1d = torch.zeros_like(t1_1d)
        b1_1d = torch.ones_like(t1_1d)
        par = torch.cat((t1_1d.unsqueeze(1), t2_1d.unsqueeze(
            1), b0_1d.unsqueeze(1), b1_1d.unsqueeze(1)), dim=1)
    else:
        # create mesh with cfg['nsteps'] steps
        print(f'Create mesh with {cfg["nsteps"]} steps in each dimension')
        t1_vec = torch.linspace(cfg['param_min'][0],
                                cfg['param_max'][0], cfg['nsteps'])
        t2_vec = torch.linspace(cfg['param_min'][1],
                                cfg['param_max'][1], cfg['nsteps'])
        t1_mesh, t2_mesh = torch.meshgrid(t1_vec, t2_vec)

        # reshape to 1D
        t1_mesh_1d = t1_mesh.reshape(-1)
        t2_mesh_1d = t2_mesh.reshape(-1)
        b0_1d = torch.zeros_like(t1_mesh_1d)
        b1_1d = torch.ones_like(t1_mesh_1d)
        par = torch.cat((t1_mesh_1d.unsqueeze(1), t2_mesh_1d.unsqueeze(
            1), b0_1d.unsqueeze(1), b1_1d.unsqueeze(1)), dim=1)

    # Simulate data
    model = Qbssfp(cfg)

    # create pc-bssfp signal
    sig = sim_bssfp(
        phi_nom_hz=model.phi_nom_hz,
        tr=model.tr,
        te=model.te,
        M0=model.M0,
        fa_nom=model.fa_nom,
        b1=par[:, 3],
        t1=par[:, 0],
        t2=par[:, 1],
        b0=par[:, 2],
    )

    # perform phase correction on the complex bssfp signal
    sig = phase_correction_2d(sig)

    # save the data
    if args.config == 'in_vivo':
        torch.save(par, path_in_silico /
                   f't1t2b0b1_{args.config}_test.pt')
        torch.save(sig, path_in_silico /
                   f'bssfp_complex_{args.config}_test.pt')
    else:
        torch.save(par, path_in_silico /
                   f't1t2b0b1_{args.config}_test.pt')
        torch.save(sig, path_in_silico /
                   f'bssfp_complex_{args.config}_test.pt')
        torch.save(t1_mesh, path_in_silico /
                   f't1_mesh_{args.config}_test.pt')
        torch.save(t2_mesh, path_in_silico /
                   f't2_mesh_{args.config}_test.pt')
