"""Prepare the raw in silico pc-bSSFP data for a given parameter distribution

Author: florian.birk@tuebingen.mpg.de (Florian Birk), February 2024
"""
import torch
import numpy as np
import yaml
import argparse
import paths
from model import Qbssfp
from fcns_mb import *


def get_frac_not_used(param_min, param_max):
    """Calculate the fraction of a 2D sampling with min and max from t1_range and t2_range that is not used for simulation.
    Condition: t1 >= t2
    """
    t1_min = param_min[0]
    t1_max = param_max[0]
    t2_min = param_min[1]
    t2_max = param_max[1]
    t1_range = t1_max - t1_min
    t2_range = t2_max - t2_min
    t2_tri_length = (t2_max - t1_min) / t2_range
    t1_tri_length = 1 - ((t1_max - t2_max) / t1_range)
    frac_not_used = (t1_tri_length * t2_tri_length) / 2
    return frac_not_used


if __name__ == '__main__':
    torch.manual_seed(2311)
    # Parse the config file
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, default='uniform')
    parser.add_argument('--b0', '-b', type=bool, default=True,
                        help='B0 in simulated data or not')
    args = parser.parse_args()
    path_density = paths.data / 'in_vivo' / 'density_map'
    path_in_silico = paths.data / 'in_silico' / 'train'
    path_density.mkdir(parents=True, exist_ok=True)
    path_in_silico.mkdir(parents=True, exist_ok=True)

    cfg = yaml.safe_load(
        open(paths.cfgs / '0.simulation' / f'cfg_{args.config}.yml'))
    cfg['b0fit'] = args.b0
    cfg['realimag'] = False
    if cfg['distr'] == 'in_vivo':
        t1t2_density = torch.from_numpy(
            np.load(path_density / f't1t2_density.npy'))
        t1_edges = torch.from_numpy(
            np.load(path_density / f't1_edges.npy'))
        t2_edges = torch.from_numpy(
            np.load(path_density / f't2_edges.npy'))
    # %% Simulate data
    model = Qbssfp(cfg)
    nsamples_final = 400000

    # create target parameters
    if cfg['distr'] == 'in_vivo':
        par = torch.zeros((0, 4))
        t1t2_hist = torch.round(t1t2_density * cfg['nsamples']).to(torch.int)
        shape = t1t2_density.shape
        for i in range(shape[0]):
            for j in range(shape[1]):
                if t1t2_hist[i, j] > 0:
                    par_temp = torch.zeros((t1t2_hist[i, j], 4))
                    par_temp[:, 0] = torch.rand(t1t2_hist[i, j]) * \
                        (t1_edges[i+1] - t1_edges[i]) + t1_edges[i]
                    par_temp[:, 1] = torch.rand(t1t2_hist[i, j]) * \
                        (t2_edges[j+1] - t2_edges[j]) + t2_edges[j]
                    par = torch.vstack((par, par_temp))
                else:
                    continue
    else:
        if cfg['param_min'][0] < cfg['param_max'][1]:  # if T1 min is smaller than T2 max
            # get fraction that will be discarded (t1 < t2) later on. This fraction needs to be added to the number of samples
            frac_nu = get_frac_not_used(cfg['param_min'], cfg['param_max'])
            # nsamples to be simulated
            nsamples = int(cfg['nsamples']/(1-frac_nu))
        else:
            nsamples = cfg['nsamples']
        # create target parameters
        par = torch.zeros((nsamples, 4))  # , dtype=torch.float64)
        # simulate T1 and T2 parameters
        for pp in range(2):
            par[:, pp] = model.param_min[pp] + \
                (model.param_max[pp] - model.param_min[pp]) * \
                torch.rand(par.shape[0])

    # For each distribution the same
    print(f'Shape of par (before removal): {par.shape}')
    # remove all rows where T1 is smaller than T2
    par = par[torch.greater_equal(par[:, 0], par[:, 1]), :]
    print(f'Shape of par (after removal): {par.shape}')
    # simulate b1 and b0 parameters after removal ot T1 < T2
    for pp in range(2, 4):
        par[:, pp] = model.param_min[pp] + \
            (model.param_max[pp] - model.param_min[pp]) * \
            torch.rand(par.shape[0])
    if args.b0:
        # multiply the b0 parameters with torch.pi and round to nearest integer
        par[:, 2] = par[:, 2] * torch.pi
    else:
        par[:, 2] = 0
    # check if t1 is always greater than t2
    if torch.all(torch.greater_equal(par[:, 0], par[:, 1])):
        print('All T1 values are greater than T2 values. Good!')

    # shuffle the parameters
    idx = torch.randperm(par.shape[0])
    par = par[idx, :]

    # take only cfg['nsamples'] samples
    par = par[:nsamples_final, :]

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

    # %% save the data
    if cfg['npc'] == 12:
        npc_name = '12pc'
    elif cfg['npc'] == 6:
        npc_name = '6pc'
    elif cfg['npc'] == 4:
        npc_name = '4pc'
    if args.b0:
        torch.save(par, path_in_silico /
                   f't1t2b0b1_{args.config}_{npc_name}.pt')
        torch.save(sig, path_in_silico /
                   f'bssfp_complex_{args.config}_{npc_name}.pt')
    else:
        torch.save(par, path_in_silico /
                   f't1t2b0b1_nob0_{args.config}_{npc_name}.pt')
        torch.save(sig, path_in_silico /
                   f'bssfp_complex_nob0_{args.config}_{npc_name}.pt')
