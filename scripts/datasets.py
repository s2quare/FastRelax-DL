"""Create all torch datasets used for this project

Author: florian.birk@tuebingen.mpg.de (Florian Birk), February 2024
"""
from torch.utils.data import Dataset


class UnsupervisedDataset(Dataset):
    """Unsupervised dataset for neural network training

    Args:
        Dataset ([type]): [description]
    """

    def __init__(self, sig, b1, b0):
        self.sig = sig  # bssfp
        self.b1 = b1
        self.b0 = b0

    def __len__(self):
        return self.sig.shape[0]

    def __getitem__(self, idx):
        sig = self.sig[idx, :]
        b1 = self.b1[idx, :]
        b0 = self.b0[idx, :]
        return sig, b1, b0


class SupervisedDataset(Dataset):
    """Supervised dataset for neural network training

    Args:
        Dataset ([type]): [description]
    """

    def __init__(self, sig, par, b1, b0):
        self.sig = sig
        self.par = par
        self.b1 = b1
        self.b0 = b0

    def __len__(self):
        return self.sig.shape[0]

    def __getitem__(self, idx):
        sig = self.sig[idx, :]
        par = self.par[idx, :]
        b1 = self.b1[idx, :]
        b0 = self.b0[idx, :]
        return sig, par, b1, b0
