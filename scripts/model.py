"""NN models used for this projects for inverse model prediction pc-bssfp/configuration modes + B1 --> T1, T2, B0. 

- Model-based/physics-informed unsupervised model
- Supervised model

Author: florian.birk@tuebingen.mpg.de (Florian Birk), February 2024
"""
# %% import
import torch
from torch import Tensor
from torch import nn
from torchinfo import summary
from fcns_mb import *

#############################################################
# Model-based/physics-informed unsupervised model
# Esimtate the conditional mean or quantiles of target parameters (T1, T2, (+B0)) given pc-bSSFP data
#############################################################


class Qbssfp(nn.Module):
    """qbssfp: Neural network class to perform quantitative multiparametric (relaxometry and field estimate) mapping using configuration modes from phase-cycled bSSFP. 

    - Use either supervised or semi-supervised training. 
    - Use either conditional mean or quantile regression.

    Input: 
    - mag + phase (normed Fn modes)
    - b1 scaling factor

    Targets:
    - T1 [ms]
    - T2 [ms]
    """
    # initialize (class constructor)

    def __init__(self, cfg: dict):
        super(Qbssfp, self).__init__()

        # get sequence parameters
        self.fa_nom = cfg['fa_nom']
        self.tr = cfg['tr']
        self.te = cfg['te']
        self.M0 = cfg['M0']
        self.npc = cfg['npc']
        self.Np = cfg['Np']
        self.b0fit = cfg['b0fit']
        self.realimag = cfg['realimag']
        self.device = cfg['device']

        # phi_pc in hz
        self.bssfp_width = 1 / (self.tr * 1e-3)
        self.phi_nom_hz = torch.arange(
            self.bssfp_width / (2 * self.npc), self.bssfp_width, self.bssfp_width / self.npc).to(self.device)
        self.phi_nom_rad = 2 * torch.pi * self.phi_nom_hz * self.tr * 1e-3
        # parameter ranges (T1, T2, B0) + B1 for simulation
        self.param_min = cfg['param_min']
        self.param_max = cfg['param_max']

        # define the model
        self.nneurons = cfg['nneurons']
        self.nlayers = len(self.nneurons)-1  # number of layers (inp, hidden)
        self.ninp = self.nneurons[0]
        self.ntar = self.nneurons[-1]
        self.mb = cfg['mb']

        # create layers
        layers = []
        for i in range(self.nlayers):
            layers.append(nn.Linear(self.nneurons[i], self.nneurons[i+1]))
            if i < (self.nlayers - 1):
                layers.append(nn.ReLU())
            # layers.append(nn.BatchNorm1d(self.nneurons[i+1]))

        # add a sigmoid layer at the end (map 0 - 1) for supervised and semi-supervised training
        layers.append(nn.Sigmoid())

        # add layers to model
        self.model = nn.ModuleList(layers)

        # print torch summary
        summary(self.model)

    def forward(self, x: Tensor) -> Tensor:
        """Full forward pass of a qbssfp_real-Net

        Args:
            x (Tensor): Input 2D tensor with magnitude and phase of configuration modes (n_vox, Np*4+2)

        Returns:
            Tensor: Output 2D tensor with bSSFP signal magnitude and phase (n_vox, n_pc*2)
        """
        for modellayer in self.model:
            x = modellayer(x)

        return x
