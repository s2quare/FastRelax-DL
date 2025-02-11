"""Additional functions used during or after model training

Author: florian.birk@tuebingen.mpg.de (Florian Birk), February 2024
"""
# %% import
import torch
from torch import Tensor

##########################################################
# TRAINING FUNCTIONS
##########################################################


# Estimator of tissue parameters from given MRI measurements

def getparams(x: Tensor, param_min, param_max, device) -> Tensor:
    """Get tissue parameters from an NN output
    """
    # Map normalised neuronal activations to MRI tissue parameter ranges
    # Mini-batch from multiple voxels
    if x.dim() == 2:

        t_allones = torch.ones((x.shape[0], 1)).to(device)

        if len(param_min) == 2:

            max_val = torch.concat(
                (param_max[0]*t_allones, param_max[1]*t_allones), dim=1)

            min_val = torch.concat(
                (param_min[0]*t_allones, param_min[1]*t_allones), dim=1)

        elif len(param_min) == 3:

            max_val = torch.concat(
                (param_max[0]*t_allones, param_max[1]*t_allones, param_max[2]*t_allones), dim=1)

            min_val = torch.concat(
                (param_min[0]*t_allones, param_min[1]*t_allones, param_min[2]*t_allones), dim=1)

    return (max_val - min_val)*x + min_val


def sim_bssfp(phi_nom_hz, tr, te, M0, fa_nom, b1, t1, t2, b0) -> Tensor:
    """Simulation of bSSFP profile in the steady-state. 
    *phase vector:                          self.phi_pc
    *Repetition time (TR) [ms]:             self.TR
    *Proton density (M0) :                  self.M0
    *Nominal FA [degree]:                   self.FA_nom
    *B1+ scaling factor [a.u.]:             self.b1
    *Spin-Gitter relaxation time (T1) [ms]: x[:,0]
    *Spin-Spin relaxation time (T2) [ms]:   x[:,1]
    *delta B0 [rad]:                         0


    Args:
        x (Tensor): 2D tensor of NN parameter estimates (n_vox, n_tar_features)
        b1 (Tensor): 1D tensor of B1+ scaling factors (n_vox)

    Returns:
        Tensor: 2D tensor of complex bSSFP signal (n_vox, n_pc)
    """
    # convert to radians
    off_resonance_rad = b0
    phi_nom_rad = (2 * torch.pi * phi_nom_hz *
                   tr * 1e-3).unsqueeze(-1)
    flip_rad = fa_nom * torch.pi / 180
    # get phi_act
    phi_act_rad = off_resonance_rad - phi_nom_rad
    # Actual flip angle
    flip_rad_act = flip_rad * b1
    M0 = 1
    E1 = torch.exp(-tr/t1)
    E2 = torch.exp(-tr/t2)
    C = E2*(E1-1)*(1+torch.cos(flip_rad_act))
    D = (1-E1*torch.cos(flip_rad_act))-(E1-torch.cos(flip_rad_act))*(E2**2)
    bssfp_complex = (1-E1)*torch.sin(flip_rad_act)*(1-E2 *
                                                    torch.exp(-1j*(phi_act_rad)))/(C*torch.cos(phi_act_rad)+D)
    bssfp_sig = M0*bssfp_complex*torch.exp(-te/t2)
    return torch.transpose(bssfp_sig * torch.exp(1j*off_resonance_rad*(te/tr)), 0, 1)


def phase_correction_2d(bssfp_complex: Tensor):
    """Perform phase correction on 2D complex bSSFP data along the second axis (phase-cycles).

    Args:
        bssfp_complex (Tensor): Complex bSSFP data (n_vox, n_pc)
    """
    phase_sum = torch.angle(torch.sum(bssfp_complex, dim=1))
    bssfp_complex_corr = torch.zeros_like(bssfp_complex)
    for i in range(bssfp_complex.shape[1]):
        bssfp_complex_corr[:, i] = bssfp_complex[:, i] * \
            torch.exp(-1j*phase_sum)
    return bssfp_complex_corr


def calc_Fn(x: Tensor, phi_pc: Tensor, Np: int, device='cpu'):
    """Calculate the pth modes from pc-bSSFP data. See Matlab function from Rahel Heule.

    Args:
        x (Tensor): pc-bSSFP data
        phi_pc (self.phi_pc): phase cycle vector
        Np (int): Highest mode number to be calculated

    Returns:
        G(p)/Fn
    """
    shape = x.shape

    if len(shape) == 1:
        Fn = torch.zeros(Np*2+1, dtype=torch.cfloat).to(device)
    elif len(shape) == 2:
        Fn = torch.zeros((shape[0], Np*2+1), dtype=torch.cfloat).to(device)
    elif len(shape) == 3:
        Fn = torch.zeros((shape[0], shape[1], Np*2+1),
                         dtype=torch.cfloat).to(device)
    elif len(shape) == 4:
        Fn = torch.zeros(
            (shape[0], shape[1], shape[2], Np*2+1), dtype=torch.cfloat).to(device)
    # complex_tensor = torch.tensor(0+1j,dtype=torch.cdouble)

    for p in range(-Np, Np+1, 1):
        sum_b = 0
        # sum_b = torch.zeros(shape[0:-1],dtype=torch.complex64)
        for j in range(shape[-1]):
            # BE AWARE OF THE + AND - SIGN!
            if len(shape) == 4:
                sum_b += x[:, :, :, j] * \
                    torch.exp(1j*phi_pc[j]*p)
            if len(shape) == 3:
                sum_b += x[:, :, j] * \
                    torch.exp(1j*phi_pc[j]*p)
            if len(shape) == 2:
                sum_b += x[:, j]*torch.exp(1j*phi_pc[j]*p)
            if len(shape) == 1:
                sum_b += x[j]*torch.exp(1j*phi_pc[j]*p)
        if len(shape) == 4:
            Fn[:, :, :, p+Np] = sum_b / shape[-1]
        elif len(shape) == 3:
            Fn[:, :, p+Np] = sum_b / shape[-1]
        elif len(shape) == 2:
            Fn[:, p+Np] = sum_b / shape[-1]
        elif len(shape) == 1:
            Fn[p+Np] = sum_b / shape[-1]

    return Fn


def norm2_vox(x: Tensor):
    """Normalize input data along axis 1. Use torch.linal.norm, order=2: Euclidean distance
    (https://en.wikipedia.org/wiki/Norm_(mathematics)#p-norm, https://math.stackexchange.com/questions/285398/what-is-the-norm-of-a-complex-number)

    Args:
        x (Tensor): Complex tensor (torch.complex64)

    Returns:
        Tensor: Normalized tensor along axis 1
    """
    if x.dim() == 1:
        norm2 = torch.linalg.norm(x, ord=2)
        return x / norm2
    else:
        norm2 = torch.linalg.norm(x, ord=2, dim=1)
        norm2_array = torch.transpose(norm2.repeat(x.shape[1], 1), 0, 1)
        return x / norm2_array


def return_without_f0_imag_2d(x: Tensor, Np: int):
    return torch.cat((torch.imag(x[:, :Np]*torch.exp(torch.tensor(1j*torch.pi))), torch.imag(x[:, Np+1:])), dim=1)


def mod_fn_train(x: Tensor, Np: int, b0: Tensor):
    """Modify the Fn phase during training. Additional conditions to remove the phase wraps in F-1 and F1 are not included.
    Only validated for three lowest configuration orders (F-1, F0, F1)"""
    # x is the phase of the fn modes
    # split the phase into Fmn, F0 and Fn
    x = torch.angle(x)
    fmn_phase = x[:, :Np].squeeze()
    # f0_phase = x[:, Np].unsqueeze(-1)
    fpn_phase = x[:, Np+1:].squeeze()
    for n in range(2):
        # if b0 > 0
        fmn_phase = torch.where((b0 >= n*2*torch.pi+torch.pi) & (
            b0 < n*2*torch.pi+3*torch.pi), fmn_phase - (n+1)*2*torch.pi, fmn_phase)
        fpn_phase = torch.where((b0 >= n*2*torch.pi+torch.pi) & (
            b0 < n*2*torch.pi+3*torch.pi), fpn_phase + (n+1)*2*torch.pi, fpn_phase)
        # if b0 < 0
        fmn_phase = torch.where((b0 <= -n*2*torch.pi-torch.pi) & (
            b0 > -n*2*torch.pi-3*torch.pi), fmn_phase + (n+1)*2*torch.pi, fmn_phase)
        fpn_phase = torch.where((b0 <= -n*2*torch.pi-torch.pi) & (
            b0 > -n*2*torch.pi-3*torch.pi), fpn_phase - (n+1)*2*torch.pi, fpn_phase)
    return torch.cat((fmn_phase.unsqueeze(-1), fpn_phase.unsqueeze(-1)), 1)
