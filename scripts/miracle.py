"""Functions used to perform MIRACLE (Motion Insensitive RApid Configuration reLaxomEtry)


Author: 
- florian.birk@tuebingen.mpg.de (Florian Birk), February 2024
"""

import numpy as np
from numba import jit


@jit(nopython=False, error_model='numpy')
def fgss_T1(fid: np.ndarray, fid2: np.ndarray, tr: float, te1: float, te2: float, flip: float, t2: float, a: float, b: float, eps: float, N: int):
    """performs golden section search to find T1 minimum of a unimodal function
    Original Matlab file (fgss_T1.m) WRITTEN by RAHEL HEULE (rahel.heule@tuebingen.mpg.de)

    Args:
        fid (np.ndarray): F0 (FISP) signal, voxel-wise
        fid2 (np.ndarray): F1 signal, voxel-wise
        tr (float): repetition time [ms]
        te1 (float): echo time of F1 [ms]
        te2 (float): echo time of F0 [ms]
        te3 (float): echo time of F-1 [ms]
        flip (float): Flip angle [rad]
        t1 (float): t1 (longitudinal/spin-gitter relaxation) estimate [ms]
        a (float): lower boundary of start interval
        b (float): upper boundary of start interval
        eps (float): convergence limit
        N (int): maximal number of function evaluations

    Returns:
        float: x_min
    """
    e2 = np.exp(-tr/t2)

    c = (-1+np.sqrt(5))/2
    x1 = c*a + (1-c)*b

    # calculate fx1=f(x1)
    e1 = np.exp(-tr/x1)
    p = 1-e1*np.cos(flip)-(e2**2)*(e1-np.cos(flip))
    q = e2*(1-e1)*(1+np.cos(flip))
    F0 = np.tan(flip/2)*(1-(e1-np.cos(flip))*(1-e2**2)/np.sqrt(p**2-q**2))
    F1 = np.tan(flip/2)*(1-(e1-np.cos(flip))*(1-e2**2) /
                         np.sqrt(p**2-q**2))*(1/q)*(p-np.sqrt(p**2-q**2))

    fx1 = np.abs(fid2/fid-F1/F0*np.exp((te2-te1)/t2))
    # ------

    x2 = (1-c)*a + c*b

    # calculate fx2=f(x2)
    e1 = np.exp(-tr/x2)
    p = 1-e1*np.cos(flip)-(e2**2)*(e1-np.cos(flip))
    q = e2*(1-e1)*(1+np.cos(flip))
    F0 = np.tan(flip/2)*(1-(e1-np.cos(flip))*(1-e2**2)/np.sqrt(p**2-q**2))
    F1 = np.tan(flip/2)*(1-(e1-np.cos(flip))*(1-e2**2) /
                         np.sqrt(p**2-q**2))*(1/q)*(p-np.sqrt(p**2-q**2))

    fx2 = abs(fid2/fid-F1/F0*np.exp((te2-te1)/t2))

    # calculate x_min updates in for loop
    for i in range(N-2):
        if fx1 < fx2:
            b = x2
            x2 = x1
            fx2 = fx1
            x1 = c*a + (1-c)*b

            # calculate fx1=f(x1)
            e1 = np.exp(-tr/x1)
            p = 1-e1*np.cos(flip)-(e2**2)*(e1-np.cos(flip))
            q = e2*(1-e1)*(1+np.cos(flip))
            F0 = np.tan(flip/2)*(1-(e1-np.cos(flip))
                                 * (1-e2**2)/np.sqrt(p**2-q**2))
            F1 = np.tan(flip/2)*(1-(e1-np.cos(flip))*(1-e2**2) /
                                 np.sqrt(p**2-q**2))*(1/q)*(p-np.sqrt(p**2-q**2))

            fx1 = np.abs(fid2/fid-F1/F0*np.exp((te2-te1)/t2))
            # ------

        else:
            a = x1
            x1 = x2
            fx1 = fx2
            x2 = (1-c)*a + c*b

            # calculate fx2=f(x2)
            e1 = np.exp(-tr/x2)
            p = 1-e1*np.cos(flip)-(e2**2)*(e1-np.cos(flip))
            q = e2*(1-e1)*(1+np.cos(flip))
            F0 = np.tan(flip/2)*(1-(e1-np.cos(flip))
                                 * (1-e2**2)/np.sqrt(p**2-q**2))
            F1 = np.tan(flip/2)*(1-(e1-np.cos(flip))*(1-e2**2) /
                                 np.sqrt(p**2-q**2))*(1/q)*(p-np.sqrt(p**2-q**2))

            fx2 = abs(fid2/fid-F1/F0*np.exp((te2-te1)/t2))
            # ------

        if (np.abs(b-a) < eps):
            # print(f'succeeded after {i+1} steps\n')
            if fx1 < fx2:
                x_min = x1
                # print('%.4f' % x_min)
            else:
                x_min = x2
                # print('%.4f' % x_min)
            return x_min
    print(f'T1: failed requirements after {N} steps\n')

    if fx1 < fx2:
        x_min = x1
        # print('%.4f' % x_min)
    else:
        x_min = x2
        # print('%.4f' % x_min)
    return x_min


@jit(nopython=False, error_model='numpy')
def fgss_T2_2(fid: np.ndarray, echo: np.ndarray, fid2: np.ndarray, tr: float, te1: float, te2: float, te3: float, flip: float, t1: float, a: float, b: float, eps: float, N: int):
    """performs golden section search to find T2 minimum of a unimodal function
    Original Matlab file (fgss_T2_2.m) WRITTEN by RAHEL HEULE (rahel.heule@tuebingen.mpg.de)

    Args:
        fid (np.ndarray): F0 (FISP) signal, voxel-wise
        echo (np.ndarray): F-1 (PSIF) signal, voxel-wise
        fid2 (np.ndarray): F1 signal, voxel-wise
        tr (float): repetition time [ms]
        te1 (float): echo time of F1 [ms]
        te2 (float): echo time of F0 [ms]
        te3 (float): echo time of F-1 [ms]
        flip (float): Flip angle [rad]
        t1 (float): t1 (longitudinal/spin-gitter relaxation) estimate [ms]
        a (float): lower boundary of start interval
        b (float): upper boundary of start interval
        eps (float): convergence limit
        N (int): maximal number of function evaluations

    Returns:
        float: x_min
    """
    e1 = np.exp(-tr/t1)
    c = (-1 + np.sqrt(5))/2
    x1 = c*a + (1-c)*b

    # calculate fx1=f(x1)
    e2 = np.exp(-tr/x1)
    p = 1-e1*np.cos(flip)-(e2**2)*(e1-np.cos(flip))
    q = e2*(1-e1)*(1+np.cos(flip))
    F1 = (1-(e1-np.cos(flip))*(1-e2**2)/np.sqrt(p**2-q**2)) * \
        (1/q)*(p-np.sqrt(p**2-q**2))
    F0 = 1-(e1-np.cos(flip))*(1-e2**2)/np.sqrt(p**2-q**2)
    Fm1 = 1-(1-e1*np.cos(flip))*(1-e2**2)/np.sqrt(p**2-q**2)

    fx1 = np.abs(echo/(fid-fid2)-Fm1*np.exp((tr-te3)/x1) /
                 (F0*np.exp(-te2/x1)-F1*np.exp(-te1/x1)))

    x2 = (1-c)*a + c*b

    # calculate fx2=f(x2)
    e2 = np.exp(-tr/x2)
    p = 1-e1*np.cos(flip)-(e2**2)*(e1-np.cos(flip))
    q = e2*(1-e1)*(1+np.cos(flip))
    F1 = (1-(e1-np.cos(flip))*(1-e2**2)/np.sqrt(p**2-q**2)) * \
        (1/q)*(p-np.sqrt(p**2-q**2))
    F0 = 1-(e1-np.cos(flip))*(1-e2**2)/np.sqrt(p**2-q**2)
    Fm1 = 1-(1-e1*np.cos(flip))*(1-e2**2)/np.sqrt(p**2-q**2)

    fx2 = abs(echo/(fid-fid2)-Fm1*np.exp((tr-te3)/x2) /
              (F0*np.exp(-te2/x2)-F1*np.exp(-te1/x2)))

    # calculate x_min updates in for loop
    for i in range(N-2):
        if fx1 < fx2:
            b = x2
            x2 = x1
            fx2 = fx1
            x1 = c*a + (1-c)*b

            # calculate fx1=f(x1)
            e2 = np.exp(-tr/x1)
            p = 1-e1*np.cos(flip)-(e2**2)*(e1-np.cos(flip))
            q = e2*(1-e1)*(1+np.cos(flip))
            F1 = (1-(e1-np.cos(flip))*(1-e2**2)/np.sqrt(p**2-q**2)) * \
                (1/q)*(p-np.sqrt(p**2-q**2))
            F0 = (1-(e1-np.cos(flip))*(1-e2**2)/np.sqrt(p**2-q**2))
            Fm1 = (1-(1-e1*np.cos(flip))*(1-e2**2)/np.sqrt(p**2-q**2))

            fx1 = abs(echo/(fid-fid2)-Fm1*np.exp((tr-te3)/x1) /
                      (F0*np.exp(-te2/x1)-F1*np.exp(-te1/x1)))
            # ------

        else:
            a = x1
            x1 = x2
            fx1 = fx2
            x2 = (1-c)*a + c*b

            # calculate fx2=f(x2)
            e2 = np.exp(-tr/x2)
            p = 1-e1*np.cos(flip)-(e2**2)*(e1-np.cos(flip))
            q = e2*(1-e1)*(1+np.cos(flip))
            F1 = (1-(e1-np.cos(flip))*(1-e2**2)/np.sqrt(p**2-q**2)) * \
                (1/q)*(p-np.sqrt(p**2-q**2))
            F0 = 1-(e1-np.cos(flip))*(1-e2**2)/np.sqrt(p**2-q**2)
            Fm1 = 1-(1-e1*np.cos(flip))*(1-e2**2)/np.sqrt(p**2-q**2)

            fx2 = abs(echo/(fid-fid2)-Fm1*np.exp((tr-te3)/x2) /
                      (F0*np.exp(-te2/x2)-F1*np.exp(-te1/x2)))
            # ------

        if (np.abs(b-a) < eps):
            # print(f'succeeded after {i+1} steps\n')
            if fx1 < fx2:
                x_min = x1
                # print('%.4f' % x_min)
            else:
                x_min = x2
                # print('%.4f' % x_min)
            return x_min
    print(f'T2: failed requirements after {N} steps\n')

    if fx1 < fx2:
        x_min = x1
        # print('%.4f' % x_min)
    else:
        x_min = x2
        # print('%.4f' % x_min)
    return x_min


@jit(nopython=False, error_model='numpy')
def calc_t1_t2_ima_gss_B1_reg(f1: np.ndarray, f0: np.ndarray, fm1: np.ndarray, b1: np.ndarray, tr: float, te: float, flipangle: int, t1_est: int, mask: np.ndarray):
    """Calculate the t1 and t2 estimates using golden section search and ratios over the three highest order modes.
    Original Matlab file (calc_t1_t2_imag_gss_B1_reg.m) WRITTEN by RAHEL HEULE (rahel.heule@tuebingen.mpg.de)

    Args:
        f1 (np.ndarray): signals/3D image stacks
        f0 (np.ndarray): signals/3D image stacks
        fm1 (np.ndarray): signals/3D image stacks
        b1 (np.ndarray): b1 scaling factor (actual flip angle/nominal flip angle), should be of same dimension as fn modes
        TR (float): repetition time [ms]
        TE (float): echo time [ms]
        flipangle (int): flip angle [degree]
        t1_est (int): t1 (longitudinal relaxation time) estimate [ms]
        mask (np.ndarray): mask [bool]

    Returns:
        list: list containing t1 and t2 estimates from MIRACLE + array of iterations steps
    """
    te1 = te
    te2 = te
    te3 = te
    fa_rad = flipangle/180 * np.pi

    # Golden section search and iteration parameters
    # The parameters eps, N, eps_iter_t1, eps_iter_t2, and iter_max control the precision of the TESS T1 and T2 calculation.
    a = 0
    b_t1 = 5000
    b_t2 = 5000
    eps = 0.001  # 0.00001
    N = 500  # 2000
    eps_iter_t1 = 0.001
    eps_iter_t2 = 0.001
    iter_max = 200  # 500

    # print(f'eps: {eps}, N:{N}, iter_max:{iter_max}')
    # Size of data
    [m, n, nsl] = f1.shape

    # Initialization of maps
    t1 = np.zeros((m, n, nsl))
    t2 = np.zeros((m, n, nsl))
    iter_map = np.zeros((m, n, nsl))
    t1_est = np.full((m, n, nsl, iter_max), t1_est)
    t2_est = np.zeros((m, n, nsl, iter_max))

    # loop over number of slices
    for k in range(nsl):
        print(f'slice number {k+1}')

        for i in range(m):
            for j in range(n):
                # thresholding ...
                if mask[i, j, k] == True:

                    iter = 0

                    # calculation of T2 from Fm1/F0 using golden section search

                    t2_est[i, j, k, iter] = fgss_T2_2(
                        f0[i, j, k], fm1[i, j, k], f1[i, j, k], tr, te1, te2, te3, fa_rad*b1[i, j, k], t1_est[i, j, k, iter], a, b_t2, eps, N)

                    # calculation of T1 from F1/F0 using golden section search

                    t1_est[i, j, k, iter] = fgss_T1(
                        f0[i, j, k], f1[i, j, k], tr, te1, te2, fa_rad*b1[i, j, k], t2_est[i, j, k, iter], a, b_t1, eps, N)

                    # first error is just the t1 or t2 estimates
                    err_t1 = t1_est[i, j, k, iter]
                    err_t2 = t2_est[i, j, k, iter]

                    while (((err_t1 > eps_iter_t1) or (err_t2 > eps_iter_t2)) and (iter < (iter_max-1))):

                        iter += 1

                        # calculation of T2 from Fm1/F0 using golden section search
                        t2_est[i, j, k, iter] = fgss_T2_2(
                            f0[i, j, k], fm1[i, j, k], f1[i, j, k], tr, te1, te2, te3, fa_rad*b1[i, j, k], t1_est[i, j, k, iter-1], a, b_t2, eps, N)

                        # calculation of T1 from F1/F0 using golden section search
                        t1_est[i, j, k, iter] = fgss_T1(
                            f0[i, j, k], f1[i, j, k], tr, te1, te2, fa_rad*b1[i, j, k], t2_est[i, j, k, iter], a, b_t1, eps, N)

                        # calculate absolute errors
                        err_t1 = np.abs(
                            t1_est[i, j, k, iter-1] - t1_est[i, j, k, iter])
                        err_t2 = np.abs(
                            t2_est[i, j, k, iter-1] - t2_est[i, j, k, iter])

                    t1[i, j, k] = t1_est[i, j, k, iter]
                    t2[i, j, k] = t2_est[i, j, k, iter]
                    iter_map[i, j, k] = iter

                else:
                    t1[i, j, k] = 0
                    # continue

    # filter
    # t1[np.where(t1 < 0)] = 0
    # t2[np.where(t2 < 0)] = 0

    return t1, t2, iter_map
