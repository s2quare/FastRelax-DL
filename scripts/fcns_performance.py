"""Functions used to evaluate model performance.

Author: florian.birk@tuebingen.mpg.de, February 2024
"""
import numpy as np


def cod(gt: np.ndarray, est: np.ndarray):
    """Coefficient of determination"""

    y_mean = np.mean(gt, axis=0)
    # Residual sum of squares
    ss_res = np.sum(np.square(gt - est), axis=0)
    ss_tot = np.sum(np.square(gt - y_mean),
                    axis=0)     # Total sum of squares
    r2 = 1 - (ss_res/ss_tot)                    # Coefficient of determination
    return r2


def rmse(gt: np.ndarray, est: np.ndarray):
    """Root Mean Square Error"""
    mse = np.mean(np.square(gt - est), axis=0)
    return np.sqrt(mse)


def mean_std_conditional_mean_interval(target, pred, min, max, steps):
    """Compute the mean and std of the prediction interval for matching target parameter"""
    max = np.array(max)
    min = np.array(min)
    step_size = (max-min)/steps
    mean = np.zeros((steps, len(min)))
    std = np.zeros((steps, len(min)))
    for i, min_par in enumerate(min):
        for j in range(steps):
            idx = np.where((target[:, i] >= min_par+j*step_size[i]) & (
                target[:, i] <= min_par+(j+1)*step_size[i]))
            mean[j, i] = np.mean(pred[idx, i])
            std[j, i] = np.std(pred[idx, i])
    return mean, std
