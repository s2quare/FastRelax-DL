"""Functions used for data preparation.

Author: florian.birk@tuebingen.mpg.de, February 2024
"""
import torch
import numpy as np


def apply_transform(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor, device):
    mean_full = torch.tile(mean, (x.shape[0], 1)).to(device)
    std_full = torch.tile(std, (x.shape[0], 1)).to(device)
    return (x - mean_full) / std_full


def revert_z_transform(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor):
    mean_full = torch.tile(mean, (x.shape[0], 1))
    std_full = torch.tile(std, (x.shape[0], 1))
    return x * std_full + mean_full


def min_max_transform(x: torch.Tensor, min: list, max: list):
    min = torch.tensor(min)
    max = torch.tensor(max)
    min_full = torch.tile(min, (x.shape[0], 1))
    max_full = torch.tile(max, (x.shape[0], 1))
    return (x - min_full) / (max_full - min_full)


def revert_min_max_transform(x: torch.Tensor, min: list, max: list):
    min = torch.tensor(min)
    max = torch.tensor(max)
    min_full = torch.tile(min, (x.shape[0], 1))
    max_full = torch.tile(max, (x.shape[0], 1))
    return x * (max_full - min_full) + min_full


def train_val_test_split(x: torch.Tensor, y: torch.Tensor, split: list):
    """Split data into train, val and test set.

    :param x: Signal data
    :param y: Parameter data
    :param split: List of split ratios
    :return: x_train, x_val, x_test, y_train, y_val, y_test
    """
    # shuffle data same way
    idx = torch.randperm(x.shape[0])
    x = x[idx, :]
    y = y[idx, :]
    # split data
    x_train = x[:int(split[0]*x.shape[0]), :]
    x_val = x[int(split[0]*x.shape[0]):int(
        (split[0]+split[1])*x.shape[0]), :]
    x_test = x[int((split[0]+split[1])*x.shape[0]):, :]
    y_train = y[:int(split[0]*y.shape[0]), :]
    y_val = y[int(split[0]*y.shape[0]):int(
        (split[0]+split[1])*y.shape[0]), :]
    y_test = y[int((split[0]+split[1])*y.shape[0]):, :]
    return x_train, x_val, x_test, y_train, y_val, y_test
