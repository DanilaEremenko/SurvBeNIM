import numpy as np
import torch


def mse_torch(y_true, y_pred, t_deltas, sample_weight, v, b):
    y_true = y_true[:, 1:]
    y_pred = y_pred[:, 1:]
    v = v[:, 1:]
    assert y_true.shape == y_pred.shape == t_deltas.shape == v.shape
    assert len(sample_weight) == len(y_true)
    return ((torch.abs(v) * t_deltas * (y_true - y_pred) ** 2) * sample_weight[:, None]).mean()


def domain_mse_np(y_true, y_pred, t_deltas, sample_weight, v, b):
    y_true = y_true[:, 1:]
    y_pred = y_pred[:, 1:]
    v = v[:, 1:]
    assert y_true.shape == y_pred.shape == t_deltas.shape == v.shape
    assert len(sample_weight) == len(y_true)
    return ((np.abs(v) * t_deltas * (y_true - y_pred) ** 2) * sample_weight[:, None]).mean()
