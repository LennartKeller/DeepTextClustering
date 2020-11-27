import torch


def lp_distance(X, Y, p=2):
    """
    Computes row wise minkowski distances between matrices X and Y
    """
    return torch.sum(torch.abs(X - Y) ** p, dim=1) ** (1 / p)
