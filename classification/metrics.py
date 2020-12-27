import numpy as np
import torch

class Gaussian:
    def __init__(self, mean, std, eps=1e-5):
        self.mean = mean
        self.std = std
        self.eps = eps

    def __call__(self, x):
        # coef = 1. / (self.std * np.sqrt(2. * np.pi) + self.eps)
        return 1 * np.exp(-(((x - self.mean) / self.std) ** 2) / 2)


def compute_confidence(distance_matrix):  #, radii_matrix, mean=0.):
    return torch.exp(-distance_matrix)  # -(distance_matrix ** 2))
    # return 1 * np.exp(-(((distance_matrix - mean) / radii_matrix) ** 2) / 2)

def compute_acc(emb, labels, prototypes, radii):
    b, n_classes, n_emb_dims = emb.shape[0], *prototypes.shape

    # print(emb.shape, prototypes.shape, radii.shape, labels.shape)
    emb = emb.unsqueeze(dim=1).repeat((1, n_classes, 1))  # b x n_classes x n_emb_dims
    prototypes = prototypes.unsqueeze(dim=0).repeat((b, 1, 1))  # b x n_classes x n_emb_dims
    radii = radii.unsqueeze(dim=0).repeat((b, 1, 1))  # b x n_classes x n_emb_dims

    assert emb.shape == prototypes.shape

    distance_matrix = (emb - prototypes).abs()  # b x n_classes x n_emb_dims
    confidence = compute_confidence(distance_matrix)  #, radii)  # b x n_classes x n_emb_dims
    confidence_per_class = confidence.mean(dim=2, keepdim=False)  # b x n_classes
    pred = confidence_per_class.argmax(dim=1)

    acc = (pred == labels).sum() / b
    return acc

def distance(emb, prototypes):
    # emb: b x 2, prototypes: n_classes x 2
    emb_sq = emb.pow(exponent=2).sum(dim=1, keepdim=True)  # b x 1
    prototypes_sq = prototypes.pow(exponent=2).sum(dim=1, keepdim=True)  # n_classes x 1
    dist = emb_sq - 2 * torch.matmul(emb, prototypes.t()) + prototypes_sq.t()  # b x n_classes
    return dist