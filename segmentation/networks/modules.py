import torch
import torch.nn as nn


# exponential moving average
class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


def init_prototypes(n_classes, n_emb_dims, n_prototypes=1, learnable=False, device=None):
    return torch.zeros((n_classes, n_prototypes, n_emb_dims), dtype=torch.float32, requires_grad=learnable, device=device)


def init_radii(n_classes, n_emb_dims):
    return torch.zeros((n_classes, n_emb_dims), dtype=torch.float32)


def init_weights(m):
    if isinstance(m, nn.Conv2d):
        kernel_size = torch.tensor(m.weight.shape)
        scale = torch.sqrt(6. / (torch.prod(kernel_size[1:]) + torch.prod(kernel_size[2:]) * kernel_size[-1]))
        torch.nn.init.uniform_(m.weight, -scale, scale)
        nn.init.constant_(m.bias, 0)

    elif isinstance(m, nn.Linear):
        kernel_size = torch.tensor(m.weight.shape)
        scale = torch.sqrt(6. / (kernel_size.sum()))
        torch.nn.init.uniform_(m.weight, -scale, scale)
        nn.init.constant_(m.bias, 0)

    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)