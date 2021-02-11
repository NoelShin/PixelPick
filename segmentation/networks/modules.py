import torch
import torch.nn as nn
from tqdm import tqdm

# exponential moving average
class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


def init_prototypes(n_classes, n_emb_dims,
                    n_prototypes=1,
                    mode='zero',
                    model=None,
                    dataset=None,
                    learnable=False,
                    device=None,
                    ignore_index=None):
    if mode == 'zero':
        return torch.zeros((n_classes, n_prototypes, n_emb_dims),
                           dtype=torch.float32, requires_grad=learnable, device=device)
    elif mode == 'mean':
        assert any((model, dataset, ignore_index)) is not None, f"For arg type='mean', model and dataset args should be given."
        print("Initialising prototypes...")
        prototypes = torch.zeros((n_classes, n_prototypes, n_emb_dims),
                                 dtype=torch.float32, requires_grad=learnable, device=device)
        from torch.utils.data import DataLoader
        dataloader = DataLoader(dataset,
                                batch_size=5,
                                num_workers=4,
                                shuffle=True,
                                drop_last=True)

        model.eval()
        with torch.no_grad():
            dict_label_counts = {i: 0 for i in range(n_classes)}
            for dict_data in tqdm(dataloader):
                x, y = dict_data['x'].to(device), dict_data['y'].to(device)
                dict_outputs = model(x)

                emb = dict_outputs['emb_']  # b x n_emb_dims x h x w

                b, c, h, w = emb.shape
                emb_flatten = emb.transpose(1, 0)  # c x b x h x w
                emb_flatten = emb_flatten.contiguous().view(c, b * h * w)  # c x (b * h * w)
                emb_flatten = emb_flatten.transpose(1, 0)  # (b * h * w) x c
                y_flatten = y.flatten()  # (b * h * w)

                unique_labels_batch = set(sorted(y_flatten.cpu().numpy().tolist())) - {ignore_index}

                for label in unique_labels_batch:
                    ind_label = (y_flatten == label)  # (b * h * w)
                    emb_label = emb_flatten[ind_label]  # m x n_emb_dims
                    prototypes[label] += emb_label.sum(dim=0, keepdim=True).detach()
                    dict_label_counts[label] = dict_label_counts[label] + emb_label.shape[0]

            for label, counts in dict_label_counts.items():
                prototypes[label] /= counts
        model.train()
        return prototypes


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