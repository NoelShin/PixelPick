import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, args):
        super(CNN, self).__init__()
        dataset_name = args.dataset_name

        if dataset_name == "MNIST":
            conv1 = nn.Sequential(nn.Conv2d(1, 32, 5, padding=2), nn.ReLU(True))
            conv2 = nn.Sequential(nn.Conv2d(32, 32, 5, padding=2), nn.ReLU(True))
            pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

            conv3 = nn.Sequential(nn.Conv2d(32, 64, 5, padding=2), nn.ReLU(True))
            conv4 = nn.Sequential(nn.Conv2d(64, 64, 5, padding=2), nn.ReLU(True))
            pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

            conv5 = nn.Sequential(nn.Conv2d(64, 128, 5, padding=2), nn.ReLU(True))
            conv6 = nn.Sequential(nn.Conv2d(128, 128, 5, padding=2), nn.ReLU(True))
            pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

            self.fc1 = nn.Linear(3 * 3 * 128, args.n_emb_dims)
            self.fc2 = nn.Sequential(nn.Linear(args.n_emb_dims, 10))

            self.feature_extractor = nn.Sequential(
                conv1, conv2, pool1,
                conv3, conv4, pool2,
                conv5, conv6, pool3
            )

        elif dataset_name in ["CIFAR10", "CIFAR100"]:
            # Noel - originally they didn't use BN but I just added it to stabilize training
            bn = nn.BatchNorm2d
            conv1 = nn.Sequential(nn.Conv2d(3, 96, 3, padding=1), bn(96), nn.ReLU(True))
            conv2 = nn.Sequential(nn.Conv2d(96, 96, 3, padding=1), bn(96), nn.ReLU(True))
            pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
            dropout1 = nn.Dropout2d(p=0.5)

            conv3 = nn.Sequential(nn.Conv2d(96, 192, 3, padding=1), bn(192), nn.ReLU(True))
            conv4 = nn.Sequential(nn.Conv2d(192, 192, 3, padding=1), bn(192), nn.ReLU(True))
            pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
            dropout2 = nn.Dropout2d(p=0.5)

            conv5 = nn.Sequential(nn.Conv2d(192, 192, 3, padding=1), bn(192), nn.ReLU(True))
            conv6 = nn.Sequential(nn.Conv2d(192, 192, 1, padding=0), bn(192), nn.ReLU(True))
            conv7 = nn.Sequential(nn.Conv2d(192, args.n_emb_dims, 1, padding=0), bn(args.n_emb_dims), nn.ReLU(True))
            pool3 = nn.AdaptiveAvgPool2d(output_size=(1, 1))

            self.fc = nn.Linear(args.n_emb_dims, args.n_classes)

            self.feature_extractor = nn.Sequential(
                conv1, conv2, pool1, dropout1,
                conv3, conv4, pool2, dropout2,
                conv5, conv6, conv7, pool3
            )

        else:
            raise ValueError
        
        self.apply(self._init_weights)
        self.dataset_name = args.dataset_name
        print(self)

    def forward(self, x):
        if self.dataset_name == "MNIST":
            o = self.feature_extractor(x)
            emb = self.fc1(o.view(o.shape[0], -1))
            return {"emb": emb, "pred": self.fc2(F.relu(emb))}

        elif self.dataset_name in ["CIFAR10", "CIFAR100"]:
            o = self.feature_extractor(x)
            emb = o.view(o.shape[0], -1)
            return {"emb": emb, "pred": self.fc(emb)}

    @staticmethod
    def _init_weights(m):
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

# exponential moving average
class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


def init_prototypes(device, n_classes=None, n_emb_dims=None, model=None, dataset=None, mode='zero'):
    assert mode in ["mean", "zero"], "argument mode should be one of mean or zero, got {}".format(mode)
    if mode == "zero":
        assert any([n_classes, n_emb_dims, device]) is not None
        # dict_prototypes.update(
        #     {i: torch.zeros((n_classes, n_emb_dims), dtype=torch.float32).to(device) for i in range(n_classes)}
        # )
        prototypes = torch.zeros((n_classes, n_emb_dims), dtype=torch.float32).to(device)


    # elif mode == "mean":
    #     assert any([model, dataset, device]) is not None
    #     with torch.no_grad():
    #         for dict_data in dataset:
    #             x, y = dict_data['x'].to(device), dict_data['y'].to(device)
    #             dict_outputs = model(x)
    #             emb = dict_outputs['emb']
    #
    #             for i in range(x.shape[0]):
    #                 emb[i], y[i]
    return prototypes

def init_radii(n_classes, n_emb_dims):
    return torch.zeros((n_classes, n_emb_dims), dtype=torch.float32)
