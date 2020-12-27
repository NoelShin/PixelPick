from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, ToTensor, Normalize

class CIFAR10Dataset:
    def __init__(self, args, val=False):
        transforms = Compose([ToTensor(), Normalize(mean=[0.4913997551666284, 0.48215855929893703, 0.4465309133731618],
                                                    std=[0.24703225141799082, 0.24348516474564, 0.26158783926049628])])
        self.dataset = CIFAR10(args.dir_datasets, download=True, train=not val, transform=transforms)
        print("CIFAR10 sample size ({:s}): {:d}".format("val" if val else "train", len(self.dataset)))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, ind):
        x, y = self.dataset[ind]
        dict_data = {'x': x, 'y': y}
        return dict_data