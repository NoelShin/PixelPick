from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize

class MNISTDataset:
    def __init__(self, args, val=False):
        transforms = Compose([ToTensor(), Normalize(mean=[0.5], std=[0.5])])
        self.dataset = MNIST(args.dir_datasets, download=True, train=not val, transform=transforms)
        print("MNIST sample size ({:s}): {:d}".format("val" if val else "train", len(self.dataset)))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, ind):
        x, y = self.dataset[ind]
        dict_data = {'x': x, 'y': y}
        return dict_data