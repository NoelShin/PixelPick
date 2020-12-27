import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.cifar10 import CIFAR10Dataset
from datasets.cifar100 import CIFAR100Dataset
from datasets.mnist import MNISTDataset
from criterions.gcpl import Criterion
from utils import AverageMeter

class Validator:
    def __init__(self, args):
        if args.dataset_name == "MNIST":
            dataset = MNISTDataset(args, val=True)
        elif args.dataset_name == "CIFAR10":
            dataset = CIFAR10Dataset(args, val=True)
        elif args.dataset_name == "CIFAR100":
            dataset = CIFAR100Dataset(args, val=True)

        self.dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.n_workers)
        self.device = torch.device("cuda:{:s}".format(args.gpu_ids) if torch.cuda.is_available() else "cpu:0")
        self.distance = Criterion.distance

    def __call__(self, model, prototypes):
        model.eval()

        acc = AverageMeter()
        dataloader_iter = iter(self.dataloader)
        tbar = tqdm(range(len(self.dataloader)))
        fmt = "acc. {:.3f}"
        with torch.no_grad():
            for _ in tbar:
                dict_data = next(dataloader_iter)
                x, y = dict_data['x'].to(self.device), dict_data['y'].to(self.device)
                dict_outputs = model(x)

                pred = self.distance(dict_outputs["emb"], prototypes).argmin(dim=1)

                acc.update(((pred == y).sum() / x.shape[0]).cpu())
                tbar.set_description(fmt.format(acc.avg))

        print('\n' + '=' * 100)
        print("acc.: {:.3f}".format(acc.avg))
        print('=' * 100 + '\n')