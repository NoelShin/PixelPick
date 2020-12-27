import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.cifar10 import CIFAR10Dataset
from datasets.cifar100 import CIFAR100Dataset
from datasets.mnist import MNISTDataset
from criterions.gcpl import Criterion
from metrics import compute_acc
from utils import AverageMeter, EmbeddingVisualiser

class Validator:
    def __init__(self, args):
        if args.dataset_name == "MNIST":
            dataset = MNISTDataset(args, val=True)
        elif args.dataset_name == "CIFAR10":
            dataset = CIFAR10Dataset(args, val=True)
        elif args.dataset_name == "CIFAR100":
            dataset = CIFAR100Dataset(args, val=True)

        self.dataset_name = args.dataset_name
        self.dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.n_workers)
        self.device = torch.device("cuda:{:s}".format(args.gpu_ids) if torch.cuda.is_available() else "cpu:0")
        self.distance = Criterion.distance
        self.dir_checkpoints = f"{args.dir_root}/checkpoints/{args.dir_checkpoints}/val"
        self.emb_vis = EmbeddingVisualiser(args.n_classes)

    def __call__(self, model, prototypes, radii, epoch):
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
                # dict_outputs['emb'] = dict_outputs['emb'] / torch.linalg.norm(dict_outputs['emb'], ord=2, dim=1,
                #                                                               keepdim=True)

                # pred = self.distance(dict_outputs["emb"], prototypes).argmin(dim=1)
                # acc.update(((pred == y).sum() / x.shape[0]).cpu())
                acc.update(compute_acc(dict_outputs['emb'].detach().cpu(),
                                       y.cpu(),
                                       prototypes.detach().cpu(),
                                       radii.detach().cpu()))

                tbar.set_description(fmt.format(acc.avg))

                if self.dataset_name == "MNIST":
                    self.emb_vis.add(dict_outputs["emb"], y)

        print('\n' + '=' * 100)
        print("acc.: {:.3f}".format(acc.avg))
        print('=' * 100 + '\n')
        if self.dataset_name == "MNIST":
            self.emb_vis.visualise(prototypes.cpu(),
                                   use_pca=False if self.dataset_name == "MNIST" else True,
                                   fp=f"{self.dir_checkpoints}/{epoch}.png",
                                   show=False)
            self.emb_vis.reset()