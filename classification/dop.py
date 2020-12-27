import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.cifar10 import CIFAR10Dataset
from datasets.cifar100 import CIFAR100Dataset
from datasets.mnist import MNISTDataset
from networks import CNN, init_prototypes, init_radii, EMA
from metrics import compute_acc
from utils import get_criterion, get_optimizer, get_lr_scheduler, get_validator, AverageMeter, EmbeddingVisualiser


def main(args):
    torch.manual_seed(args.seed)

    device = torch.device("cuda:{:s}".format(args.gpu_ids) if torch.cuda.is_available() else "cpu:0")
    torch.backends.cudnn.benchmark = True

    if args.dataset_name == "MNIST":
        dataset = MNISTDataset(args)
    elif args.dataset_name == "CIFAR10":
        dataset = CIFAR10Dataset(args)
    elif args.dataset_name == "CIFAR100":
        dataset = CIFAR100Dataset(args)

    dataloader = DataLoader(dataset,
                            batch_size=args.batch_size,
                            num_workers=args.n_workers,
                            shuffle=True)

    model = CNN(args).to(device)
    prototypes = init_prototypes(device, args.n_classes, args.n_emb_dims)
    radii = init_radii(args.n_classes, args.n_emb_dims).to(device)

    prototypes_updater = EMA(args.momentum_prototypes)
    radii_updater = EMA(args.momentum_radii)

    list_params = list(model.parameters())

    criterion = get_criterion(args)
    optimizer = get_optimizer(args, params=list_params)
    lr_scheduler = get_lr_scheduler(args, optimizer=optimizer)
    validator = get_validator(args)

    if args.dataset_name == "MNIST":
        emb_vis = EmbeddingVisualiser(args.n_classes)

    loss_before, acc_before = np.inf, 0.0
    running_loss, running_acc = AverageMeter(), AverageMeter()

    cnt = 0
    from itertools import count
    for e in count(start=1):
        model.train()

        dataloader_iter = iter(dataloader)
        tbar = tqdm(range(len(dataloader)))

        for batch_ind in tbar:
            dict_data = next(dataloader_iter)
            x, y = dict_data['x'].to(device), dict_data['y'].to(device)
            dict_outputs = model(x)
            # dict_outputs['emb'] = dict_outputs['emb'] / (torch.linalg.norm(dict_outputs['emb'], ord=2, dim=1, keepdim=True) + 1e-5)
            emb = dict_outputs['emb']
            # assert not any([torch.isnan(emb).any()]), print(emb)

            for i in range(x.shape[0]):
                label = y[i].cpu().item()

                prototypes[label] = prototypes_updater.update_average(old=prototypes[label],
                                                                      new=emb[i].detach())

                radii_class = (prototypes[label] - emb[i].detach()).pow(2).sqrt()
                radii[label] = radii_updater.update_average(old=radii[label],
                                                            new=radii_class)

            dict_losses = criterion(dict_outputs, prototypes, labels=y, non_isotropic=True)

            loss = torch.tensor(0, dtype=torch.float32).to(device)

            fmt = "Epoch {:d} | Acc.: {:.3f} | Loss: {:.3f}"
            for loss_k, loss_v in dict_losses.items():
                fmt += " | {:s}: {:.3f}".format(loss_k, loss_v.detach().cpu().item())
                loss += loss_v

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss.update(loss.detach().item())

            # pred = criterion.distance(dict_outputs["emb"], prototypes).argmin(dim=1)
            # running_acc.update(((pred == y).sum() / x.shape[0]).cpu())
            running_acc.update(compute_acc(emb.detach().cpu(), y.cpu(), prototypes.detach().cpu(), radii.detach().cpu()))

            tbar.set_description(fmt.format(e, running_acc.avg, running_loss.avg))

            if args.dataset_name == "MNIST":
                emb_vis.add(dict_outputs["emb"], y)

            # emb_vis.visualise(prototypes.cpu(),
            #                   radii.cpu(),
            #                   use_pca=False if args.dataset_name == "MNIST" else True,
            #                   fp=f"{args.dir_root}/checkpoints/{args.dir_checkpoints}/{e}.png",
            #                   show=True)
            # exit(12)

            if args.debug:
                break

        if loss_before < running_loss.avg or acc_before > running_acc.avg:
            if lr_scheduler is not None:
                lr_scheduler.step()
            cnt += 1
            print(f"cnt has been changed from {cnt - 1} to {cnt}.")

        loss_before, acc_before = running_loss.avg, running_acc.avg

        running_loss.reset()
        running_acc.reset()

        if args.dataset_name == "MNIST":
            emb_vis.visualise(prototypes.cpu(),
                              radii.cpu(),
                              use_pca=False,
                              fp=f"{args.dir_root}/checkpoints/{args.dir_checkpoints}/train/{e}.png",
                              show=False)
            emb_vis.reset()

        validator(model, prototypes, radii, e)

        if cnt == args.cnt or args.debug:
            break


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser("")

    parser.add_argument("--debug", "-d", action="store_true", default=False)
    parser.add_argument("--dir_root", type=str, default="/home/gishin/Projects/DeepLearning/Oxford/open_set")
    parser.add_argument("--seed", "-s", type=int, default=0)
    parser.add_argument("--model_name", type=str, default="dop", choices=["dop"])

    # system
    parser.add_argument("--gpu_ids", type=str, nargs='+', default='0')
    parser.add_argument("--n_workers", type=int, default=4)

    # dataset
    parser.add_argument("--dataset_name", type=str, default="CIFAR10", choices=["MNIST", "CIFAR10", "CIFAR100"])
    parser.add_argument("--dir_datasets", type=str, default="/scratch/shared/beegfs/gyungin/datasets")

    # gcpl
    parser.add_argument("--loss_type", type=str, default="dce", choices=["dce"])
    parser.add_argument("--use_pl", action="store_true", default=False, help="prototype loss")
    parser.add_argument("--w_pl", type=float, default=1, help="weight for prototype loss")
    parser.add_argument("--use_repl", action="store_true", default=False, help="repulsive loss")
    parser.add_argument("--w_repl", type=float, default=0.01, help="weight for repulsive loss")

    args = parser.parse_args()

    if args.dataset_name == "MNIST":
        args.batch_size = 50
        args.n_classes = 10
        args.n_emb_dims = 2

    elif args.dataset_name == "CIFAR10":
        args.batch_size = 50
        args.n_classes = 10
        args.n_emb_dims = 100

    elif args.dataset_name == "CIFAR100":
        args.batch_size = 50
        args.n_classes = 100
        args.n_emb_dims = 512

    if args.model_name == "gcpl":
        list_keywords = ["gcpl"]
        if args.use_pl:
            list_keywords.append("pl")
            list_keywords.append(str(args.w_pl))
        list_keywords.append(args.dataset_name)
        args.dir_checkpoints = '_'.join(list_keywords)

        args.cnt = 3

        args.optimizer_type = "Adam"
        args.optimizer_params = {
            "lr": 1e-3,
            "betas": (0.9, 0.999),
            "eps": 1e-7
        }

        args.lr_scheduler_type = "MultiStep"
        args.lr_scheduler_params = {
            "milestones": [1, 2, 3],
            "gamma": 0.3
        }

    elif args.model_name == "dop":
        args.cnt = 3

        args.momentum_prototypes = 0.99
        args.momentum_radii = 0.99

        args.optimizer_type = "Adam"
        args.optimizer_params = {
            "lr": 1e-3,
            "betas": (0.9, 0.999),
            "eps": 1e-7
        }

        args.lr_scheduler_type = "MultiStep"
        args.lr_scheduler_params = {
            "milestones": [1, 2, 3],
            "gamma": 0.3
        }

        list_keywords = list()
        list_keywords.append(args.dataset_name)
        list_keywords.append("dop")

        if args.use_pl:
            list_keywords.append("pl")
            list_keywords.append(str(args.w_pl))

        if args.use_repl:
            list_keywords.append("repl")
            list_keywords.append(str(args.w_repl))

        args.dir_checkpoints = '_'.join(list_keywords)

    os.makedirs(f"{args.dir_root}/checkpoints/{args.dir_checkpoints}/train", exist_ok=True)
    os.makedirs(f"{args.dir_root}/checkpoints/{args.dir_checkpoints}/val", exist_ok=True)
    main(args)
