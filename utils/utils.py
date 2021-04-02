import os
from glob import glob
from shutil import make_archive, rmtree
from time import sleep

import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image

from networks.model import FPNSeg
from networks.deeplab import DeepLab


def get_model(args):
    if args.network_name == "FPN":
        model = FPNSeg(args)

        if args.weight_type == "moco_v2":
            assert args.n_layers == 50, args.n_layers
            # path to moco_v2 weights. Current path is relative to scripts dir
            try:
                self_sup_weights = torch.load("../networks/backbones/pretrained/moco_v2_800ep_pretrain.pth.tar")["state_dict"]
            except FileNotFoundError:
                self_sup_weights = torch.load("../networks/backbones/pretrained/moco_v2_800ep_pretrain.pth.tar")["state_dict"]

            model_state_dict = model.encoder.state_dict()

            for k in list(self_sup_weights.keys()):
                if k.replace("module.encoder_q.", '') in ["fc.0.weight", "fc.0.bias", "fc.2.weight", "fc.2.bias"]:
                    self_sup_weights.pop(k)
            for name, param in self_sup_weights.items():
                name = name.replace("encoder_q.", '').replace("module", 'base')

                if name.replace("base.", '') in ["conv1.weight", "bn1.weight", "bn1.bias", "bn1.running_mean",
                                                 "bn1.running_var", "bn1.num_batches_tracked"]:
                    name = name.replace("base", "base.prefix")

                if name not in model_state_dict:
                    print(f"{name} is not applied!")
                    continue

                if isinstance(param, torch.nn.Parameter):
                    param = param.data
                model_state_dict[name].copy_(param)
            model.encoder.load_state_dict(model_state_dict)
            print("moco_v2 weights are loaded successfully.")

    elif args.network_name == "deeplab":
        model = DeepLab(args)
    return model


def zip_dir(d, fp=None, fmt='zip', remove_dir=False):
    assert os.path.isdir(d), f"{d} does not exist."
    if fp is None:
        fp = d

    make_archive(fp, fmt, d)

    if remove_dir:
        rmtree(d)
    return f"{fp}.{fmt}"


def write_log(fp, list_entities=None, header=None):
    if list_entities is not None:
        list_entities = [str(e) for e in list_entities]
    with open(fp, 'w' if header is not None else 'a') as f:
        f.write(','.join(header) + '\n') if header is not None else None
        f.write(','.join(list_entities) + '\n') if list_entities is not None else None
    f.close()


def get_dataloader(args, batch_size, n_workers, shuffle, val=False, query=False):
    if args.dataset_name == "cs":
        from datasets.cityscapes import CityscapesDataset
        dataset = CityscapesDataset(args, val=val, query=query)

    elif args.dataset_name == "cv":
        from datasets.camvid import CamVidDataset
        dataset = CamVidDataset(args, val=val, query=query)

    elif args.dataset_name == "voc":
        from datasets.voc import VOC2012Segmentation
        dataset = VOC2012Segmentation(args, val=val, query=query)

    else:
        raise ValueError(args.dataset_name)

    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            num_workers=n_workers,
                            shuffle=shuffle,
                            drop_last=len(dataset) % batch_size == 1)
    return dataloader


def get_optimizer(args, model):
    optimizer_params = args.optimizer_params
    if args.dataset_name == "cs":
        from torch.optim import Adam
        if args.network_name == "FPN":
            list_params = [{'params': model.encoder.parameters(),
                            'lr': optimizer_params['lr'] / 10,
                            'weight_decay': optimizer_params['weight_decay']}]

            list_params += [{'params': model.decoder.parameters(),
                             'lr': optimizer_params['lr'],
                             'weight_decay': optimizer_params['weight_decay']}]
        else:
            list_params = [{'params': model.backbone.parameters(),
                            'lr': optimizer_params['lr'] / 10,
                            'weight_decay': optimizer_params['weight_decay']}]

            list_params += [{'params': model.aspp.parameters(),
                             'lr': optimizer_params['lr'],
                             'weight_decay': optimizer_params['weight_decay']}]

            list_params += [{'params': model.low_level_conv.parameters(),
                             'lr': optimizer_params['lr'],
                             'weight_decay': optimizer_params['weight_decay']}]

            list_params += [{'params': model.seg_head.parameters(),
                             'lr': optimizer_params['lr'],
                             'weight_decay': optimizer_params['weight_decay']}]

        optimizer = Adam(list_params)

    elif args.dataset_name == "cv":
        if args.optimizer_type == "SGD":
            from torch.optim import SGD
            if args.network_name == "FPN":
                list_params = [{'params': model.encoder.parameters(),
                                'lr': 1e-3,
                                'weight_decay': 5e-4,
                                'momentum': 0.9}]

                list_params += [{'params': model.decoder.parameters(),
                                 'lr': 1e-2,
                                 'weight_decay': 5e-4,
                                 'momentum': 0.9}]
            else:
                list_params = [{'params': model.backbone.parameters(),
                                'lr': 1e-3,
                                'weight_decay': 5e-4,
                                'momentum': 0.9}]

                list_params += [{'params': model.aspp.parameters(),
                                 'lr': 1e-2,
                                 'weight_decay': 5e-4,
                                 'momentum': 0.9}]

                list_params += [{'params': model.low_level_conv.parameters(),
                                 'lr': 1e-2,
                                 'weight_decay': 5e-4,
                                 'momentum': 0.9}]

                list_params += [{'params': model.seg_head.parameters(),
                                 'lr': 1e-2,
                                 'weight_decay': 5e-4,
                                 'momentum': 0.9}]

            optimizer = SGD(list_params)

        elif args.optimizer_type == "Adam":
            from torch.optim import Adam
            if args.network_name == "FPN":
                list_params = [{'params': model.encoder.parameters(),
                                'lr': optimizer_params['lr'] / 10,
                                'weight_decay': optimizer_params['weight_decay']}]

                list_params += [{'params': model.decoder.parameters(),
                                 'lr': optimizer_params['lr'],
                                 'weight_decay': optimizer_params['weight_decay']}]
            else:
                list_params = [{'params': model.backbone.parameters(),
                                'lr': optimizer_params['lr'] / 10,
                                'weight_decay': optimizer_params['weight_decay']}]

                list_params += [{'params': model.aspp.parameters(),
                                 'lr': optimizer_params['lr'],
                                 'weight_decay': optimizer_params['weight_decay']}]

                list_params += [{'params': model.low_level_conv.parameters(),
                                 'lr': optimizer_params['lr'],
                                 'weight_decay': optimizer_params['weight_decay']}]

                list_params += [{'params': model.seg_head.parameters(),
                                 'lr': optimizer_params['lr'],
                                 'weight_decay': optimizer_params['weight_decay']}]

            optimizer = Adam(list_params)

    elif args.dataset_name == "voc":
        from torch.optim import SGD
        if args.network_name == "FPN":
            list_params = [{'params': model.encoder.parameters(),
                            'lr': 1e-3,
                            'weight_decay': 1e-4,
                            'momentum': 0.9}]

            list_params += [{'params': model.decoder.parameters(),
                             'lr': 1e-2,
                             'weight_decay': 1e-4,
                             'momentum': 0.9}]
        else:
            list_params = [{'params': model.backbone.parameters(),
                            'lr': 1e-3,
                            'weight_decay': 5e-4,
                            'momentum': 0.9}]

            list_params += [{'params': model.aspp.parameters(),
                             'lr': 1e-2,
                             'weight_decay': 5e-4,
                             'momentum': 0.9}]

            list_params += [{'params': model.low_level_conv.parameters(),
                             'lr': 1e-2,
                             'weight_decay': 5e-4,
                             'momentum': 0.9}]

            list_params += [{'params': model.seg_head.parameters(),
                             'lr': 1e-2,
                             'weight_decay': 5e-4,
                             'momentum': 0.9}]

        optimizer = SGD(list_params)

    return optimizer


def get_lr_scheduler(args, optimizer, iters_per_epoch=-1):
    if args.dataset_name == "cs":
        if args.lr_scheduler_type == "MultiStepLR":
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40], gamma=0.1)
        elif args.lr_scheduler_type == "Poly":
            from .lr_scheduler import Poly
            lr_scheduler = Poly(optimizer, args.n_epochs, iters_per_epoch)

    elif args.dataset_name == "cv":
        if args.lr_scheduler_type == "MultiStepLR":
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40], gamma=0.1)
        elif args.lr_scheduler_type == "Poly":
            from .lr_scheduler import Poly
            lr_scheduler = Poly(optimizer, args.n_epochs, iters_per_epoch)

    elif args.dataset_name == "voc":
        from utils.lr_scheduler import Poly
        lr_scheduler = Poly(optimizer, args.n_epochs, iters_per_epoch)

    return lr_scheduler


def get_dict_label_cnt(arr_masks, arr_labels):
    dict_label_cnt = dict()
    for mask, label in zip(arr_masks, arr_labels):
        label_flat = label.flatten()
        selected_label_pixels = label_flat[mask.flatten()]
        set_labels = set(selected_label_pixels)
        for l in set_labels:
            cnt = (selected_label_pixels == l).sum()
            try:
                dict_label_cnt[l] += cnt
            except KeyError:
                dict_label_cnt.update({l: cnt})
    return dict_label_cnt


def colorise_label(arr, dataset="cv"):
    assert len(arr.shape) == 2, arr.shape
    assert dataset in ["cv", "cs", "voc"], dataset
    if dataset == "cv":
        global palette_cv
        palette = palette_cv

    elif dataset == "cs":
        global palette_cs
        palette = palette_cs

    else:
        global palette_voc
        palette = palette_voc

    grid = np.empty((3, *arr.shape), dtype=np.uint8)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            grid[:, i, j] = palette[arr[i, j]]

    return np.transpose(grid, (1, 2, 0))


class Visualiser:
    def __init__(self, dataset_name):
        if dataset_name == "cv":
            global palette_cv
            self.palette = palette_cv

        elif dataset_name == "cs":
            global palette_cs
            self.palette = palette_cs

        elif dataset_name == "voc":
            global palette_voc
            self.palette = palette_voc

    def _preprocess(self, tensor, seg, downsample=2):
        if len(tensor.shape) == 2:
            h, w = tensor.shape
        elif len(tensor.shape) == 3:
            c, h, w = tensor.shape
        else:
            raise ValueError(f"{tensor.shape}")

        if seg:
            tensor_flatten = tensor.flatten()
            grid = torch.zeros([h * w, 3], dtype=torch.uint8)
            for i in range(len(tensor_flatten)):
                grid[i] = torch.tensor(self.palette[tensor_flatten[i].item()], dtype=torch.uint8)
            tensor = grid.view(h, w, 3)

        else:
            tensor -= tensor.min()
            tensor = tensor / (tensor.max() + 1e-7)
            tensor *= 255

            if len(tensor.shape) == 3:
                tensor = tensor.permute(1, 2, 0)

        arr = np.clip(tensor.numpy(), 0, 255).astype(np.uint8)
        return Image.fromarray(arr).resize((w // downsample, h // downsample))

    @staticmethod
    def _make_grid(list_imgs):
        width = 0
        height = list_imgs[0].height
        for img in list_imgs:
            width += img.width

        grid = Image.new("RGB", (width, height))
        x_offset = 0
        for img in list_imgs:
            grid.paste(img, (x_offset, 0))
            x_offset += img.width
        return grid

    def __call__(self, dict_tensors, fp='', show=False):
        list_imgs = list()

        list_imgs.append(self._preprocess(dict_tensors['input'], seg=False))

        list_imgs.append(self._preprocess(dict_tensors['target'], seg=True))
        list_imgs.append(self._preprocess(dict_tensors['pred'], seg=True))
        list_imgs.append(self._preprocess(dict_tensors['confidence'], seg=False))
        list_imgs.append(self._preprocess(dict_tensors['margin'], seg=False))
        list_imgs.append(self._preprocess(dict_tensors['entropy'], seg=False))

        img = self._make_grid(list_imgs)

        if fp:
            img.save(fp)

        if show:
            img.show()
            sleep(60)
        return


palette_cv = {
    0: (128, 128, 128),
    1: (128, 0, 0),
    2: (192, 192, 128),
    3: (128, 64, 128),
    4: (0, 0, 192),
    5: (128, 128, 0),
    6: (192, 128, 128),
    7: (64, 64, 128),
    8: (64, 0, 128),
    9: (64, 64, 0),
    10: (0, 128, 192),
    11: (0, 0, 0)
}

palette_cs = {
    0: (128, 64, 128),
    1: (244, 35, 232),
    2: (70, 70, 70),
    3: (102, 102, 156),
    4: (190, 153, 153),
    5: (153, 153, 153),
    6: (250, 170, 30),
    7: (220, 220, 0),
    8: (107, 142, 35),
    9: (152, 251, 152),
    10: (70, 130, 180),
    11: (220, 20, 60),
    12: (255, 0, 0),
    13: (0, 0, 142),
    14: (0, 0, 70),
    15: (0, 60, 100),
    16: (0, 80, 100),
    17: (0, 0, 230),
    18: (119, 11, 32),
    19: (0, 0, 0)
}

palette_voc = {
    0: [0, 0, 0],
    1: [128, 0, 0],
    2: [0, 128, 0],
    3: [128, 128, 0],
    4: [0, 0, 128],
    5: [128, 0, 128],
    6: [0, 128, 128],
    7: [128, 128, 128],
    8: [64, 0, 0],
    9: [192, 0, 0],
    10: [64, 128, 0],
    11: [192, 128, 0],
    12: [64, 0, 128],
    13: [192, 0, 128],
    14: [64, 128, 128],
    15: [192, 128, 128],
    16: [0, 64, 0],
    17: [128, 64, 0],
    18: [0, 192, 0],
    19: [128, 192, 0],
    20: [0, 64, 128],
    255: [255, 255, 255]
}

dict_cv_label_category = {
    0: "sky",
    1: "building",
    2: "pole",
    3: "road",
    4: "pavement",
    5: "tree",
    6: "sign symbol",
    7: "fence",
    8: "car",
    9: "pedestrian",
    10: "bicyclist",
    11: "void"
}