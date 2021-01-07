import os
from shutil import make_archive, rmtree
from time import sleep

import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt


def send_file(fp, remove_file=False):
    try:
        os.system(f"scp -P 2969 {fp} gishin@59.16.193.79:/home/gishin/vgg")
        if remove_file:
            os.remove(fp)
    except:
        pass
    return


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


def get_criterion(args, device):
    from criterions.dop import Criterion
    return Criterion(args, device)


def get_dataloader(args, batch_size, n_workers, shuffle, val=False, query=False):
    if args.dataset_name == "cv":
        from datasets.camvid import CamVidDataset
        dataset = CamVidDataset(args, val, query=query)

    elif args.dataset_name == "voc":
        from datasets.voc import VOC2012Segmentation
        dataset = VOC2012Segmentation(args, val)

    else:
        raise ValueError(args.dataset_name)

    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            num_workers=n_workers,
                            shuffle=shuffle,
                            drop_last=True if len(dataset) % batch_size != 0 else False)
    return dataloader


def get_optimizer(args, model, prototypes=None):
    optimizer_params = args.optimizer_params
    if args.dataset_name == "cv":
        from torch.optim import Adam
        list_params = [{'params': model.encoder.parameters(),
                        'lr': optimizer_params['lr'] / 10,
                        'weight_decay': optimizer_params['weight_decay']}]

        list_params += [{'params': model.decoder.parameters(),
                         'lr': optimizer_params['lr'],
                         'weight_decay': optimizer_params['weight_decay']}]

        if prototypes is not None:
            list_params += [{'params': prototypes,
                             'lr': 0.1,
                             'betas': (0.9, 0.999),
                             'weight_decay': optimizer_params['weight_decay']}] if args.model_name == "gcpl_seg" else []
        optimizer = Adam(list_params)

    elif args.dataset_name == "voc":
        from torch.optim import SGD
        list_params = [{'params': model.encoder.parameters(),
                        'lr': optimizer_params['lr'] / 10,
                        'weight_decay': optimizer_params['weight_decay'],
                        'momentum': optimizer_params['momentum']}]

        list_params += [{'params': model.decoder.parameters(),
                         'lr': optimizer_params['lr'],
                         'weight_decay': optimizer_params['weight_decay'],
                         'momentum': optimizer_params['momentum']}]

        if prototypes is not None:
            list_params += [{'params': prototypes,
                             'lr': 1.,
                             'weight_decay': optimizer_params['weight_decay'],
                             'momentum': optimizer_params['momentum']}] if args.model_name == "gcpl_seg" else []
        optimizer = SGD(list_params)

    else:
        raise ValueError(f"Invalid optimizer_type {args.optimizer_type}. Choices: ['Adam']")

    return optimizer


def get_lr_scheduler(args, optimizer, iters_per_epoch=-1):
    if args.dataset_name == "cv":
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40], gamma=0.1)

    elif args.dataset_name == "voc":
        from utils.lr_scheduler import Poly
        lr_scheduler = Poly(optimizer, args.n_epochs, iters_per_epoch)

    return lr_scheduler


def get_validator(args):
    from validators.dop import Validator
    return Validator(args)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.initialized = False
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.initialized = False
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = np.multiply(val, weight)
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum = np.add(self.sum, np.multiply(val, weight))
        self.count = self.count + weight
        self.avg = self.sum / self.count

    @property
    def value(self):
        return self.val

    @property
    def average(self):
        return np.round(self.avg, 5)


class EmbeddingVisualiser:
    def __init__(self, n_classes, unseen_class=-1):
        self.dict_label_emb = {i: list() for i in range(n_classes)}
        self.n_classes = n_classes
        self.unseen_class = unseen_class

        self.dict_cmap = {
            0: "tab:blue",
            1: "tab:orange",
            2: "tab:green",
            3: "tab:red",
            4: "tab:purple",
            5: "tab:brown",
            6: "tab:pink",
            7: "tab:gray",
            8: "tab:olive",
            9: "tab:cyan"
        }

    def add(self, emb, labels=None):
        assert isinstance(emb, torch.Tensor), "argument emb should be an instance of torch.Tensor, got {}".format(type(emb))
        if labels is not None:
            assert isinstance(labels, torch.Tensor), "argument labels should be an instance of torch.Tensor, got {}".format(type(labels))
            assert emb.shape[0] == labels.shape[0], "batch size of arguments emb and labels should be the same, {} and {}". format(emb.shape[0], labels.shape[0])
            for i in range(labels.shape[0]):
                label = labels[i].detach().cpu().item()
                self.dict_label_emb[label].append(emb[i].detach().cpu().numpy())

        else:
            self.dict_label_emb.update({self.unseen_class: list()})
            for i in range(emb.shape[0]):
                self.dict_label_emb[self.unseen_class].append(emb.detach().cpu().numpy())

    def visualise(self, prototypes=None, radii=None, use_pca=False, fp=None, show=False):
        if use_pca:
            from sklearn.decomposition import PCA

            list_all_embs = list()
            for label, list_embs in sorted(self.dict_label_emb.items()):
                list_all_embs.extend(np.array(list_embs))

            arr = np.array(list_all_embs)

            pca = PCA(n_components=2, random_state=0)

            arr = pca.fit(arr).transform(arr)

            ind = 0
            for label, list_embs in sorted(self.dict_label_emb.items()):
                self.dict_label_emb[label] = arr[ind: ind + len(list_embs)]
                ind += len(list_embs)

        plt.figure()
        for label, list_embs in sorted(self.dict_label_emb.items()):
            embs = np.array(list_embs).transpose(1, 0)
            plt.plot(*embs, ls='none', marker='o', markersize=1.0, color=self.dict_cmap[label], alpha=0.1,
                     label=None if prototypes is not None else label)
            if prototypes is not None:
                plt.plot(*prototypes[label], ls='none', marker='^', markersize=7.0, color=self.dict_cmap[label],
                         label=label)

            if radii is not None:
                assert prototypes is not None, "radii should be given with prototypes (centers)"
                prototype_class = prototypes[label]
                x_basis = torch.tensor((1, 0))
                angle = torch.arccos((prototype_class * x_basis).sum() / prototype_class.pow(2).sum().sqrt())
                # print(radii, angle)
                from matplotlib.patches import Ellipse
                ellipse = Ellipse(xy=prototype_class, width=radii[label][0], height=radii[label][1], angle=angle.numpy(),
                                  fc=self.dict_cmap[label], lw=10, alpha=0.7)

                plt.gca().add_patch(ellipse)

        plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
        plt.xlabel("Activation of the 1st neuron")
        plt.ylabel("Activation of the 2nd neuron")
        plt.tight_layout()

        if fp:
            plt.savefig(fp)

        if show:
            plt.show()
        plt.close()

    def reset(self):
        self.dict_label_emb = {i: list() for i in range(self.n_classes)}


class Visualiser:
    def __init__(self, dataset_name):
        if dataset_name == "cv":
            self.palette = {
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

        elif dataset_name == "cs":
            self.palette = {
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

        elif dataset_name == "voc":
            self.palette = {
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

        # for name, tensor in dict_tensors.items():
        #     list_imgs.append(self._preprocess(tensor, seg=name in ['target', 'pred']))

        img = self._make_grid(list_imgs)

        if fp:
            img.save(fp)

        if show:
            img.show()
            sleep(60)

        return
