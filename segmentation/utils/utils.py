import os
from glob import glob
from shutil import make_archive, rmtree
from time import sleep

import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt
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
                self_sup_weights = torch.load("/home/gishin-temp/projects/open_set/segmentation/networks/backbones/pretrained/moco_v2_800ep_pretrain.pth.tar")["state_dict"]

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

        if args.width_multiplier < 1.0:
            state_dict = model.state_dict()
            for n, p in state_dict.items():
                # if n.find("downsample") != -1:
                #     print(type(p), n, p.shape)
                if n.find("num_batches_tracked") != -1:
                    continue

                if n.find("encoder") != -1:
                    if len(p.shape) == 1:
                        out_ch = p.shape[0]
                        state_dict[n] = p[:int(out_ch * args.width_multiplier), ...]
                    elif len(p.shape) == 4:
                        out_ch, in_ch = p.shape[:2]
                        # print(out_ch, in_ch)
                        state_dict[n] = p[:int(out_ch * args.width_multiplier), :int(in_ch * args.width_multiplier), ...]
                    else:
                        raise ValueError(n, p.shape)
                    # if any(map(lambda x: x != -1, map(n.find, ["bn", "running", "bias"]))):
                    #
                    # if n.find("bn") != -1 or n.find("bias") != -1 or n.find("downsample.1.weight") != -1:
                    #     out_ch = p.shape[0]
                    #     state_dict[n] = p[:int(out_ch * args.width_multiplier), ...]
                    #
                    # else:
                    #     try:
                    #         out_ch, in_ch = p.shape[:2]
                    #         print(out_ch, in_ch)
                    #         state_dict[n] = p[:int(out_ch * args.width_multiplier), :int(in_ch * args.width_multiplier), ...]
                    #     except ValueError:
                    #         raise ValueError(n, p.shape)
                    # if n.find("weight") != -1 and n.find("bn") == -1 and n.find("downsample") == -1 and n.find("prefix.conv1.weight") == -1:
                    #     out_ch, in_ch = p.shape[:2]
                    #     print(out_ch, in_ch)
                    #     state_dict[n] = p[:int(out_ch * args.width_multiplier), :int(in_ch * args.width_multiplier), ...]
                    #
                    # elif n.find("weight") != -1 and n.find("bn") == -1 and n.find("prefix.conv1.weight") == -1:
                    #
                    #
                    # else:
                    #     out_ch = p.shape[0]
                    #     state_dict[n] = p[:int(out_ch * args.width_multiplier), ...]
            for n, p in model.state_dict().items():
                print(n, p.shape)
            exit(12)
            model = FPNSeg(args, load_pretrained=False)
            m_s = model.state_dict()
            for n, p in state_dict.items():
                if n.find("downsample") != -1:
                    print(n, m_s[n].shape, p.shape)
            exit(12)
            model.load_state_dict(state_dict)
            print(f"Prunned model (width: {args.width_multiplier}) loaded.")
            # m_s = model.state_dict()
            # for n, p in state_dict.items():
            #     print(n, m_s[n].shape, p.shape)
            # exit(12)

            # for n, p in model.named_parameters():
            #     if n.find("encoder") != -1:
            #         print(n, p.shape)
                # if n.find("encoder") != -1:
                #     n_ch = p.shape[0]
                #     p = p[:int(n_ch * args.width_multiplier), ...]
            # for n, p in model.named_parameters():
            #     if n.find("encoder") != -1:
            #         n_ch = p.shape[0]
            #         p = p[:int(n_ch * args.width_multiplier), ...]
        elif args.width_multiplier > 1.0:
            pass

    elif args.network_name == "deeplab":
        model = DeepLab(args)

    return model


def send_file(fp, file_name=None, remove_file=False):
    try:
        if file_name is not None:
            os.system(f"scp -P 2969 {fp} gishin@59.16.193.79:/home/gishin/vgg/{file_name}.zip")
        else:
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


def get_optimizer(args, model, prototypes=None):
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

        if args.use_contrastive_loss:
            list_params += [{'params': model.projection_head.parameters(),
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

                if prototypes is not None:
                    list_params += [{'params': prototypes,
                                     'lr': 0.1,
                                     'betas': (0.9, 0.999)}] if args.model_name == "gcpl_seg" else []

                    list_params += [{'params': model.fc,
                                     'lr': 5e-4,
                                     'betas': (0.9, 0.999)}] if args.model_name == "gcpl_seg" else []

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

            if prototypes is not None:
                list_params += [{'params': prototypes,
                                 'lr': optimizer_params['lr'],
                                 'weight_decay': optimizer_params['weight_decay']}] if args.model_name == "gcpl_seg" else []

                # list_params += [{'params': model.fc.parameters(),
                #                  'lr': optimizer_params['lr'],
                #                  'betas': (0.9, 0.999)}] if args.model_name == "gcpl_seg" else []
                list_params += [{'params': model.aspp_.parameters(),
                                 'lr': optimizer_params['lr'],
                                 'weight_decay': optimizer_params['weight_decay']}]

                list_params += [{'params': model.low_level_conv_.parameters(),
                                 'lr': optimizer_params['lr'],
                                 'weight_decay': optimizer_params['weight_decay']}]

                list_params += [{'params': model.seg_head_.parameters(),
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

            if prototypes is not None:
                list_params += [{'params': prototypes,
                                 'lr': 0.1,
                                 'betas': (0.9, 0.999)}] if args.model_name == "gcpl_seg" else []

                list_params += [{'params': model.fc,
                                 'lr': 5e-4,
                                 'betas': (0.9, 0.999)}] if args.model_name == "gcpl_seg" else []

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


def get_dict_label_cnt(arr_masks, arr_labels):
    dict_label_cnt = dict()
    for mask, label in zip(arr_masks, arr_labels):
        label_flat = label.flatten()
        selected_label_pixels = label_flat[mask.flatten()]
        print(selected_label_pixels)
        set_labels = set(selected_label_pixels)
        for l in set_labels:
            cnt = (selected_label_pixels == l).sum()
            try:
                dict_label_cnt[l] += cnt
            except KeyError:
                dict_label_cnt.update({l: cnt})
    return dict_label_cnt


def colorise_label(arr):
    palette = {
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
    assert len(arr.shape) == 2
    grid = np.empty((3, *arr.shape), dtype=np.uint8)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            grid[:, i, j] = palette[arr[i, j]]

    return np.transpose(grid, (1, 2, 0))


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
        try:
            list_imgs.append(self._preprocess(dict_tensors['img_inp_target'], seg=False))
        except KeyError:
            pass

        try:
            list_imgs.append(self._preprocess(dict_tensors['img_inp_output'], seg=False))
        except KeyError:
            pass

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


def get_gt_labels(dataset_name="cv", as_array=True):
    if dataset_name == "cv":
        dir_annot = "/scratch/shared/beegfs/gyungin/datasets/camvid/trainannot"
        assert os.path.isdir(dir_annot)
        path_gt_labels = f"{dir_annot}/*.png"
        list_gt_labels = sorted(glob(path_gt_labels))

    elif dataset_name == "cs":
        dir_annot = "/scratch/shared/beegfs/gyungin/datasets/cityscapes_d4/gtFine/train"
        assert os.path.isdir(dir_annot)
        path_gt_labels = f"{dir_annot}/**/*.png"
        list_gt_labels = sorted(glob(path_gt_labels))

    if as_array:
        gt_labels = list()
        for p in list_gt_labels:
            gt_labels.append(np.array(Image.open(p)))
        gt_labels = np.stack(gt_labels, axis=0)
    else:
        gt_labels = list_gt_labels
    return gt_labels