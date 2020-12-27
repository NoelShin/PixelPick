import os
from shutil import make_archive, rmtree
import numpy as np
import torch
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


def get_criterion(args):
    from criterions.dop import Criterion
    return Criterion(args)

def get_optimizer(args, params):
    if args.optimizer_type == "Adam":
        from torch.optim import Adam
        optimizer = Adam(params, **args.optimizer_params)
    else:
        raise ValueError(f"Invalid optimizer_type {args.optimizer_type}. Choices: ['Adam']")

    return optimizer

def get_lr_scheduler(args, optimizer):
    if args.dataset_name == "cv":
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40], gamma=0.1)

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

