import os
import sys
# sys.path.append("../../../segmentation")
from math import ceil

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

from networks.model import FPNSeg
from networks.deeplab import DeepLab
from utils.utils import get_dataloader, get_criterion, get_optimizer, get_lr_scheduler, get_validator, AverageMeter
from networks.modules import init_prototypes
from utils.metrics import RunningScore


class ClassHistogram:
    def __init__(self, args, nth_query):
        self.args = args
        self.dataset_name = args.dataset_name
        self.debug = args.debug
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu:0")
        self.dict_label_category = {
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

        self.dir_checkpoints = f"{args.dir_root}/checkpoints/{args.experim_name}/{nth_query}_query"
        self.experim_name = args.experim_name
        self.ignore_index = args.ignore_index

        state_dict = torch.load(f"{self.dir_checkpoints}/best_miou_model.pt")
        if args.network_name == "FPN":
            model = FPNSeg(args).to(self.device)
        else:
            model = DeepLab(args).to(self.device)
        model.load_state_dict(state_dict["model"])

        if not args.use_softmax:
            # prototypes = init_prototypes(args.n_classes, args.n_emb_dims, args.n_prototypes,
            #                              mode='zero',
            #                              ignore_index=args.ignore_index,
            #                              learnable=args.model_name == "gcpl_seg",
            #                              device=self.device)
            # prototypes.load_state_dict(state_dict["prototypes"])
            prototypes = state_dict["prototypes"].to(self.device)
        model.eval()

        self.max_budget = args.max_budget
        self.n_classes = args.n_classes
        self.n_epochs = args.n_epochs
        self.n_epochs_query = args.n_epochs_query
        self.n_pixels_per_img = args.n_pixels_per_img
        self.n_pixels_per_query = args.n_pixels_per_query
        self.nth_query = -1
        self.stride_total = args.stride_total
        self.use_softmax = args.use_softmax
        self.use_img_inp = args.use_img_inp

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu:0")
        self.dataloader_val = get_dataloader(args,
                                             val=True,
                                             query=False,
                                             shuffle=False,
                                             batch_size=1,
                                             n_workers=args.n_workers)

        running_score = RunningScore(self.n_classes)
        self.dict_img_class_hist_pix_acc, self.dict_img_class_hist_miou = dict(), dict()
        model.eval()
        with torch.no_grad():
            for ind, dict_data in tqdm(enumerate(self.dataloader_val)):
                x, y = dict_data['x'].to(self.device), dict_data['y'].to(self.device)
                dict_outputs = model(x)
                from utils.metrics import prediction, eval_metrics

                if self.use_softmax:
                    logits = dict_outputs["pred"]
                    pred = logits.argmax(dim=1)

                else:
                    emb = dict_outputs['emb']
                    pred, dist = prediction(emb, prototypes, return_distance=True)
                    # prob = F.softmax(-dist, dim=1)

                class_hist_pixel_acc, class_hist_miou = dict(), dict()
                set_unique_classes = set(y.flatten().cpu().numpy())

                running_score_image = RunningScore(self.n_classes)
                running_score_image.update(y.cpu().numpy(), pred.cpu().numpy())
                running_score.update(y.cpu().numpy(), pred.cpu().numpy())

                self.dict_img_class_hist_miou.update({ind: running_score_image.get_scores()[1]})

                for c in range(self.n_classes):
                    if c in set_unique_classes:
                        n_gt_pixels_class = (y.flatten() == c).sum().cpu().item()
                        n_pred_pixels_class = (pred.flatten()[pred.flatten() == y.flatten()] == c).sum().cpu().item()
                        assert n_pred_pixels_class <= n_gt_pixels_class
                        class_hist_pixel_acc.update({c: n_pred_pixels_class / n_gt_pixels_class})
                    else:
                        class_hist_pixel_acc.update({c: np.NaN})

                self.dict_img_class_hist_pix_acc.update({ind: class_hist_pixel_acc})

    def draw_hist(self):
        for i in range(self.n_classes):
            self.draw_class_hist(i)
        return

    def draw_class_hist(self, ind_class):
        list_pix_acc = []
        list_miou = []

        for ind_img, dict_class_hist_pix_acc in self.dict_img_class_hist_pix_acc.items():
            list_pix_acc.append(dict_class_hist_pix_acc[ind_class]) if dict_class_hist_pix_acc[ind_class] is not np.NaN else None

        from math import isnan
        for ind_img, dict_class_hist_miou in self.dict_img_class_hist_miou.items():
            if not isnan(dict_class_hist_miou[ind_class]):
                list_miou.append(dict_class_hist_miou[ind_class])

        fig, ax = plt.subplots(ncols=1, figsize=(24, 6))

        fig.suptitle(f"Label {ind_class} ({self.dict_label_category[ind_class]})")

        ax.bar(range(len(list_miou)), list_miou)

        ax.set_xlabel("image")

        ax.set_ylabel("IoU")
        plt.tight_layout()
        plt.savefig(f"{self.dir_checkpoints}/hist_{ind_class}.png")


if __name__ == '__main__':
    from args import Arguments
    torch.backends.cudnn.benchmark = True
    args = Arguments().parse_args()
    args.weight_type = "random"
    args.use_softmax = True
    c_hist = ClassHistogram(args, model).draw_hist()

