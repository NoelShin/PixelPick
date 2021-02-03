import os
from glob import glob
from sys import path
path.append("../")
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from PIL import Image
from tqdm import tqdm
from args import Arguments
from utils.utils import get_dataloader
from utils.metrics import AverageMeter
from networks.deeplab import DeepLab


class LocalSimilarity:
    def __init__(self, args):
        self.ignore_index = args.ignore_index
        self.h = 360
        self.w = 480
        self.bin_size = 0.05
        self.dict_class_metrics = {c: {'prec': AverageMeter(),
                                       'recall': AverageMeter(),
                                       'f1_score': AverageMeter()} for c in range(12)}

    def _compute_metrics(self, window_grid, window_label, point_label):
        n_total_class = (window_label == point_label).sum()
        n_total_sim_class = (window_grid == point_label).sum()  # tp + fp

        TP = torch.logical_and((window_grid == point_label), (window_label == point_label)).sum()
        FN = ((window_grid != point_label) != (window_label != point_label)).sum()
        TP_FP = n_total_sim_class
        TP_FN = n_total_class
        assert TP <= TP_FP, f"{TP}, {TP_FP}"
        assert TP <= TP_FN, f"{TP}, {TP_FN}"
        precision = TP / TP_FP
        recall = (TP / (TP + FN))
        f1_score = 2 * (precision * recall) / (precision + recall)

        assert precision <= 1
        assert recall <= 1

        self.dict_class_metrics[point_label.cpu().numpy().item()]["prec"].update(precision.cpu().numpy())
        self.dict_class_metrics[point_label.cpu().numpy().item()]["recall"].update(recall.cpu().numpy())
        self.dict_class_metrics[point_label.cpu().numpy().item()]["f1_score"].update(f1_score.cpu().numpy())

    def print(self):
        list_prec, list_recall, list_f1 = list(), list(), list()
        for k, v in self.dict_class_metrics.items():
            list_prec.append(v["prec"].avg)
            list_recall.append(v["recall"].avg)
            list_f1.append(v["f1_score"].avg)
            print("Label {:d} | prec: {:.3f} | recall: {:.3f} | F1 score: {:.3f}"
                  .format(k, v["prec"].avg, v["recall"].avg, v['f1_score'].avg))

        print("mean prec: {:.3f} | mean recall: {:.3f} | mean F1 score: {:.3f}\n"
              .format(np.mean(list_prec), np.mean(list_recall), np.mean(list_f1)))

    def reset_metrics(self):
        for k, v in self.dict_class_metrics.items():
            v["prec"].reset()
            v["recall"].reset()
            v["f1_score"].reset()

    def get_metrics(self):
        return self.dict_class_metrics

    def _otsu(self, sim, window_grid, point_label):
        sim = torch.tanh(sim)  # using tanh gives a slightly better f1_score

        sim -= sim.min()
        sim = sim / sim.max()
        sim *= 255
        sim = np.clip(sim.cpu().numpy(), 0, 255).astype(np.uint8)
        thres, window_binary = cv2.threshold(sim, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        window_grid.flatten()[torch.tensor(window_binary.flatten() == 255)] = point_label
        return window_grid

    def _basic(self, sim, window_grid, point_label):
        sim_flat = sim.flatten()
        sim_flat *= 100
        sim_flat = torch.round(sim_flat)
        n_bins = int(2 / self.bin_size)
        list_bins = torch.arange(-1., 1., 2 / n_bins)
        hist = torch.histc(sim_flat, bins=n_bins, min=-100, max=100)

        diff = hist[1:] - hist[:-1]
        threshold = list_bins[torch.where((diff <= 0))[0][-1] + 1]
        window_grid.flatten()[sim_flat > threshold * 100] = point_label
        return window_grid

    def get_window(self, loc, x, window_size):
        (i, j) = loc
        k = window_size // 2

        # if i is NOT at the bottommost and j is NOT at the rightmost
        if i < (self.h - k) and j < (self.w - k):
            window = x[..., max(i - k, 0): i + (k + 1), max(j - k, 0): j + (k + 1)]
            top, left = max(i - k, 0), max(j - k, 0)

        # if i is at the bottommost and j is NOT at the rightmost
        elif i >= (self.h - k) and j < (self.w - k):
            window = x[..., i - k: min(i + (k + 1), self.h), max(j - k, 0): j + (k + 1)]
            top, left = i - k, max(j - k, 0)

        # if j is at the rightmost and i is NOT at the bottommost
        elif i < (self.h - k) and j >= (self.w - k):
            window = x[..., max(i - k, 0): i + (k + 1), j - k: min(j + (k + 1), self.w)]
            top, left = max(i - k, 0), j - k

        # if i is at the bottommost and j is at the rightmost
        elif i >= (self.h - k) and j >= (self.w - k):
            window = x[..., i - k: min(i + (k + 1), self.h), j - k: min(j + (k + 1), self.w)]
            top, left = i - k, j - k

        else:
            raise NotImplementedError(f"{i}, {j}, {k}")

        return window, (top, left)

    def _compute_local_similarity(self, emb, y, mask, window_size):
        grid = torch.zeros_like(y).fill_(self.ignore_index)
        for p, loc in enumerate(zip(*torch.where(mask))):
            (i, j) = loc

            point_label = y[i, j]
            assert point_label != 11

            point_feature = emb[:, i, j]
            window_emb, (top, left) = self.get_window(loc, emb, window_size)
            window_label, _ = self.get_window(loc, y, window_size)
            window_grid = torch.zeros_like(window_label).fill_(self.ignore_index)

            point_feature = point_feature.view((-1, 1, 1)).repeat((1, *window_emb.shape[1:]))
            sim = F.cosine_similarity(point_feature, window_emb, dim=0)

            # Otsu algorithm
            window_grid = self._otsu(sim, window_grid, point_label)
            grid[top: top + window_grid.shape[0], left: left + window_grid.shape[1]] = window_grid

            # compute metrics - precision, recall, f1 score
            self._compute_metrics(window_grid, window_label, point_label)
        return grid

    def __call__(self, emb, y, mask):
        emb = F.interpolate(emb, size=(self.h, self.w), mode='bilinear', align_corners=True)
        list_pseudo_labels = list()

        n_pixels_per_img = mask[0].sum()
        window_size = int(np.sqrt(self.h * self.w) // n_pixels_per_img + 1)

        for b in range(emb.shape[0]):
            pseudo_label = self._compute_local_similarity(emb[b], y[b], mask[b], window_size)
            list_pseudo_labels.append(pseudo_label)

        return list_pseudo_labels


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


def get_gt_labels(dir_annot):
    assert os.path.isdir(dir_annot)
    path_gt_labels = f"{dir_annot}/*.png"
    return sorted(glob(path_gt_labels))


if __name__ == '__main__':


    args = Arguments().parse_args()

    args.use_softmax = True
    args.query_strategy = "random"
    args.n_pixels_by_us = 10

    dataloader = get_dataloader(args, val=False, query=True,
                                shuffle=False, batch_size=args.batch_size, n_workers=args.n_workers)

    device = torch.device("cuda:0")

    torch.manual_seed(0)
    model = DeepLab(args).to(device)
    state_dict = torch.load("best_miou_model.pt")
    model.load_state_dict(state_dict["model"])

    pseudo_labelling = LocalSimilarity(args)
    masks = torch.tensor(np.load("ms_query_9.npy")).to(device)

    list_gt_labels = get_gt_labels(dir_annot="/scratch/shared/beegfs/gyungin/datasets/camvid/trainannot")
    gt_labels = list()
    for p in list_gt_labels:
        gt_labels.append(np.array(Image.open(p)))
    gt_labels = np.array(gt_labels)

    masked_labels = torch.tensor(gt_labels) * masks.cpu()

    with torch.no_grad():
        for batch_ind, dict_data in tqdm(enumerate(dataloader)):
            x, y = dict_data['x'].to(device), dict_data['y'].to(device)
            b = x.shape[0]
            mask = masks[batch_ind * args.batch_size: (batch_ind + 1) * args.batch_size]

            dict_output = model(x)
            emb, pred = dict_output["emb"], dict_output["pred"]

            pseudo_labelling(emb, y, mask)