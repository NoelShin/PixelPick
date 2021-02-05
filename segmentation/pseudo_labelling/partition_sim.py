import os
from glob import glob
from sys import path

path.append("../")
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from args import Arguments
from utils.utils import get_dataloader, colorise_label

from utils.metrics import AverageMeter
from networks.deeplab import DeepLab


class LocalSimilarity:
    def __init__(self, args):
        self.ignore_index = args.ignore_index
        self.h = 360
        self.w = 480
        self.bin_size = 0.05
        self.labelling_strategy = args.labelling_strategy

        self.dict_class_metrics = {c: {'prec': AverageMeter(),
                                       'recall': AverageMeter(),
                                       'f1_score': AverageMeter(),
                                        'prec_g': AverageMeter(),
                                        'recall_g': AverageMeter(),
                                        'f1_score_g': AverageMeter()
                                       } for c in range(12)}


    def _partition(self, points):
        ind_points = np.where(points)
        arr_h = np.expand_dims(np.array(range(self.h), np.int32), axis=1).repeat(self.w, axis=1)
        arr_w = np.expand_dims(np.array(range(self.w), np.int32), axis=0).repeat(self.h, axis=0)

        arr = np.stack((arr_h, arr_w), axis=0)

        list_distance_maps = list()
        for num_p, (i_p, j_p) in enumerate(zip(*ind_points)):
            arr_p = np.empty_like(arr)
            arr_p[0].fill(i_p)
            arr_p[1].fill(j_p)

            distance_map = ((arr_p - arr) ** 2).sum(axis=0)
            list_distance_maps.append(distance_map)

        distance_maps = np.array(list_distance_maps)
        grid = distance_maps.argmin(axis=0).squeeze()

        return grid, ind_points

    def _compute_metrics(self, window_grid, window_label, point_label, inplace=True):
        TP_FP_mask = (window_grid == point_label)
        P_mask = (window_label == point_label)
        assert P_mask.sum() > 0, point_label

        TP_mask = torch.logical_and(TP_FP_mask, P_mask)
        TP = TP_mask.sum()

        TN_FN_mask = window_grid != point_label
        FN = torch.logical_and(TN_FN_mask, P_mask).sum()

        # dict_masks = {"TP": TP_mask, "TP_FP": TP_FP_mask, "TN_FN_mask": TN_FN_mask, "P": P_mask}
        # self._plot_masks(dict_masks)

        TP_FP = (window_grid == point_label).sum()
        precision = TP / (TP_FP + 1e-8)
        recall = (TP / (TP + FN + 1e-8))
        f1_score = 2 * (precision * recall) / ((precision + recall) + 1e-8)

        assert precision <= 1, precision
        assert recall <= 1, recall

        if inplace:
            self.dict_class_metrics[point_label.cpu().numpy().item()]["prec"].update(precision.cpu().numpy())
            self.dict_class_metrics[point_label.cpu().numpy().item()]["recall"].update(recall.cpu().numpy())
            self.dict_class_metrics[point_label.cpu().numpy().item()]["f1_score"].update(f1_score.cpu().numpy())
        return precision, recall, f1_score

    def _compute_metrics_global(self, grid, label, mask, merged_mask):
        h, w = label.shape

        set_unique_labels = set((label.flatten()[mask.flatten()]).cpu().numpy())

        label_flat = label.flatten()
        label_flat[~merged_mask.flatten()] = self.ignore_index
        label = label_flat.view(h, w)

        for ul in set_unique_labels:
            prec, recall, f1 = self._compute_metrics(grid, label, ul, inplace=False)
            self.dict_class_metrics[ul]["prec_g"].update(prec.cpu().numpy())
            self.dict_class_metrics[ul]["recall_g"].update(recall.cpu().numpy())
            self.dict_class_metrics[ul]["f1_score_g"].update(f1.cpu().numpy())

        return

    def print(self):
        list_prec, list_recall, list_f1 = list(), list(), list()
        list_prec_g, list_recall_g, list_f1_g = list(), list(), list()
        for k, v in self.dict_class_metrics.items():
            list_prec.append(v["prec"].avg)
            list_recall.append(v["recall"].avg)
            list_f1.append(v["f1_score"].avg)

            list_prec_g.append(v["prec_g"].avg)
            list_recall_g.append(v["recall_g"].avg)
            list_f1_g.append(v["f1_score_g"].avg)

            print("Label {:d} | prec: {:.3f} | recall: {:.3f} | F1 score: {:.3f} |"
                  "prec_g: {:.3f} | recall_g: {:.3f} | F1 score_g: {:.3f}"
                  .format(k, v["prec"].avg, v["recall"].avg, v['f1_score'].avg,
                          v["prec_g"].avg, v["recall_g"].avg, v['f1_score_g'].avg))

        print("mean prec: {:.3f} | mean recall: {:.3f} | mean F1 score: {:.3f} |"
              "mean prec_g: {:.3f} | mean recall_g: {:.3f} | mean F1 score_g: {:.3f}\n"
              .format(np.mean(list_prec[:-1]), np.mean(list_recall[:-1]), np.mean(list_f1[:-1]),
                      np.mean(list_prec_g[:-1]), np.mean(list_recall_g[:-1]), np.mean(list_f1_g[:-1])))

    def write(self, fp):
        with open(fp, 'w') as f:
            list_prec, list_recall, list_f1 = list(), list(), list()
            list_prec_g, list_recall_g, list_f1_g = list(), list(), list()

            for k, v in self.dict_class_metrics.items():
                list_prec.append(v["prec"].avg)
                list_recall.append(v["recall"].avg)
                list_f1.append(v["f1_score"].avg)
                list_prec_g.append(v["prec_g"].avg)
                list_recall_g.append(v["recall_g"].avg)
                list_f1_g.append(v["f1_score_g"].avg)

                f.write(f"{k}, {v['prec'].avg}, {v['recall'].avg}, {v['f1_score'].avg}, {v['prec_g'].avg}, {v['recall_g'].avg}, {v['f1_score_g'].avg}\n")

            f.write(f"mean, {np.mean(list_prec[:-1])}, {np.mean(list_recall[:-1])}, {np.mean(list_f1[:-1])}, {np.mean(list_prec_g[:-1])}, {np.mean(list_recall_g[:-1])}, {np.mean(list_f1_g[:-1])}\n")
            f.close()

    def reset_metrics(self):
        for k, v in self.dict_class_metrics.items():
            v["prec"].reset()
            v["recall"].reset()
            v["f1_score"].reset()

    def get_metrics(self):
        return self.dict_class_metrics

    def _otsu(self, sim, window_grid=None, point_label=None):
        sim = torch.tanh(sim)  # using tanh gives a slightly better f1_score

        sim -= sim.min()
        sim = sim / sim.max()
        sim *= 255
        sim = np.clip(sim.cpu().numpy(), 0, 255).astype(np.uint8)
        thres, window_binary = cv2.threshold(sim, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if None not in [window_grid, point_label]:
            window_grid.flatten()[torch.tensor(window_binary.flatten() == 255)] = point_label
            return window_grid
        else:
            return thres, window_binary

    def _baseline(self, window_grid, point_label):
        window_grid.fill_(point_label)
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
        if len(x.shape) == 2:
            h, w = x.shape
        elif len(x.shape) > 2:
            h, w = x.shape[-2:]
        else:
            raise IndexError("Please check shape of x: {}".format(x.shape))

        # if i is NOT at the bottommost and j is NOT at the rightmost
        if i < (h - k) and j < (w - k):
            window = x[..., max(i - k, 0): i + (k + 1), max(j - k, 0): j + (k + 1)]
            top, left = max(i - k, 0), max(j - k, 0)

        # if i is at the bottommost and j is NOT at the rightmost
        elif i >= (h - k) and j < (w - k):
            window = x[..., i - k: min(i + (k + 1), h), max(j - k, 0): j + (k + 1)]
            top, left = i - k, max(j - k, 0)

        # if j is at the rightmost and i is NOT at the bottommost
        elif i < (h - k) and j >= (w - k):
            window = x[..., max(i - k, 0): i + (k + 1), j - k: min(j + (k + 1), w)]
            top, left = max(i - k, 0), j - k

        # if i is at the bottommost and j is at the rightmost
        elif i >= (h - k) and j >= (w - k):
            window = x[..., i - k: min(i + (k + 1), h), j - k: min(j + (k + 1), w)]
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

            if self.labelling_strategy == "baseline":
                window_grid = self._baseline(window_grid, point_label)

            elif self.labelling_strategy == "local_sim":
                # Otsu algorithm
                point_feature = point_feature.view((-1, 1, 1)).repeat((1, *window_emb.shape[1:]))
                sim = F.cosine_similarity(point_feature, window_emb, dim=0)
                window_grid = self._otsu(sim, window_grid, point_label)
            grid[top: top + window_grid.shape[0], left: left + window_grid.shape[1]] = window_grid

            # compute metrics - precision, recall, f1 score
            self._compute_metrics(window_grid, window_label, point_label)

        return grid

    def _plot_masks(self, dict_masks):
        for k, v in dict_masks.items():
            if isinstance(v, torch.Tensor):
                dict_masks[k] = v.cpu().numpy()

        fig, ax = plt.subplots(ncols=len(dict_masks))
        for i, (k, v) in enumerate(dict_masks.items()):
            ax[i].imshow(v, cmap="gray")
            ax[i].set_xlabel(k)
        plt.tight_layout()
        plt.show()
        plt.close()
        exit(12)
        return

    def __call__(self, emb, y, mask, window_size):
        emb = F.interpolate(emb, size=(self.h, self.w), mode='bilinear', align_corners=True)

        grid = torch.zeros_like(y).fill_(self.ignore_index)
        partitions, ind_points = self._partition(mask[0].cpu().numpy())

        merged_mask = torch.zeros_like(grid[0], dtype=torch.bool)
        for p in range(partitions.max() + 1):
            i, j = ind_points[0][p], ind_points[1][p]

            emb_point = emb[0, :, i, j]  # 256
            y_point = y[0, i, j]
            assert y_point != 11

            mask_p = (partitions == p)  # h x w

            mask_rec, (t, l) = self.get_window(loc=(i, j), x=mask_p, window_size=window_size)
            window_h, window_w = mask_rec.shape
            assert window_h >= 1 and window_w >= 1
            assert t <= i < t + window_h
            assert l <= j < l + window_w

            emb_rec = emb[0, :, t: t + window_h, l: l + window_w]  # 256 x window_h x window_w
            emb_point = emb_point.unsqueeze(1).unsqueeze(2).repeat(1, window_h, window_w)  # 256 x window_h x window_w
            sim = F.cosine_similarity(emb_point, emb_rec, dim=0)  # window_size x window_size

            mask_rec = torch.tensor(mask_rec).to(device)
            window = torch.zeros_like(mask_rec, dtype=torch.long).fill_(self.ignore_index)  # window_size x window_siz
            thres, window_binary = self._otsu(sim)
            window_binary = torch.from_numpy(window_binary).to(device, torch.bool)
            window[torch.logical_and(mask_rec, window_binary)] = y_point

            # compute metrics
            y_rec = y[0, t: t + window_h, l: l + window_w]  # window_size x window_size
            self._compute_metrics(window, y_rec, y_point)

            grid[0, t: t + window_h, l: l + window_w] = window
            merged_mask[t: t + window_h, l: l + window_w] = True

        # all points should be included in merged mask
        set_mask_locs = set(zip(*np.where(mask[0].cpu().numpy())))
        set_merged_mask_locs = set(zip(*np.where(merged_mask.cpu().numpy())))
        assert len(set_mask_locs - set_merged_mask_locs) == 0, set_mask_locs - set_merged_mask_locs

        self._compute_metrics_global(grid[0], y[0], mask[0], merged_mask)
        return grid


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
    args.use_pseudo_label = True

    args.labelling_strategy = "local_sim"

    WINDOW_SIZE = 3

    dataloader = get_dataloader(args, val=False, query=True,
                                shuffle=False, batch_size=1, n_workers=args.n_workers)

    device = torch.device("cuda:0")

    torch.manual_seed(0)
    model = DeepLab(args).to(device)
    state_dict = torch.load("best_model_q0_rand.pt")
    model.load_state_dict(state_dict["model"])

    pseudo_labelling = LocalSimilarity(args)
    masks = torch.tensor(np.load("rand_q1.npy")).to(device)

    list_gt_labels = get_gt_labels(dir_annot="/scratch/shared/beegfs/gyungin/datasets/camvid/trainannot")
    gt_labels = list()
    for p in list_gt_labels:
        gt_labels.append(np.array(Image.open(p)))
    gt_labels = np.array(gt_labels)

    masked_labels = torch.tensor(gt_labels) * masks.cpu()

    model.eval()
    for window_size in range(255, 259, 2):
        with torch.no_grad():
            list_plabels = list()
            for batch_ind, dict_data in tqdm(enumerate(dataloader)):
                x, y = dict_data['x'].to(device), dict_data['y'].to(device)
                b = x.shape[0]
                mask = masks[batch_ind].unsqueeze(dim=0)

                dict_output = model(x)
                emb, pred = dict_output["emb"], dict_output["pred"]

                plabel = pseudo_labelling(emb, y, mask, window_size=window_size)
                list_plabels.append(plabel)

        pseudo_labelling.print()
        pseudo_labelling.write(fp=f"{window_size}_partition_{args.labelling_strategy}.txt")
