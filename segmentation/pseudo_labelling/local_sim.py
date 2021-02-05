import os
from glob import glob
from sys import path
path.append("../")
import numpy as np
import matplotlib.pyplot as plt
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
        print(ind_points)
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
        f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)

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

                f.write(
                    f"{k}, {v['prec'].avg}, {v['recall'].avg}, {v['f1_score'].avg}, {v['prec_g'].avg}, {v['recall_g'].avg}, {v['f1_score_g'].avg}\n")

            f.write(
                f"mean, {np.mean(list_prec[:-1])}, {np.mean(list_recall[:-1])}, {np.mean(list_f1[:-1])}, {np.mean(list_prec_g[:-1])}, {np.mean(list_recall_g[:-1])}, {np.mean(list_f1_g[:-1])}\n")
            f.close()

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

    def _baseline(self, window_grid, point_label):
        window_grid.fill_(point_label)
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
        merged_mask = torch.zeros_like(y, dtype=torch.bool)

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
            merged_mask[top: top + window_grid.shape[0], left: left + window_grid.shape[1]] = True

            # compute metrics - precision, recall, f1 score
            self._compute_metrics(window_grid, window_label, point_label)
        self._compute_metrics_global(grid, y, mask, merged_mask)

        return grid

    def __call__(self, emb, y, mask, window_size):
        emb = F.interpolate(emb, size=(self.h, self.w), mode='bilinear', align_corners=True)
        list_pseudo_labels = list()

        n_pixels_per_img = mask[0].sum()
        # window_size = window_size  # int(np.sqrt(self.h * self.w) // n_pixels_per_img + 1)

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

    # list_gt_labels = get_gt_labels(dir_annot="/scratch/shared/beegfs/gyungin/datasets/camvid/trainannot")
    # gt_labels = list()
    # for p in list_gt_labels:
    #     gt_labels.append(np.array(Image.open(p)))
    # gt_labels = np.array(gt_labels)
    #
    # masked_labels = torch.tensor(gt_labels) * masks.cpu()

    model.eval()
    for window_size in range(255, 259, 2):
        with torch.no_grad():
            for batch_ind, dict_data in tqdm(enumerate(dataloader)):
                x, y = dict_data['x'].to(device), dict_data['y'].to(device)
                b = x.shape[0]
                mask = masks[batch_ind].unsqueeze(dim=0)

                dict_output = model(x)
                emb, pred = dict_output["emb"], dict_output["pred"]

                pseudo_labelling(emb, y, mask, window_size)
        pseudo_labelling.print()
        pseudo_labelling.write(fp=f"{window_size}_{args.labelling_strategy}.txt")
