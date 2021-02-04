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
                                       'f1_score': AverageMeter()} for c in range(12)}

    def _compute_metrics_global(self, grid, label, mask):
        set_unique_labels = set((label * mask).cpu().numpy().flatten())
        print(set_unique_labels)
        for ul in set_unique_labels:
            TP_FP = (grid == ul)
            TP = torch.logical_and(grid == ul, label == grid)

        TP = (grid == label).sum()



        return

    def _partition(self, points):
        ind_points = np.where(points)
        # print(ind_points)
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

    def write(self, fp):
        with open(fp, 'w') as f:
            list_prec, list_recall, list_f1 = list(), list(), list()
            for k, v in self.dict_class_metrics.items():
                list_prec.append(v["prec"].avg)
                list_recall.append(v["recall"].avg)
                list_f1.append(v["f1_score"].avg)
                f.write(f"{k}, {v['prec'].avg}, {v['recall'].avg}, {v['f1_score'].avg}\n")
            f.write(f"mean, {np.mean(list_prec)}, {np.mean(list_recall)}, {np.mean(list_f1)}\n")
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
            return thres

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

    def __call__(self, emb, y, mask, window_size):
        emb = F.interpolate(emb, size=(self.h, self.w), mode='bilinear', align_corners=True)

        grid = torch.zeros_like(y).fill_(self.ignore_index)

        partitions, ind_points = self._partition(mask[0].cpu().numpy())
        for p in range(partitions.max()):
            emb_point = emb[0, :, ind_points[0][p], ind_points[1][p]]  # 256
            y_point = y[0, ind_points[0][p], ind_points[1][p]]
            mask_p = (partitions == p)  # h x w

            mask_rec = np.where(mask_p)
            t, b, l, r = mask_rec[0].min(), mask_rec[0].max(), mask_rec[1].min(), mask_rec[1].max()
            mask_rec = mask_p[t: b + 1, l: r + 1]  # h_p x w_p

            emb_rec = emb[0, :, t: b + 1, l: r + 1]  # 256 x h_p x w_p
            emb_point = emb_point.unsqueeze(1).unsqueeze(2).repeat(1, emb_rec.shape[1], emb_rec.shape[2])
            sim = F.cosine_similarity(emb_rec, emb_point, dim=0)  # h_p x w_p
            sim_p = sim.flatten()[mask_rec.flatten()]
            thres = self._otsu(sim_p)
            thres = torch.tensor(thres).to(device)
            # print(sim_p.shape, thres)

            sim = torch.tanh(sim)  # using tanh gives a slightly better f1_score
            sim -= sim.min()
            sim = sim / sim.max()
            sim *= 255

            # print(mask_rec.shape, sim.shape)
            mask_rec = torch.tensor(mask_rec).to(device)
            window = torch.zeros_like(mask_rec, dtype=torch.long).fill_(self.ignore_index)  # h_p x w_p
            window[torch.logical_and(mask_rec, sim > thres)] = y_point

            # compute metrics
            y_rec = y[0, t: b + 1, l: r + 1]  # h_p x w_p
            y_rec_flat = y_rec.flatten()
            y_rec_flat[~mask_rec.flatten()] = self.ignore_index
            y_rec = y_rec_flat.reshape(b - t + 1, r - l + 1)
            # y = colorise_label(y_rec_flat.reshape(b - t + 1, r - l + 1).cpu().numpy())
            # Image.fromarray(np.transpose(y, (1, 2, 0))).show()
            # exit(12)

            TP_FP = (window == y_point).sum()
            TP_mask = torch.logical_and(window == y_point, y_rec == y_point)
            TP = TP_mask.sum()

            TN_FN_mask = torch.logical_and(window != y_point, y_rec != self.ignore_index)

            FN_mask = torch.logical_and(TN_FN_mask, TP_mask)
            FN = FN_mask.sum()

            prec = TP / TP_FP
            recall = TP / (TP + FN)
            f1 = 2 * (prec * recall) / (prec + recall)

            self.dict_class_metrics[y_point.cpu().numpy().item()]["prec"].update(prec.cpu().numpy())
            self.dict_class_metrics[y_point.cpu().numpy().item()]["recall"].update(recall.cpu().numpy())
            self.dict_class_metrics[y_point.cpu().numpy().item()]["f1_score"].update(f1.cpu().numpy())

            grid[0, t: b + 1, l: r + 1] = window

            # print(grid)
            # a = colorise_label(grid[0].cpu().numpy())
            # Image.fromarray(np.transpose(a, (1, 2, 0))).show()
            # y = y[0].cpu().numpy()

            # y_flat = y.flatten()
            # y_flat[~mask_p.flatten()] = self.ignore_index
            # y = colorise_label(y_flat.reshape(360, 480))
            # Image.fromarray(np.transpose(y, (1, 2, 0))).show()
            # exit(12)

            # emb_p = emb[:, :, torch.tensor(mask_p).to(device)].view(c, -1)
            # emb_point = emb_point.unsqueeze(1).repeat(1, emb_p.shape[1])
            # sim = F.cosine_similarity(emb_p, emb_point, dim=0).unsqueeze(dim=0)  # to make it a binary image

            # hist = torch.histc(sim, bins=255)
            # import matplotlib.pyplot as plt
            # plt.bar(range(255), hist.cpu().numpy(), width=1)
            # plt.show()
            # plt.close()

            # y_p = y.flatten()[torch.tensor(mask_p).flatten().to(device)]
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

    args.labelling_strategy = "baseline"

    WINDOW_SIZE = 3

    dataloader = get_dataloader(args, val=False, query=True,
                                shuffle=False, batch_size=1, n_workers=args.n_workers)

    device = torch.device("cuda:0")

    torch.manual_seed(0)
    model = DeepLab(args).to(device)
    state_dict = torch.load("best_model_q0.pt")
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
    with torch.no_grad():
        list_plabels = list()
        for batch_ind, dict_data in tqdm(enumerate(dataloader)):
            x, y = dict_data['x'].to(device), dict_data['y'].to(device)
            b = x.shape[0]
            mask = masks[batch_ind].unsqueeze(dim=0)
            # * args.batch_size: (batch_ind + 1) * args.batch_size]

            dict_output = model(x)
            emb, pred = dict_output["emb"], dict_output["pred"]

            plabel = pseudo_labelling(emb, y, mask, None)
            list_plabels.append(plabel)

    pseudo_labelling.print()
    pseudo_labelling.write(fp=f"partition_{args.labelling_strategy}.txt")
