import os
import sys
import pickle as pkl
from glob import glob
from copy import deepcopy

import torch
from random import seed
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.append("../")
from args import Arguments
from networks.deeplab import DeepLab
from utils.utils import get_dataloader, colorise_label, get_model
from utils.metrics import AverageMeter, RunningScore
from utils.utils import get_dataloader, get_optimizer, get_lr_scheduler

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


class QuerySelector:
    def __init__(self, ignore_index, gt_distribution=None, device=torch.device("cuda:0")):
        self.device = device
        self.gt_dist = gt_distribution
        self.ignore_index = ignore_index

    def __call__(self, dataloader_query, model, is_class_blance=False):
        print("Selecting queries...")
        masks = dataloader_query.dataset.arr_masks

        list_grids, list_wrong_pred, list_uc_map = list(), list(), list()
        model.eval()
        with torch.no_grad():
            for batch_ind, dict_data in tqdm(enumerate(dataloader_query)):
                x, gt = dict_data['x'].to(self.device), dict_data['y'].to(self.device).squeeze()
                h, w = x.shape[-2:]
                mask, mask_void = masks[batch_ind], (gt == self.ignore_index)

                dict_outputs = model(x)
                logits = dict_outputs["pred"].squeeze()
                prob = F.softmax(logits, dim=0)

                uc = torch.sub(*prob.topk(k=2, dim=0, largest=True).values)
                uc = torch.exp(-uc)

                thres = uc.flatten().topk(k=int(h * w * 0.05), largest=True).values[-1]
                uc = (uc > thres)

                gt_flat = gt.flatten()

                grid_cand = torch.zeros_like(gt_flat, dtype=torch.bool)  # (h * w)
                grid_cand[uc.flatten()] = True
                grid_cand[mask.flatten()] = False
                grid_cand[mask_void.flatten()] = False

                grid = np.zeros_like(gt_flat.cpu().numpy(), dtype=np.bool)  # (h * w)
                grid[np.random.choice(np.where(grid_cand.cpu().numpy())[0], 1, False)] = True
                list_grids.append(grid.reshape(h, w))

        queries = np.stack(list_grids, axis=0)
        assert queries.sum() <= 2975, queries.sum()

        return queries


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


def val(dataloader_val, model, epoch, experim_name, debug=False):
    running_score = RunningScore(args.n_classes)

    model.eval()
    with torch.no_grad():
        for batch_ind, dict_data in tqdm(enumerate(dataloader_val)):
            x, y = dict_data['x'].to(device), dict_data['y']

            dict_output = model(x)
            emb, pred = dict_output["emb"], dict_output["pred"].argmax(dim=1)

            running_score.update(y.squeeze().numpy(), pred.squeeze().cpu().numpy())

            if debug:
                break

        scores = running_score.get_scores()
        miou = scores[0]["Mean IoU"]

    with open(f"{experim_name}/log_val.txt", 'a') as f:
        f.write(f"{miou}\n")
        f.close()

    print("Epoch {:d} | mIoU: {:.3f}".format(epoch, miou))
    return scores


# training
def train_stage(dataloader_train, dataloader_val, experim_name, debug=False):
    model = get_model(args).to(device)
    running_score = RunningScore(args.n_classes)

    with open(f"{experim_name}/log_val.txt", 'w') as f:
        f.close()

    # training
    optimizer = get_optimizer(args, model)
    lr_scheduler = get_lr_scheduler(args, optimizer=optimizer, iters_per_epoch=len(dataloader_train))

    best_miou = 0.
    for e in range(1, 51, 1):
        # train an epoch
        dataloader_iter, tbar = iter(dataloader_train), tqdm(range(len(dataloader_train)))
        model.train()
        for _ in tbar:
            dict_data = next(dataloader_iter)
            x, y, mask = dict_data['x'].to(device), dict_data['y'].to(device), dict_data['mask'].to(device, torch.bool)
            h, w = x.shape[-2:]
            y_flat = y.flatten()
            y_flat[~mask.flatten()] = args.ignore_index
            y = y_flat.view(x.shape[0], h, w)

            dict_output = model(x)

            logits = dict_output["pred"]
            loss = F.cross_entropy(logits, y, ignore_index=args.ignore_index, reduction='mean')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step(epoch=e - 1)

            running_score.update(y.squeeze().cpu().numpy(), logits.argmax(dim=1).squeeze().cpu().numpy())
            tbar.set_description("Epoch {:d} | mIoU: {:.3f}".format(e, running_score.get_scores()[0]["Mean IoU"]))

            if debug:
                break

        scores = val(dataloader_val, model, e, experim_name)
        miou = scores[0]["Mean IoU"]
        if miou > best_miou:
            print("Best mIoU score has been renewed from {:.3f} to {:.3f}.".format(best_miou, miou))
            best_miou = miou
            state_dict = {"model": model.state_dict()}
            torch.save(state_dict, f"{experim_name}/best_model.pt")
            pkl.dump(scores, open(f"{experim_name}/scores.pkl", 'wb'))

        if debug:
            break
    return model


if __name__ == '__main__':
    MODEL_NAME = "cs_d4_aug_FPN_50_supervised_sm_margin_sampling_1_p0.05_0_9_query"

    EXPERIM_NAME = f"{MODEL_NAME}_plabel"

    # set options
    args = Arguments().parse_args()
    torch.backends.cudnn.benchmark = True

    # set seeds
    seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda:0")

    # set dataloaders
    dataloader_query = get_dataloader(args, val=False, query=True,
                                      shuffle=False, batch_size=1, n_workers=args.n_workers)
    dataloader_train = get_dataloader(args, val=False, query=False,
                                      shuffle=True, batch_size=1, n_workers=args.n_workers)
    dataloader_val = get_dataloader(args, val=True, query=False,
                                    shuffle=False, batch_size=1, n_workers=args.n_workers)

    masks = np.load(f"{MODEL_NAME}/label.npy")
    h, w = masks.shape[1:]

    list_d2_masks = list()
    for m in tqdm(masks):
        d2_m = Image.fromarray(m).resize((2 * w, 2 * h), Image.NEAREST)
        list_d2_masks.append(np.array(d2_m))

    d2_masks = np.stack(list_d2_masks, axis=0)
    dataloader_train.dataset.arr_masks = d2_masks
    dataloader_query.dataset.arr_masks = d2_masks

    # query
    if os.path.exists("d2_new_queries.npy"):
        queries = np.load("d2_new_queries.npy")

    else:
        model = get_model(args).to(device)
        state_dict = torch.load(f"{MODEL_NAME}/best_miou_model.pt")
        model.load_state_dict(state_dict["model"])

        qs = QuerySelector(args.ignore_index)

        queries = qs(dataloader_query, model)
        np.save("d2_new_queries.npy", queries)

    dataloader_query.dataset.label_queries(queries)
    dataloader_train.dataset.label_queries(queries)

    experim_name = f"{EXPERIM_NAME}_{args.seed}"
    os.makedirs(experim_name, exist_ok=True)

    model = train_stage(dataloader_train, dataloader_val, experim_name, debug=args.debug)