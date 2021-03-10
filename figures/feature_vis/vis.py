import os
import sys
import pickle as pkl
from glob import glob

import torch
import torchvision.transforms.functional as TF
from random import seed
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.append("../../segmentation")
from args import Arguments
from networks.deeplab import DeepLab
from utils.utils import get_dataloader, colorise_label
from utils.metrics import AverageMeter, RunningScore
from utils.utils import get_dataloader, get_model, get_optimizer, get_lr_scheduler

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


def val(dataloader_val, model, epoch, debug=False):
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

    print("Epoch {:d} | mIoU: {:.3f}".format(epoch, miou))
    return scores


if __name__ == '__main__':
    WINDOW_SIZE = 3
    DATASET_NAME = "voc"

    # set options
    args = Arguments().parse_args()
    args.network_name = "FPN"
    args.weight_type = "supervised"  # "moco_v2"  # "supervised"
    args.n_layers = 101

    torch.backends.cudnn.benchmark = True

    # set seeds
    seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda:0")

    # set dataloaders
    dataloader_val = get_dataloader(args, val=True, query=False, shuffle=False, batch_size=1, n_workers=args.n_workers)

    import torch.nn as nn
    # query
    model = get_model(args).encoder.to(device)
    model.eval()

    list_imgs = sorted(glob(f"{DATASET_NAME}_examples/*.{'jpg' if DATASET_NAME == 'voc' else 'png'}"))

    with torch.no_grad():
        for p in list_imgs:
            fname = p.split('/')[-1].split('.')[0]
            img = Image.open(p)
            x = TF.to_tensor(img)
            h, w = x.shape[1:]
            if DATASET_NAME == "voc":
                x = TF.normalize(x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            elif DATASET_NAME == "cv":
                x = TF.normalize(x,
                                 mean=[0.41189489566336, 0.4251328133025, 0.4326707089857],
                                 std=[0.27413549931506, 0.28506257482912, 0.28284674400252])

            x = x.unsqueeze(dim=0).to(device)
            o = model(x)[-1].squeeze()
            # o = F.interpolate(o, size=(h, w), mode="bilinear", align_corners=True).squeeze()

            if WINDOW_SIZE == 1:
                o = o[:, 1:-1, 1:-1].sum(dim=0, keepdim=True)

                c = o[:, 1:-1, 1:-1]
                l = o[:, 1:-1, :-2]
                r = o[:, 1:-1, 2:]
                t = o[:, :-2, 1:-1]
                b = o[:, 2:, 1:-1]

                t_l = o[:, :-2, :-2]
                t_r = o[:, :-2, 2:]
                b_l = o[:, 2:, :-2]
                b_r = o[:, 2:, 2:]

                # avg = (l + r + t + b + t_l + t_r + b_l + b_r) / 8
                # grid += ((c - avg) ** 2).sum(dim=0).sqrt()
                # plt.imshow(grid.cpu().numpy())
                # plt.show()
                # exit(12)
                grid = torch.zeros(c.shape[1:], dtype=torch.float32, device=device)
                for g in [l, r, t, b, t_l, t_r, b_l, b_r]:
                    grid += ((c - g) ** 2).sum(dim=0).sqrt()  # torch.linalg.norm(c - g, ord=2, dim=0)
                plt.imshow(grid.cpu().numpy())
                plt.show()
                exit(12)
                    # print(torch.linalg.norm(c - g, ord=2, dim=0).shape, c.shape, g.shape)
                    # exit(12)

                # h_grid, w_grid = d_map.shape
                # grid = torch.zeros((h_grid, w_grid), dtype=torch.float32, device=device)
                # for i in range(h_grid):
                #     for j in range(w_grid):
                #         try:
                #             grid[i, j] = d_map[i, j]
                #         except IndexError:
                #             print(i, j, grid.shape, d_map.shape, WINDOW_SIZE)
                #             raise IndexError

            # h_grid, w_grid = c.shape[1:]
            # grid = torch.zeros((h_grid - WINDOW_SIZE, w_grid - WINDOW_SIZE), dtype=torch.float32, device=device)
            # for ind, g in tqdm(enumerate([l, r, t, b, t_l, t_r, b_l, b_r])):
            #     for i in range(h_grid - WINDOW_SIZE):
            #         for j in range(w_grid - WINDOW_SIZE):
            #             try:
            #                 grid[i, j] = ((c - g)[:, i + WINDOW_SIZE, j + WINDOW_SIZE] ** 2).sum().sqrt()
            #             except IndexError:
            #                 print(i, j, grid.shape, (c - g).shape, WINDOW_SIZE)
            #                 raise IndexError
            # grid /= 8

            # d_map = torch.zeros(c.shape[1:], dtype=torch.float32, device=device)
            # for g in [l, r, t, b, t_l, t_r, b_l, b_r]:
            #     d_map += torch.linalg.norm(c - g, ord=2, dim=0)
            #
            # h_grid, w_grid = d_map.shape
            # grid = torch.zeros((h_grid - WINDOW_SIZE, w_grid - WINDOW_SIZE), dtype=torch.float32, device=device)
            # for i in range(h_grid - WINDOW_SIZE):
            #     for j in range(w_grid - WINDOW_SIZE):
            #         try:
            #             grid[i, j] = d_map[i + WINDOW_SIZE, j + WINDOW_SIZE].mean()
            #         except IndexError:
            #             print(i, j, grid.shape, d_map.shape, WINDOW_SIZE)
            #             raise IndexError

            else:
                WINDOW_SIZE_ = WINDOW_SIZE + 1
                features = F.pad(o,
                                 pad=[WINDOW_SIZE_ // 2, WINDOW_SIZE_ // 2, WINDOW_SIZE_ // 2, WINDOW_SIZE_ // 2])
                print("padded features:", features.shape)

                grid = torch.zeros(size=(h, w)).to(device)
                print("grid:", grid.shape)

                halfWindow = WINDOW_SIZE_ // 2 - 1  # 10
                offset = [halfWindow + 1, halfWindow + 1]
                print("offset:", offset)

                for j in tqdm(range(h)):
                    indH = offset[0] + j
                    for i in range(w):
                        indW = offset[1] + i
                        c_t, c_b, c_l, c_r = indH - halfWindow, indH + halfWindow + 1, indW - halfWindow, indW + halfWindow + 1
                        c = features[:, c_t: c_b, c_l: c_r]
                        # print("center:", c.shape)
                        t = features[:, c_t - 1: c_b - 1, c_l: c_r]
                        # print("top:", t.shape)
                        b = features[:, c_t + 1: c_b + 1, c_l: c_r]
                        # print("bottom:", b.shape)
                        l = features[:, c_t: c_b, c_l - 1: c_r - 1]
                        # print("left:", l.shape)
                        r = features[:, c_t: c_b, c_l + 1: c_r + 1]
                        # print("bottom:", r.shape)

                        t_l = features[:, c_t - 1: c_b - 1, c_l - 1: c_r - 1]
                        # print("top_left:", t_l.shape)
                        t_r = features[:, c_t - 1: c_b - 1, c_l + 1: c_r + 1]
                        # print("top_right:", t_r.shape)
                        b_l = features[:, c_t + 1: c_b + 1, c_l - 1: c_r - 1]
                        # print("bottom_left:", b_l.shape)
                        b_r = features[:, c_t + 1: c_b + 1, c_l + 1: c_r + 1]
                        # print("bottom_right:", b_r.shape)

                        listPatches = [c, t, b, l, r, t_l, t_r, b_l, b_r]
                        for patch in listPatches:
                            assert patch.shape == torch.Size([2048, WINDOW_SIZE, WINDOW_SIZE]), patch.shape
                        l2Sum = 0
                        for ind in range(1, 9):
                            l2 = torch.sqrt(((listPatches[0] - listPatches[ind]) ** 2).sum())
                            # print("l2:", l2.item())
                            l2Sum += l2
                        l2Avg = l2Sum / 8
                        # print("avg l2:", l2Avg)
                        grid[j, i] = l2Avg
            #     grid = grid.cpu().numpy()
            #     mean, std = np.mean(grid), np.std(grid)
            #     print("mean, std:", mean, std)
            #     grid -= mean
            #     grid = grid / (std + 1e-5)
            #     grid = 1 / (1 + np.exp(-grid))  # sigmoid

            # grid /= 8
            # print(grid.min(), grid.max(), grid.mean(), grid.std())
            # grid = -grid.cpu().numpy()
            #
            grid -= grid.min()
            grid /= grid.max()
            # grid -= grid.mean()
            # grid = grid / (grid.std() + 1e-5)
            # grid = torch.sigmoid(grid)
            # plt.imshow(grid.cpu().numpy(), cmap="gray", interpolation="antialiased")
            # plt.show()
            # exit(12)
            grid *= 255.0

            grid = np.clip(grid.cpu().numpy(), 0, 255)

            grid = grid.astype(np.uint8)
            grid = 255 - grid

            img = Image.fromarray(grid)
            img.resize((w, h)).save(f"{fname}_{args.weight_type}_{WINDOW_SIZE}.png")
            # grid = 255 - grid  # to make dark region indicate high average distance
            # Image.fromarray(grid).save(f"{fname}_{args.weight_type}_{WINDOW_SIZE}.png")
    # model = val(dataloader_val, model, 0,  debug=args.debug)
