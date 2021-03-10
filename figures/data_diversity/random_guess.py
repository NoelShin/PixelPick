import os
import sys
import pickle as pkl
from glob import glob

import torch
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
    # set options
    args = Arguments().parse_args()

    torch.backends.cudnn.benchmark = True

    # set seeds
    seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda:0")

    # set dataloaders
    dataloader_val = get_dataloader(args, val=True, query=False, shuffle=False, batch_size=1, n_workers=args.n_workers)

    # query
    model = DeepLab(args).to(device)
    model = val(dataloader_val, model, 0,  debug=args.debug)