if __name__ == '__main__':
    import os
    import sys
    sys.path.append("../")
    import torch
    from glob import glob
    from random import seed
    import numpy as np
    import torch
    import torch.nn.functional as F
    import cv2
    from PIL import Image
    from tqdm import tqdm
    from args import Arguments
    from utils.utils import get_dataloader, get_optimizer, get_lr_scheduler
    from utils.metrics import AverageMeter, eval_metrics

    from networks.deeplab import DeepLab
    from local_sim import LocalSimilarity

    # baseline model records
    list_val = list()
    from csv import reader
    with open("cv_aug_deeplab_sm_random_10_0/1_query/log_val.txt", 'r') as f:
        csv_reader = reader(f, delimiter=',')
        for i, line in enumerate(csv_reader):
            if i == 0:
                continue
            list_val.append(line[1])
        f.close()

    # set seeds
    seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    # set options
    args = Arguments().parse_args()
    args.use_softmax = True
    args.query_strategy = "random"
    args.n_pixels_by_us = 10
    args.use_pseudo_label = True
    args.labelling_strategy = "local_sim"

    device = torch.device("cuda:0")

    # set window size for pseudo-labelling
    WINDOW_SIZE = 7
    MODEL = "ms"

    # set dataloaders
    dataloader_train = get_dataloader(args, val=False, query=False,
                                      shuffle=True, batch_size=4, n_workers=args.n_workers)

    dataloader_val = get_dataloader(args, val=True, query=False,
                                    shuffle=False, batch_size=1, n_workers=args.n_workers)

    # pseudo-labelling
    model = DeepLab(args).to(device)
    state_dict = torch.load(f"best_model_q0_{MODEL}.pt")
    model.load_state_dict(state_dict["model"])

    pseudo_labelling = LocalSimilarity(args)
    dataloader_train.dataset.arr_masks = np.load(f"{MODEL}_q1.npy")

    dataloader_train.dataset.update_pseudo_label(model, WINDOW_SIZE, nth_query=1)

    # training
    best_miou = 0
    model = DeepLab(args).to(device)  # randomly initialized model
    optimizer = get_optimizer(args, model)
    lr_scheduler = get_lr_scheduler(args, optimizer=optimizer, iters_per_epoch=len(dataloader_train))

    meter_loss = AverageMeter()
    for e in range(1, 51, 1):
        total_inter, total_union = 0, 0
        total_correct, total_label = 0, 0
        meter_miou = AverageMeter()

        dataloader_iter = iter(dataloader_train)
        tbar = tqdm(range(len(dataloader_train)))
        model.train()
        for batch_ind in tbar:
            dict_data = next(dataloader_iter)
            x, y = dict_data['x'].to(device), dict_data['y'].to(device)
            y_pseudo = dict_data['y_pseudo'].to(device)
            mask = dict_data['mask'].to(device, torch.bool)

            y_flat = y.flatten()
            y_flat[~mask.flatten()] = args.ignore_index
            y = y_flat.view(args.batch_size, 360, 480)

            # y_pseudo_flat = y_pseudo.flatten()
            # y_pseudo_flat[mask.flatten()] = args.ignore_index
            # y_pseudo = y_pseudo_flat.view(args.batch_size, 360, 480)

            # mask_flat = mask.flatten()
            # labelled_points = mask_flat  # torch.where(mask_flat)[0]
            #
            # y_flat = y.flatten()
            # mask_pseudo_flat = (y_flat != args.ignore_index)
            # mask_pseudo_flat = torch.logical_xor(mask_flat, mask_pseudo_flat)

            dict_output = model(x)

            logits = dict_output["pred"]
            pred = logits.argmax(dim=1)
            loss_sup = F.cross_entropy(logits, y, ignore_index=args.ignore_index, reduction='mean')
            loss_pseudo = F.cross_entropy(logits, y_pseudo, ignore_index=args.ignore_index, reduction='mean')

            loss = loss_sup + loss_pseudo

            # loss = F.cross_entropy(logits, y, ignore_index=args.ignore_index, reduction='none')
            # loss_flat = loss.flatten()

            # loss_label = loss_flat[mask_flat].sum()
            # loss_pseudo_label = loss_flat[mask_pseudo_flat].sum()
            #
            # loss = (loss_label + loss_pseudo_label * 0.1) / (mask_flat.sum() + mask_pseudo_flat.sum())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr_scheduler.step(epoch=e - 1)

            correct, labeled, inter, union = eval_metrics(pred, y, args.n_classes, args.ignore_index)

            total_inter, total_union = total_inter + inter, total_union + union
            total_correct, total_label = total_correct + correct, total_label + labeled

            pix_acc = 1.0 * total_correct / (np.spacing(1) + total_label)
            IoU = 1.0 * total_inter / (np.spacing(1) + total_union)
            mIoU = IoU.mean()

            meter_miou.update(mIoU)
            tbar.set_description("Epoch {:d} | mIoU: {:.3f}".format(e, meter_miou.avg))

        # validation
        total_inter, total_union = 0, 0
        total_correct, total_label = 0, 0
        meter_miou = AverageMeter()

        model.eval()
        with torch.no_grad():
            for batch_ind, dict_data in tqdm(enumerate(dataloader_val)):
                x, y = dict_data['x'].to(device), dict_data['y'].to(device)

                dict_output = model(x)
                emb, pred = dict_output["emb"], dict_output["pred"].argmax(dim=1)

                correct, labeled, inter, union = eval_metrics(pred, y, args.n_classes, args.ignore_index)

                total_inter, total_union = total_inter + inter, total_union + union
                total_correct, total_label = total_correct + correct, total_label + labeled

                pix_acc = 1.0 * total_correct / (np.spacing(1) + total_label)
                IoU = 1.0 * total_inter / (np.spacing(1) + total_union)
                mIoU = IoU.mean()

                meter_miou.update(mIoU)

            if meter_miou.avg > best_miou:
                best_miou = meter_miou.avg

        delta = float(list_val[e - 1]) - meter_miou.avg  # gap between baseline model and the current model
        print("Epoch {:d} | mIoU: {:.3f} | best mIoU: {:.3f} | delta {:.3f}"
              .format(e, meter_miou.avg, best_miou, delta))