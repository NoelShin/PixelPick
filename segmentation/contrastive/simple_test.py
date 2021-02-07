if __name__ == '__main__':
    import os
    import sys
    sys.path.append("../")
    import torch
    from random import seed
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from tqdm import tqdm
    from args import Arguments
    from utils.utils import get_dataloader, get_optimizer, get_lr_scheduler
    from utils.metrics import AverageMeter, eval_metrics
    from criterions.contrastive_loss import get_dict_label_projection, compute_contrastive_loss

    from networks.deeplab import DeepLab

    NTH_QUERY = 9
    MODEL = "margin_sampling"
    DIR = f"cv_aug_deeplab_sm_{MODEL}_n_pixels_10_0"

    # baseline model records
    list_val = list()
    from csv import reader
    with open(f"{DIR}_{NTH_QUERY}_query/log_val.txt", 'r') as f:
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

    with open(f"{args.experim_name}.txt", 'w') as f:
        f.close()

    device = torch.device("cuda:0")

    # set window size for pseudo-labelling
    MODEL = "rand"

    # set dataloaders
    dataloader_train = get_dataloader(args, val=False, query=False,
                                      shuffle=True, batch_size=4, n_workers=args.n_workers)
    arr_masks = np.load(f"{DIR}_{NTH_QUERY}_query/label.npy")
    dataloader_train.dataset.arr_masks = arr_masks
    print("n_labels:", arr_masks.sum())

    dataloader_query = get_dataloader(args, val=False, query=True,
                                      shuffle=False, batch_size=1, n_workers=args.n_workers)
    dataloader_query.dataset.arr_masks = arr_masks

    dataloader_val = get_dataloader(args, val=True, query=False,
                                    shuffle=False, batch_size=1, n_workers=args.n_workers)

    # training
    best_miou = 0
    model = DeepLab(args).to(device)  # randomly initialized model
    dict_label_projection = get_dict_label_projection(dataloader_query, model,
                                                      arr_masks=arr_masks,
                                                      ignore_index=args.ignore_index,
                                                      region_contrast=args.use_region_contrast)

    if NTH_QUERY > 0:
        state_dict = torch.load(f"{DIR}_{NTH_QUERY - 1}_query/best_miou_model.pt")
        model.load_state_dict(state_dict["model"], strict=False)

    # dict_label_projection = get_dict_label_projection(dataloader_query, model,
    #                                                   ignore_index=args.ignore_index,
    #                                                   region_contrast=args.use_region_contrast)

    if NTH_QUERY > 0:
        model = DeepLab(args).to(device)  # randomly initialized model
    optimizer = get_optimizer(args, model)
    lr_scheduler = get_lr_scheduler(args, optimizer=optimizer, iters_per_epoch=len(dataloader_train))

    for e in range(1, 51, 1):
        total_inter, total_union = 0, 0
        total_correct, total_label = 0, 0
        meter_miou = AverageMeter()
        meter_loss_cont = AverageMeter()
        meter_loss_ce = AverageMeter()

        dataloader_iter = iter(dataloader_train)
        tbar = tqdm(range(len(dataloader_train)))
        model.train()
        for batch_ind in tbar:
            dict_data = next(dataloader_iter)
            x, y = dict_data['x'].to(device), dict_data['y'].to(device)
            mask = dict_data['mask'].to(device, torch.bool)

            y_flat = y.flatten()
            y_flat[~mask.flatten()] = args.ignore_index
            y = y_flat.view(args.batch_size, 360, 480)

            dict_output = model(x)

            projection, logits = dict_output['projection'], dict_output["pred"]

            pred = logits.argmax(dim=1)
            loss_sup = F.cross_entropy(logits, y, ignore_index=args.ignore_index, reduction='mean')

            loss_cont = args.w_contrastive * compute_contrastive_loss(projection, y, mask,
                                                                      dict_label_projection=dict_label_projection,
                                                                      selection_mode=args.selection_mode,
                                                                      temperature=args.temperature)

            loss = loss_sup + loss_cont

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr_scheduler.step(epoch=e-1)

            correct, labeled, inter, union = eval_metrics(pred, y, args.n_classes, args.ignore_index)

            total_inter, total_union = total_inter + inter, total_union + union
            total_correct, total_label = total_correct + correct, total_label + labeled

            pix_acc = 1.0 * total_correct / (np.spacing(1) + total_label)
            IoU = 1.0 * total_inter / (np.spacing(1) + total_union)
            mIoU = IoU.mean()

            meter_miou.update(mIoU)
            meter_loss_cont.update(loss_cont.detach().cpu().item())
            meter_loss_ce.update(loss_sup.detach().cpu().item())
            tbar.set_description("Epoch {:d} | loss_ce: {:.3f} | loss_cont: {:.3f} | mIoU: {:.3f}"
                                 .format(e, meter_loss_ce.avg, meter_loss_cont.avg, meter_miou.avg))

        # validation
        total_inter, total_union = 0, 0
        total_correct, total_label = 0, 0
        meter_miou = AverageMeter()

        model.eval()
        with torch.no_grad():
            for batch_ind, dict_data in tqdm(enumerate(dataloader_val)):
                x, y = dict_data['x'].to(device), dict_data['y'].to(device)

                dict_output = model(x)
                pred = dict_output["pred"].argmax(dim=1)

                correct, labeled, inter, union = eval_metrics(pred, y, args.n_classes, args.ignore_index)

                total_inter, total_union = total_inter + inter, total_union + union
                total_correct, total_label = total_correct + correct, total_label + labeled

                pix_acc = 1.0 * total_correct / (np.spacing(1) + total_label)
                IoU = 1.0 * total_inter / (np.spacing(1) + total_union)
                mIoU = IoU.mean()

                meter_miou.update(mIoU)

            if meter_miou.avg > best_miou:
                best_miou = meter_miou.avg

        delta = float(list_val[e-1]) - meter_miou.avg  # gap between baseline model and the current model
        print("Epoch {:d} | mIoU: {:.3f} | best mIoU: {:.3f} | delta {:.3f}"
              .format(e, meter_miou.avg, best_miou, delta))

        with open(f"{MODEL}_{args.experim_name}.txt", 'a') as f:
            f.write(f"{meter_miou.avg}\n")
            f.close()