import os
from math import ceil

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from networks.model import FPNSeg
from networks.modules import init_prototypes, init_radii, EMA
from utils.metrics import prediction, eval_metrics
from utils.utils import get_criterion, get_optimizer, get_lr_scheduler, get_validator, AverageMeter, EmbeddingVisualiser
from utils.utils import write_log, send_file, Visualiser, zip_dir


def main(args):
    torch.manual_seed(args.seed)

    device = torch.device("cuda:{:s}".format(args.gpu_ids) if torch.cuda.is_available() else "cpu:0")
    torch.backends.cudnn.benchmark = True

    if args.dataset_name == "cv":
        from datasets.camvid import CamVidDataset
        dataset = CamVidDataset(args)

    elif args.dataset_name == "voc":
        from datasets.voc import VOC2012Segmentation
        dataset = VOC2012Segmentation(args)

    else:
        raise ValueError(args.dataset_name)

    dataloader = DataLoader(dataset,
                            batch_size=args.batch_size,
                            num_workers=args.n_workers,
                            shuffle=True,
                            drop_last=True)

    model = FPNSeg(args).to(device)
    print("\nExperim name:", args.experim_name + '\n')
    if not args.use_softmax:
        prototypes = init_prototypes(args.n_classes,
                                     args.n_emb_dims,
                                     args.n_prototypes,
                                     mode='mean',
                                     model=model,
                                     dataset=dataset,
                                     ignore_index=args.ignore_index,
                                     learnable=args.model_name == "gcpl_seg",
                                     device=device)

    if args.model_name == "mp_seg":
        prototypes_updater = EMA(args.momentum_prototypes)

    criterion = get_criterion(args, device)
    optimizer = get_optimizer(args, model, prototypes=prototypes if not args.use_softmax else None)
    lr_scheduler = get_lr_scheduler(args, optimizer=optimizer, iters_per_epoch=len(dataloader))
    validator = get_validator(args)
    vis = Visualiser(args.dataset_name)

    running_loss, running_miou, running_pixel_acc = AverageMeter(), AverageMeter(), AverageMeter()

    for e in range(1, args.n_epochs + 1):
        model.train()

        dataloader_iter = iter(dataloader)
        tbar = tqdm(range(len(dataloader)))

        if args.active_learning:
            dict_img_uncertain_pix = dict()

        for batch_ind in tbar:
            total_inter, total_union = 0, 0
            total_correct, total_label = 0, 0

            dict_data = next(dataloader_iter)
            x, y = dict_data['x'].to(device), dict_data['y'].to(device)
            if args.n_pixels_per_img != 0:
                mask = dict_data['mask'].to(device, torch.bool)
                y.flatten()[~mask.flatten()] = args.ignore_index

            dict_outputs = model(x)

            if args.use_softmax:
                confidence = dict_outputs["pred"]
                dict_losses = {"ce": F.cross_entropy(confidence, y, ignore_index=args.ignore_index)}
                pred = confidence.argmax(dim=1)  # for computing mIoU, pixel acc.
                confidence = confidence.max(dim=1)[0]

            else:
                emb = dict_outputs['emb']  # b x n_emb_dims x h x w

                b, c, h, w = emb.shape
                emb_flatten = emb.transpose(1, 0)  # c x b x h x w
                emb_flatten = emb_flatten.contiguous().view(c, b * h * w)  # c x (b * h * w)
                emb_flatten = emb_flatten.transpose(1, 0)  # (b * h * w) x c
                y_flatten = y.flatten()  # (b * h * w)

                unique_labels_batch = set(sorted(y_flatten.cpu().numpy().tolist())) - {args.ignore_index}

                dict_label_emb = dict()
                for label in unique_labels_batch:
                    ind_label = (y_flatten == label)
                    emb_label = emb_flatten[ind_label]
                    dict_label_emb.update({label: emb_label})

                dict_outputs.update({"dict_label_emb": dict_label_emb})
                pred, dist = prediction(emb.detach(), prototypes.detach(), return_distance=True)
                confidence = F.softmax(-dist, dim=1).max(dim=1)[0]

                dict_losses = criterion(dict_outputs, prototypes, labels=y)

            loss = torch.tensor(0, dtype=torch.float32).to(device)

            fmt = "Epoch {:d} | mIoU.: {:.3f} | pixel acc.: {:.3f} | Loss: {:.3f}"
            for loss_k, loss_v in dict_losses.items():
                fmt += " | {:s}: {:.3f}".format(loss_k, loss_v.detach().cpu().item())
                loss += loss_v

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss.update(loss.detach().item())
            correct, labeled, inter, union = eval_metrics(pred, y, args.n_classes, args.ignore_index)

            total_inter, total_union = total_inter + inter, total_union + union
            total_correct, total_label = total_correct + correct, total_label + labeled

            pix_acc = 1.0 * total_correct / (np.spacing(1) + total_label)
            IoU = 1.0 * total_inter / (np.spacing(1) + total_union)
            mIoU = IoU.mean()

            running_miou.update(mIoU)
            running_pixel_acc.update(pix_acc)

            tbar.set_description(fmt.format(e, running_miou.avg, running_pixel_acc.avg, running_loss.avg))

            if args.active_learning:
                for ind in dict_data["ind"]:
                    list_uncertain_pix = confidence[ind].flatten().topk(k=args.n_pixels_per_add_labels, largest=False)[1]  # n_pixels x (h * w)
                    dict_img_uncertain_pix.update({ind.item(): list_uncertain_pix})

                if e % args.epoch_add_labels == 0 and e < args.n_add_labels:
                    dataset.add_labels(dict_img_uncertain_pix)

            if args.debug:
                break

        lr_scheduler.step(e)

        write_log(args.log_train, list_entities=[e, running_miou.avg, running_pixel_acc.avg, running_loss.avg])

        running_loss.reset()
        running_miou.reset()

        dict_tensors = {'input': dict_data['x'][0].cpu(),
                        'target': dict_data['y'][0].cpu(),
                        'pred': pred[0].detach().cpu(),
                        'confidence': -confidence[0].detach().cpu()}  # minus sign is to make (un)certain part dark (bright)

        vis(dict_tensors, fp=f"{args.dir_checkpoints}/train/{e}.png")

        if e % 5 == 0:
            validator(model, e, prototypes=prototypes if not args.use_softmax else None)

        if args.debug:
            break

    zip_file = zip_dir(args.dir_checkpoints)
    # send_file(zip_file)


if __name__ == '__main__':
    from args import Arguments
    args = Arguments().parse_args()
    main(args)
