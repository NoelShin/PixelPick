import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.camvid import CamVidDataset
from networks.model import FPNSeg
from networks.modules import init_prototypes, init_radii, EMA
from utils.metrics import prediction, eval_metrics
from utils.utils import get_criterion, get_optimizer, get_lr_scheduler, get_validator, AverageMeter, EmbeddingVisualiser
from utils.utils import write_log, send_file, zip_dir


def main(args):
    torch.manual_seed(args.seed)

    device = torch.device("cuda:{:s}".format(args.gpu_ids) if torch.cuda.is_available() else "cpu:0")
    torch.backends.cudnn.benchmark = True

    if args.dataset_name == "cv":
        dataset = CamVidDataset(args)

    dataloader = DataLoader(dataset,
                            batch_size=args.batch_size,
                            num_workers=args.n_workers,
                            shuffle=True)

    model = FPNSeg(args).to(device)
    print("\nExperim name:", args.experim_name + '\n')
    prototypes = init_prototypes(args.n_classes, args.n_emb_dims, learnable=args.model_name == "gcpl_seg", device=device)

    if args.model_name == "mp_seg":
        prototypes_updater = EMA(args.momentum_prototypes)

    list_params = list(model.parameters())
    list_params += [prototypes] if args.model_name == "gcpl_seg" else []

    criterion = get_criterion(args)
    optimizer = get_optimizer(args, params=list_params)
    lr_scheduler = get_lr_scheduler(args, optimizer=optimizer)
    validator = get_validator(args)

    running_loss, running_miou, running_pixel_acc = AverageMeter(), AverageMeter(), AverageMeter()

    for e in range(1, args.n_epochs + 1):
        model.train()

        dataloader_iter = iter(dataloader)
        tbar = tqdm(range(len(dataloader)))
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
                pred = dict_outputs["pred"]
                dict_losses = {"ce": F.cross_entropy(pred, y, ignore_index=args.ignore_index)}
                pred = pred.argmax(dim=1)  # for computing mIoU, pixel acc.

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

                    if args.model_name == "mp_seg":
                        if len(emb_label) > 1:
                            emb_label = emb_label.mean(dim=0)
                        prototypes[label] = prototypes_updater.update_average(old=prototypes[label],
                                                                              new=emb_label.detach())

                dict_outputs.update({"dict_label_emb": dict_label_emb})
                pred = prediction(emb.detach(), prototypes.detach(), non_isotropic=args.non_isotropic)
                dict_losses = criterion(dict_outputs, prototypes, labels=y, non_isotropic=args.non_isotropic)

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
            if args.debug:
                break

        lr_scheduler.step()

        write_log(args.log_train, list_entities=[e, running_miou.avg, running_pixel_acc.avg, running_loss.avg])

        running_loss.reset()
        running_miou.reset()

        validator(model, prototypes, e)

        if args.debug:
            break

    zip_file = zip_dir(args.dir_checkpoints)
    # send_file(zip_file)


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser("")

    parser.add_argument("--debug", "-d", action="store_true", default=False)
    parser.add_argument("--dir_root", type=str, default="..")
    parser.add_argument("--seed", "-s", type=int, default=0)
    parser.add_argument("--model_name", type=str, default="gcpl_seg", choices=["gcpl_seg", "mp_seg"])
    parser.add_argument("--n_pixels_per_img", type=int, default=0)
    parser.add_argument("--use_aug", action='store_true', default=False)
    parser.add_argument("--use_softmax", action='store_true', default=False)

    # system
    parser.add_argument("--gpu_ids", type=str, nargs='+', default='0')
    parser.add_argument("--n_workers", type=int, default=4)

    # dataset
    parser.add_argument("--dataset_name", type=str, default="cv", choices=["cv"])
    parser.add_argument("--dir_datasets", type=str, default="/scratch/shared/beegfs/gyungin/datasets")

    # gcpl
    parser.add_argument("--loss_type", type=str, default="dce", choices=["dce"])
    parser.add_argument("--use_pl", action="store_true", default=False, help="prototype loss")
    parser.add_argument("--w_pl", type=float, default=0.001, help="weight for prototype loss")
    parser.add_argument("--use_repl", action="store_true", default=False, help="repulsive loss")
    parser.add_argument("--w_repl", type=float, default=1, help="weight for repulsive loss")
    parser.add_argument("--use_vl", action="store_true", default=False, help="prototype loss")
    parser.add_argument("--w_vl", type=float, default=1, help="weight for prototype loss")

    parser.add_argument("--non_isotropic", action="store_true", default=False)
    parser.add_argument("--n_emb_dims", type=int, default=32)
    # encoder
    parser.add_argument("--weight_type", type=str, default="supervised", choices=["random", "supervised", "moco_v2", "swav", "deepcluster_v2"])
    parser.add_argument("--use_dilated_resnet", type=bool, default=True, help="whether to use dilated resnet")

    args = parser.parse_args()

    if args.dataset_name == "cv":
        args.batch_size = 5
        args.dir_dataset = "/scratch/shared/beegfs/gyungin/datasets/camvid"
        args.ignore_index = 11
        args.mean = [0.41189489566336, 0.4251328133025, 0.4326707089857]
        args.std = [0.27413549931506, 0.28506257482912, 0.28284674400252]
        args.n_classes = 11
        args.n_emb_dims = args.n_emb_dims
        args.n_epochs = 50

        args.optimizer_type = "Adam"
        args.optimizer_params = {
            "lr": 5e-4,  # 1e-3,
            "betas": (0.9, 0.999),
            "weight_decay": 2e-4,
            "eps": 1e-7
        }

        args.augmentations = {
            "geometric": {
                "random_scale": args.use_aug,
                "random_hflip": args.use_aug,
                "crop": args.use_aug
            },

            "photometric": {
                "random_color_jitter": args.use_aug,
                "random_grayscale": args.use_aug,
                "random_gaussian_blur": args.use_aug
            }
        }

    # naming
    list_keywords = list()
    list_keywords.append(args.dataset_name)
    list_keywords.append("aug") if args.use_aug else "None"

    if args.use_softmax:
        list_keywords.append("sm")

    elif args.model_name == "gcpl_seg":
        list_keywords.append("gcpl_seg")
        list_keywords.append(f"n_emb_dims_{args.n_emb_dims}")

        if args.use_pl:
            list_keywords.append("pl")
            list_keywords.append(str(args.w_pl))

        if args.use_vl:
            list_keywords.append("vl")
            list_keywords.append(str(args.w_vl))

    elif args.model_name == "mp_seg":
        list_keywords.append("mp_seg")
        list_keywords.append(f"n_emb_dims_{args.n_emb_dims}")

        args.momentum_prototypes = 0.99
        args.momentum_radii = 0.99

        if args.use_pl:
            list_keywords.append("pl")
            list_keywords.append(str(args.w_pl))

        if args.use_repl:
            list_keywords.append("repl")
            list_keywords.append(str(args.w_repl))

    list_keywords.append(f"n_pixels_{args.n_pixels_per_img}")
    list_keywords.append(str(args.seed))
    list_keywords.append("debug") if args.debug else None
    args.experim_name = '_'.join(list_keywords)

    # create dirs
    args.dir_checkpoints = f"{args.dir_root}/checkpoints/{args.experim_name}"
    os.makedirs(f"{args.dir_checkpoints}/train", exist_ok=True)
    os.makedirs(f"{args.dir_checkpoints}/val", exist_ok=True)

    # initialise logs
    args.log_train = f"{args.dir_checkpoints}/log_train.txt"
    args.log_val = f"{args.dir_checkpoints}/log_val.txt"
    write_log(args.log_train, header=["epoch", "mIoU", "pixel_acc", "loss"])
    write_log(args.log_val, header=["epoch", "mIoU", "pixel_acc"])

    main(args)
