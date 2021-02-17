import sys
sys.path.append('..')
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm

from camvid import CamVidDataset
from cityscapes import CityscapesDataset
from networks.deeplab import DeepLab
from networks.model import FPNSeg
from utils.utils import get_optimizer, get_lr_scheduler
from utils.metrics import eval_metrics, AverageMeter, RunningScore


def main(args):
    torch.manual_seed(args.seed)

    device = torch.device("cuda:0")

    fp = f"{args.dataset_name}_dr{args.diversity_ratio}_{args.seed}.txt"
    with open(fp, 'w') as f:
        f.close()
    print(fp)

    # dataset
    if args.dataset_name == "cv":
        dataset_train = CamVidDataset(args, val=False)
        dataset_val = CamVidDataset(args, val=True)
    else:
        dataset_train = CityscapesDataset(args, val=False)
        dataset_val = CityscapesDataset(args, val=True)

    dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.n_workers,
                                  drop_last=len(dataset_train) % args.batch_size == 1)
    dataloader_val = DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=args.n_workers)

    # model
    if args.network_name == "deeplab":
        model = DeepLab(args).to(device)
    else:
        model = FPNSeg(args).to(device)

    optimizer = get_optimizer(args, model)
    lr_scheduler = get_lr_scheduler(args, optimizer=optimizer, iters_per_epoch=len(dataloader_train))

    running_score = RunningScore(args.n_classes)
    # training
    for e in range(1, 51):
        tbar, iter_dataloader = tqdm(range(len(dataloader_train))), iter(dataloader_train)
        model.train()
        for batch_ind in tbar:
            dict_data = next(iter_dataloader)
            x, y, mask = dict_data['x'].to(device), dict_data['y'].to(device), dict_data['mask'].to(device, torch.bool)
            y_flat = y.flatten()
            y_flat[~mask.flatten()] = args.ignore_index
            y = y_flat.view(y.shape)

            dict_outputs = model(x)

            logits = dict_outputs['pred']
            pred = logits.argmax(dim=1)

            loss = F.cross_entropy(logits, y, ignore_index=args.ignore_index)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr_scheduler.step(epoch=e-1)

            running_score.update(y.cpu().numpy(), pred.cpu().numpy())

            tbar.set_description("Epoch {:d} (train) | {:.3f}".format(e, running_score.get_scores()[0]['Mean IoU']))
        running_score.reset()

        model.eval()
        with torch.no_grad():
            for batch_ind, dict_data in tqdm(enumerate(dataloader_val)):
                x, y = dict_data['x'].to(device), dict_data['y'].to(device)

                dict_outputs = model(x)

                pred = dict_outputs['pred'].argmax(dim=1)

                running_score.update(y.squeeze().cpu().numpy(), pred.squeeze().cpu().numpy())

        print("Epoch {:d} (val) | {:.3f}".format(e, running_score.get_scores()[0]['Mean IoU']))
        with open(fp, 'a') as f:
            f.write(f"{running_score.get_scores()[0]['Mean IoU']}\n")
            f.close()
        running_score.reset()
    return


if __name__ == '__main__':
    import sys
    sys.path.append('..')
    from args import Arguments

    args = Arguments().parse_args()
    main(args)