import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.camvid import CamVidDataset
from criterions.dop import Criterion
from utils.metrics import prediction, eval_metrics
from utils.utils import AverageMeter, EmbeddingVisualiser, write_log


class Validator:
    def __init__(self, args):
        if args.dataset_name == "cv":
            dataset = CamVidDataset(args, val=True)

        self.dataset_name = args.dataset_name
        self.dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.n_workers)
        self.device = torch.device("cuda:{:s}".format(args.gpu_ids) if torch.cuda.is_available() else "cpu:0")
        self.distance = Criterion.distance
        self.dir_checkpoints = f"{args.dir_root}/checkpoints/{args.experim_name}/val"
        self.experim_name = args.experim_name
        self.ignore_index = args.ignore_index
        self.log_val = args.log_val
        self.n_classes = args.n_classes
        self.non_isotropic = args.non_isotropic
        self.use_softmax = args.use_softmax

    def __call__(self, model, prototypes, epoch):
        model.eval()

        running_miou = AverageMeter()
        running_pixel_acc = AverageMeter()
        dataloader_iter = iter(self.dataloader)
        tbar = tqdm(range(len(self.dataloader)))
        fmt = "mIoU: {:.3f} | pixel acc.: {:.3f}"

        total_inter, total_union = 0, 0
        total_correct, total_label = 0, 0
        with torch.no_grad():
            for _ in tbar:
                dict_data = next(dataloader_iter)
                x, y = dict_data['x'].to(self.device), dict_data['y'].to(self.device)
                dict_outputs = model(x)

                if self.use_softmax:
                    pred = dict_outputs['pred'].argmax(dim=1)
                else:
                    emb = dict_outputs['emb']
                    pred = prediction(emb, prototypes, non_isotropic=self.non_isotropic)

                correct, labeled, inter, union = eval_metrics(pred, y, self.n_classes, self.ignore_index)

                total_inter, total_union = total_inter + inter, total_union + union
                total_correct, total_label = total_correct + correct, total_label + labeled

                # PRINT INFO
                pix_acc = 1.0 * total_correct / (np.spacing(1) + total_label)
                IoU = 1.0 * total_inter / (np.spacing(1) + total_union)
                mIoU = IoU.mean()

                running_miou.update(mIoU)
                running_pixel_acc.update(pix_acc)
                tbar.set_description(fmt.format(running_miou.avg, running_pixel_acc.avg))

        print('\n' + '=' * 100)
        print("Experim name:", self.experim_name)
        print("Epoch {:d} | miou: {:.3f} | pixel_acc.: {:.3f}".format(epoch, running_miou.avg, running_pixel_acc.avg))
        print('=' * 100 + '\n')
        write_log(self.log_val, list_entities=[epoch, running_miou.avg, running_pixel_acc.avg])