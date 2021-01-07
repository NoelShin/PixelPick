import os
from math import ceil
from copy import copy, deepcopy

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from networks.model import FPNSeg
from networks.modules import init_prototypes, init_radii, EMA
from utils.metrics import prediction, eval_metrics
from utils.utils import get_dataloader, get_criterion, get_optimizer, get_lr_scheduler, get_validator, AverageMeter
from utils.utils import write_log, send_file, Visualiser, zip_dir


class Model:
    def __init__(self, args):
        # common args
        self.args = args
        self.dataset_name = args.dataset_name
        self.debug = args.debug
        self.dir_checkpoints = f"{args.dir_root}/checkpoints/{args.experim_name}"
        self.experim_name = args.experim_name
        self.ignore_index = args.ignore_index

        self.max_budget = args.max_budget
        self.n_classes = args.n_classes
        self.n_epochs = args.n_epochs
        self.n_epochs_query = args.n_epochs_query
        self.n_pixels_per_img = args.n_pixels_per_img
        self.n_pixels_per_query = args.n_pixels_per_query
        self.nth_query = -1
        self.stride_total = args.stride_total
        self.use_softmax = args.use_softmax

        self.device = torch.device("cuda:{:s}".format(args.gpu_ids) if torch.cuda.is_available() else "cpu:0")
        self.dataloader = get_dataloader(args, val=False, query=False,
                                         shuffle=True, batch_size=args.batch_size, n_workers=args.n_workers)
        self.dataloader_query = get_dataloader(args, val=False, query=True,
                                               shuffle=False, batch_size=1, n_workers=args.n_workers)
        self.dataloader_val = get_dataloader(args, val=True, query=False,
                                             shuffle=False, batch_size=1, n_workers=args.n_workers)
        self.model = FPNSeg(args).to(self.device)
        self.prototypes = init_prototypes(args.n_classes, args.n_emb_dims, args.n_prototypes,
                                          mode='mean',
                                          model=self.model,
                                          dataset=self.dataloader.dataset,
                                          ignore_index=args.ignore_index,
                                          learnable=args.model_name == "gcpl_seg",
                                          device=self.device) if not self.use_softmax else None
        self.criterion = get_criterion(args, self.device)
        self.optimizer = get_optimizer(args, self.model, prototypes=self.prototypes)
        self.lr_scheduler = get_lr_scheduler(args, optimizer=self.optimizer, iters_per_epoch=len(self.dataloader))
        self.vis = Visualiser(args.dataset_name)
        self.query_strategy = args.query_strategy

        # for tracking stats
        self.running_loss = AverageMeter()
        self.running_miou = AverageMeter()
        self.running_pixel_acc = AverageMeter()

    def __call__(self):
        for nth_query in range(self.max_budget // self.n_pixels_per_query + 1):
            os.makedirs(f"{self.dir_checkpoints}/{nth_query}_query", exist_ok=True)
            write_log(f"{self.dir_checkpoints}/{nth_query}_query/log_train.txt",
                      header=["epoch", "mIoU", "pixel_acc", "loss"])
            write_log(f"{self.dir_checkpoints}/{nth_query}_query/log_train_query.txt",
                      header=["epoch", "mIoU", "pixel_acc", "loss"])
            write_log(f"{self.dir_checkpoints}/{nth_query}_query/log_val.txt",
                      header=["epoch", "mIoU", "pixel_acc"])

            self.nth_query = nth_query

            # Train the current model with the annotated dataset for n_epochs and validate it.
            # This does not change parameters of the model defined as an attribute of this instance.
            self._val_al_stage()
            if nth_query == (self.max_budget // self.n_pixels_per_query):
                break

            # Train the model with the current dataset for n_epochs_query (not to be confused with n_epochs).
            # This updates parameters of the model defined as an attribute of this instance.
            self._train()

            # select queries using the current model and label them.
            queries = self._select_queries()
            self.dataloader.dataset.label_queries(queries)
        return

    def _train_epoch(self, epoch, model=None, optimizer=None, prototypes=None):
        print(f"training an epoch {epoch} of {self.nth_query}th query")
        if any(i is not None for i in [model, optimizer, prototypes]):
            assert all(i is not None for i in [model, optimizer, prototypes]) is not None

            log = f"{self.dir_checkpoints}/{self.nth_query}_query/log_train.txt"
            fp = f"{self.dir_checkpoints}/{self.nth_query}_query/{epoch}_train.png"

        else:
            model = copy(self.model)
            optimizer = copy(self.optimizer)
            prototypes = copy(self.prototypes)

            log = f"{self.dir_checkpoints}/{self.nth_query}_query/log_train_query.txt"
            fp = f"{self.dir_checkpoints}/{self.nth_query}_query/{epoch}_train_query.png"

        model.train()
        dataloader_iter = iter(self.dataloader)
        tbar = tqdm(range(len(self.dataloader)))

        for batch_ind in tbar:
            total_inter, total_union = 0, 0
            total_correct, total_label = 0, 0

            dict_data = next(dataloader_iter)
            x, y = dict_data['x'].to(self.device), dict_data['y'].to(self.device)
            if self.n_pixels_per_img != 0:
                mask = dict_data['mask'].to(self.device, torch.bool)
                y.flatten()[~mask.flatten()] = self.ignore_index

            dict_outputs = model(x)

            if self.use_softmax:
                logits = dict_outputs["pred"]
                dict_losses = {"ce": F.cross_entropy(logits, y, ignore_index=self.ignore_index)}
                pred = logits.argmax(dim=1)  # for computing mIoU, pixel acc.
                prob = F.softmax(logits.detach())

            else:
                emb = dict_outputs['emb']  # b x n_emb_dims x h x w

                b, c, h, w = emb.shape
                emb_flatten = emb.transpose(1, 0)  # c x b x h x w
                emb_flatten = emb_flatten.contiguous().view(c, b * h * w)  # c x (b * h * w)
                emb_flatten = emb_flatten.transpose(1, 0)  # (b * h * w) x c
                y_flatten = y.flatten()  # (b * h * w)

                unique_labels_batch = set(sorted(y_flatten.cpu().numpy().tolist())) - {self.ignore_index}

                dict_label_emb = dict()
                for label in unique_labels_batch:
                    ind_label = (y_flatten == label)
                    emb_label = emb_flatten[ind_label]
                    dict_label_emb.update({label: emb_label})

                dict_outputs.update({"dict_label_emb": dict_label_emb})
                pred, dist = prediction(emb.detach(), prototypes.detach(), return_distance=True)
                prob = F.softmax(-dist.detach(), dim=1)

                dict_losses = self.criterion(dict_outputs, prototypes, labels=y)

            loss = torch.tensor(0, dtype=torch.float32).to(self.device)

            fmt = "({:s}) Epoch {:d} | mIoU.: {:.3f} | pixel acc.: {:.3f} | Loss: {:.3f}"
            for loss_k, loss_v in dict_losses.items():
                fmt += " | {:s}: {:.3f}".format(loss_k, loss_v.detach().cpu().item())
                loss += loss_v

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            self.running_loss.update(loss.detach().item())
            correct, labeled, inter, union = eval_metrics(pred, y, self.n_classes, self.ignore_index)

            total_inter, total_union = total_inter + inter, total_union + union
            total_correct, total_label = total_correct + correct, total_label + labeled

            pix_acc = 1.0 * total_correct / (np.spacing(1) + total_label)
            IoU = 1.0 * total_inter / (np.spacing(1) + total_union)
            mIoU = IoU.mean()

            self.running_miou.update(mIoU)
            self.running_pixel_acc.update(pix_acc)

            tbar.set_description(fmt.format(self.experim_name,
                                            epoch,
                                            self.running_miou.avg,
                                            self.running_pixel_acc.avg,
                                            self.running_loss.avg))

            # if self.debug:
            #     break

        write_log(log, list_entities=[epoch, self.running_miou.avg, self.running_pixel_acc.avg, self.running_loss.avg])
        self._reset_meters()

        confidence = self._query(prob, 'least_confidence')  # prob.max(dim=1)[0]
        margin = self._query(prob, 'margin_sampling')
        entropy = self._query(prob, 'entropy')

        dict_tensors = {'input': dict_data['x'][0].cpu(),
                        'target': dict_data['y'][0].cpu(),
                        'pred': pred[0].detach().cpu(),
                        'confidence': -confidence[0].cpu(),  # minus sign is to draw more uncertain part brighter
                        'margin': -margin[0].cpu(),  # minus sign is to draw smaller margin part brighter
                        'entropy': entropy[0].cpu()}

        self.vis(dict_tensors, fp=fp)
        return model, optimizer, prototypes

    def _val_al_stage(self):
        print(f"\n ({self.experim_name}) validating active learning stage {self.nth_query}...\n")
        model = deepcopy(self.model)
        prototypes = deepcopy(self.prototypes)
        optimizer = get_optimizer(self.args, model, prototypes=prototypes)
        lr_scheduler = get_lr_scheduler(self.args, optimizer=optimizer, iters_per_epoch=len(self.dataloader))

        for e in range(1, 1 + self.n_epochs):
            model, optimizer, prototypes = self._train_epoch(e, model, optimizer, prototypes)
            lr_scheduler.step(e)
            self._val(e, model, prototypes)
            if self.debug:
                break
        return

    def _train(self):
        print(f"\n ({self.experim_name}) training...\n")
        for e in range(1, 1 + self.n_epochs_query):
            self._train_epoch(e)
            if self.debug:
                break
        return

    def _val(self, epoch, model, prototypes=None):
        log = f"{self.dir_checkpoints}/{self.nth_query}_query/log_val.txt"

        model.eval()

        dataloader_iter = iter(self.dataloader_val)
        tbar = tqdm(range(len(self.dataloader_val)))
        fmt = "mIoU: {:.3f} | pixel acc.: {:.3f}"

        total_inter, total_union = 0, 0
        total_correct, total_label = 0, 0
        with torch.no_grad():
            for _ in tbar:
                dict_data = next(dataloader_iter)
                x, y = dict_data['x'].to(self.device), dict_data['y'].to(self.device)

                if self.dataset_name == "voc":
                    h, w = y.shape[1:]
                    pad_h = ceil(h / self.stride_total) * self.stride_total - x.shape[2]
                    pad_w = ceil(w / self.stride_total) * self.stride_total - x.shape[3]
                    x = F.pad(x, pad=(0, pad_w, 0, pad_h), mode='reflect')
                    dict_outputs = model(x)
                    k = 'pred' if self.use_softmax else 'emb'
                    dict_outputs[k] = dict_outputs[k][:, :, :h, :w]

                else:
                    dict_outputs = model(x)

                if self.use_softmax:
                    logits = dict_outputs['pred']
                    pred = logits.argmax(dim=1)
                    prob = F.softmax(logits.detach())

                else:
                    emb = dict_outputs['emb']
                    pred, dist = prediction(emb, prototypes, return_distance=True)
                    prob = F.softmax(-dist, dim=1)

                    if self.dataset_name == "voc":
                        pred.flatten()[prob.flatten() < 0.5] = -1
                        pred += 1

                # b x h x w
                correct, labeled, inter, union = eval_metrics(pred, y, self.n_classes, self.ignore_index)

                total_inter, total_union = total_inter + inter, total_union + union
                total_correct, total_label = total_correct + correct, total_label + labeled

                # compute metrics
                pix_acc = 1.0 * total_correct / (np.spacing(1) + total_label)
                IoU = 1.0 * total_inter / (np.spacing(1) + total_union)
                mIoU = IoU.mean()

                self.running_miou.update(mIoU)
                self.running_pixel_acc.update(pix_acc)
                tbar.set_description(fmt.format(self.running_miou.avg, self.running_pixel_acc.avg))

                if self.debug:
                    break

        print('\n' + '=' * 100)
        print("Experim name:", self.experim_name)
        print("Epoch {:d} | miou: {:.3f} | pixel_acc.: {:.3f}".format(epoch,
                                                                      self.running_miou.avg,
                                                                      self.running_pixel_acc.avg))
        print('=' * 100 + '\n')

        write_log(log, list_entities=[epoch, self.running_miou.avg, self.running_pixel_acc.avg])
        self._reset_meters()

        confidence = self._query(prob, 'least_confidence')  # prob.max(dim=1)[0]
        margin = self._query(prob, 'margin_sampling')
        entropy = self._query(prob, 'entropy')

        dict_tensors = {'input': dict_data['x'][0].cpu(),
                        'target': dict_data['y'][0].cpu(),
                        'pred': pred[0].detach().cpu(),
                        'confidence': -confidence[0].cpu(),  # minus sign is to draw more uncertain part brighter
                        'margin': -margin[0].cpu(),  # minus sign is to draw smaller margin part brighter
                        'entropy': entropy[0].cpu()}

        self.vis(dict_tensors, fp=f"{self.dir_checkpoints}/{self.nth_query}_query/{epoch}_val.png")
        return

    def _select_queries(self):
        print(f"\n({self.experim_name}) selecting queries...")
        self.model.eval()

        dataloader_iter = iter(self.dataloader_query)
        tbar = tqdm(range(len(self.dataloader_query)))

        list_quries = list()
        with torch.no_grad():
            for batch_ind in tbar:
                dict_data = next(dataloader_iter)
                x = dict_data['x'].to(self.device)

                dict_outputs = self.model(x)

                if self.use_softmax:
                    prob = dict_outputs["pred"]

                else:
                    emb = dict_outputs['emb']  # b x n_emb_dims x h x w
                    _, dist = prediction(emb.detach(), self.prototypes.detach(), return_distance=True)
                    prob = F.softmax(-dist, dim=1)

                list_quries.append(self._query(prob, self.query_strategy).cpu())

        queries = torch.cat(list_quries, dim=0)  # b x h x w
        b, h, w = queries.shape
        queries = queries.view(b, h * w)  # b x (h * w)
        grid = torch.zeros_like(queries)  # b x (h * w)

        ind_best_queries = torch.topk(queries,
                                      k=self.n_pixels_per_query,
                                      dim=1,
                                      largest=self.query_strategy == "entropy").indices  # b x k

        for i in range(b):
            for j in range(self.n_pixels_per_query):
                grid[i, ind_best_queries[i, j]] = 1
        grid = grid.view(b, h, w)
        return grid.numpy().astype(np.bool)

    @staticmethod
    def _query(prob, query_strategy):
        # prob: b x n_classes x h x w
        if query_strategy == "least_confidence":
            query = prob.max(dim=1)[0]  # b x h x w

        elif query_strategy == "margin_sampling":
            query = prob.topk(k=2, dim=1).values  # b x k x h x w
            query = (query[:, 0, :, :] - query[:, 1, :, :]).abs()  # b x h x w

        elif query_strategy == "entropy":
            query = (-prob * torch.log(prob)).sum(dim=1)  # b x h x w
        return query

    def _reset_meters(self):
        self.running_miou.reset()
        self.running_pixel_acc.reset()
        self.running_loss.reset()
