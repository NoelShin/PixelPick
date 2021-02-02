import os
from math import ceil
from copy import copy, deepcopy

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from networks.model import FPNSeg
from networks.deeplab import DeepLab
from networks.modules import init_prototypes, init_radii, EMA
from utils.class_histogram import ClassHistogram
from utils.metrics import prediction, eval_metrics
from utils.utils import get_dataloader, get_criterion, get_optimizer, get_lr_scheduler, get_validator, AverageMeter
from utils.utils import write_log, send_file, Visualiser, zip_dir
from al_selectors.query import QuerySelector


class Model:
    def __init__(self, args):
        # common args
        self.args = args
        self.best_miou = -1.0
        self.dataset_name = args.dataset_name
        self.debug = args.debug
        self.dir_checkpoints = f"{args.dir_root}/checkpoints/{args.experim_name}"
        self.experim_name = args.experim_name
        self.ignore_index = args.ignore_index

        self.max_budget = args.max_budget
        self.n_classes = args.n_classes
        self.n_epochs = args.n_epochs
        self.n_epochs_query = args.n_epochs_query
        self.n_pixels_per_img = args.n_pixels_by_us + args.n_pixels_by_oracle_cb  # args.n_pixels_per_img
        self.n_pixels_per_query = (args.n_pixels_by_us + args.n_pixels_by_oracle_cb)
        self.nth_query = -1
        self.stride_total = args.stride_total
        self.use_softmax = args.use_softmax
        self.use_img_inp = args.use_img_inp

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu:0")
        self.dataloader = get_dataloader(args, val=False, query=False,
                                         shuffle=True, batch_size=args.batch_size, n_workers=args.n_workers)
        self.dataloader_query = get_dataloader(args, val=False, query=True,
                                               shuffle=False, batch_size=1, n_workers=args.n_workers)
        self.dataloader_val = get_dataloader(args, val=True, query=False,
                                             shuffle=False, batch_size=1, n_workers=args.n_workers)
        if args.network_name == "FPN":
            self.model = FPNSeg(args).to(self.device)
        else:
            self.model = DeepLab(args).to(self.device)

        self.criterion = get_criterion(args, self.device)

        self.lr_scheduler_type = args.lr_scheduler_type

        self.vis = Visualiser(args.dataset_name)
        self.query_strategy = args.query_strategy

        self.query_selector = QuerySelector(args, self.dataloader_query)

        # for tracking stats
        self.running_loss = AverageMeter()
        self.running_miou = AverageMeter()
        self.running_pixel_acc = AverageMeter()

        self.w_img_inp = args.w_img_inp

        self.model_name = args.model_name
        self.network_name = args.network_name
        self.n_emb_dims = args.n_emb_dims
        self.n_prototypes = args.n_prototypes

        self.use_visual_acuity = args.use_visual_acuity
        self.w_visual_acuity = args.w_visual_acuity

    def __call__(self):
        # fully-supervised model
        if self.n_pixels_per_query == 0:
            self.dir_checkpoints = f"{self.dir_checkpoints}/fully_sup"
            self.log_train = f"{self.dir_checkpoints}/log_train.txt"
            self.log_val = f"{self.dir_checkpoints}/log_val.txt"

            os.makedirs(f"{self.dir_checkpoints}", exist_ok=True)
            write_log(f"{self.log_train}", header=["epoch", "mIoU", "pixel_acc", "loss"])
            write_log(f"{self.log_val}", header=["epoch", "mIoU", "pixel_acc"])

            self._train()

        # active learning model
        else:
            for nth_query in range(self.max_budget // self.n_pixels_per_query + 1):
                self.dir_checkpoints = f"{self.dir_checkpoints}/{nth_query}_query"
                self.log_train = f"{self.dir_checkpoints}/log_train.txt"
                self.log_val = f"{self.dir_checkpoints}/log_val.txt"

                os.makedirs(f"{self.dir_checkpoints}", exist_ok=True)
                write_log(f"{self.log_train}", header=["epoch", "mIoU", "pixel_acc", "loss"])
                write_log(f"{self.log_val}", header=["epoch", "mIoU", "pixel_acc"])

                # os.makedirs(f"{self.dir_checkpoints}/{nth_query}_query", exist_ok=True)
                # write_log(f"{self.dir_checkpoints}/{nth_query}_query/log_train.txt",
                #           header=["epoch", "mIoU", "pixel_acc", "loss"])
                # write_log(f"{self.dir_checkpoints}/{nth_query}_query/log_val.txt",
                #           header=["epoch", "mIoU", "pixel_acc"])

                self.nth_query = nth_query

                self._train()

                # draw histograms
                # ClassHistogram(self.args, nth_query).draw_hist()  # dst=f"{self.dir_checkpoints}/{nth_query}_query")

                # zip_file = zip_dir(f"{self.dir_checkpoints}/{nth_query}_query")
                # send_file(zip_file, file_name=f"{self.experim_name}_{nth_query}_query")
                if nth_query == (self.max_budget // self.n_pixels_per_query) or self.n_pixels_per_img == 0:
                    break

                # select queries using the current model and label them.
                queries = self.query_selector(nth_query)
                self.dataloader.dataset.label_queries(queries, nth_query + 1)
        return

    def _train_epoch(self, epoch, model, optimizer, lr_scheduler, prototypes=None):
        if self.n_pixels_per_img != 0:
            print(f"training an epoch {epoch} of {self.nth_query}th query ({self.dataloader.dataset.arr_masks.sum()} labelled pixels)")
        log = f"{self.log_train}"
        fp = f"{self.dir_checkpoints}/{epoch}_train.png"
        # log = f"{self.dir_checkpoints}/{self.nth_query}_query/log_train.txt"
        # fp = f"{self.dir_checkpoints}/{self.nth_query}_query/{epoch}_train.png"

        model.train()
        dataloader_iter = iter(self.dataloader)
        tbar = tqdm(range(len(self.dataloader)))

        total_inter, total_union = 0, 0
        total_correct, total_label = 0, 0
        for batch_ind in tbar:
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
                prob = F.softmax(logits.detach(), dim=1)

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

            loss = torch.tensor(0., dtype=torch.float32).to(self.device)

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

            if self.lr_scheduler_type == "Poly":
                lr_scheduler.step(epoch=epoch-1)

            if self.debug:
                break

        if self.lr_scheduler_type == "MultiStepLR":
            lr_scheduler.step(epoch=epoch - 1)

        write_log(log, list_entities=[epoch, self.running_miou.avg, self.running_pixel_acc.avg, self.running_loss.avg])
        self._reset_meters()

        confidence = self._query(prob, 'least_confidence')  # prob.max(dim=1)[0]
        margin = self._query(prob, 'margin_sampling')
        entropy = self._query(prob, 'entropy')

        dict_tensors = {'input': x[0].cpu(),  # dict_data['x'][0].cpu(),
                        'target': dict_data['y'][0].cpu(),
                        'pred': pred[0].detach().cpu(),
                        'confidence': confidence[0].cpu(),
                        'margin': -margin[0].cpu(),  # minus sign is to draw smaller margin part brighter
                        'entropy': entropy[0].cpu()}

        self.vis(dict_tensors, fp=fp)
        return model, optimizer, lr_scheduler, prototypes

    def _train(self):
        print(f"\n ({self.experim_name}) training...\n")
        if self.network_name == "FPN":
            model = FPNSeg(self.args).to(self.device)
        else:
            model = DeepLab(self.args).to(self.device)

        prototypes = init_prototypes(self.n_classes, self.n_emb_dims, self.n_prototypes,
                                     mode='mean',
                                     model=self.model,
                                     dataset=self.dataloader.dataset,
                                     ignore_index=self.ignore_index,
                                     learnable=self.model_name == "gcpl_seg",
                                     device=self.device) if not self.use_softmax else None

        optimizer = get_optimizer(self.args, model, prototypes=prototypes)
        lr_scheduler = get_lr_scheduler(self.args, optimizer=optimizer, iters_per_epoch=len(self.dataloader))

        for e in range(1, 1 + self.n_epochs):
            model, optimizer, lr_scheduler, prototypes = self._train_epoch(e, model, optimizer, lr_scheduler, prototypes)

            # if type(lr_scheduler, )
            # lr_scheduler.step(e)
            self._val(e, model, prototypes)
            if self.debug:
                break
        self.best_miou = -1.0
        return

    def _val(self, epoch, model, prototypes=None):
        # log = f"{self.dir_checkpoints}/{self.nth_query}_query/log_val.txt"
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
                    prob = F.softmax(logits.detach(), dim=1)

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

        if self.running_miou.avg > self.best_miou:
            state_dict = dict()
            state_dict_model = model.state_dict()
            state_dict.update({"model": state_dict_model})
            if not self.use_softmax:
                # state_dict_prototypes = prototypes.state_dict()
                state_dict.update({"prototypes": prototypes.cpu()})

            torch.save(state_dict, f"{self.dir_checkpoints}/best_miou_model.pt")
            # torch.save(state_dict, f"{self.dir_checkpoints}/{self.nth_query}_query/best_miou_model.pt")
            print("best model has been saved (epoch: {:d} | prev. miou: {:.4f} => new miou: {:.4f})."
                  .format(epoch, self.best_miou, self.running_miou.avg))
            self.best_miou = self.running_miou.avg

        write_log(self.log_val, list_entities=[epoch, self.running_miou.avg, self.running_pixel_acc.avg])
        print('\n' + '=' * 100)
        print("Experim name:", self.experim_name)
        print("Epoch {:d} | miou: {:.3f} | pixel_acc.: {:.3f}".format(epoch,
                                                                      self.running_miou.avg,
                                                                      self.running_pixel_acc.avg))
        print('=' * 100 + '\n')

        self._reset_meters()

        confidence = self._query(prob, 'least_confidence')  # prob.max(dim=1)[0]
        margin = self._query(prob, 'margin_sampling')
        entropy = self._query(prob, 'entropy')

        dict_tensors = {'input': dict_data['x'][0].cpu(),
                        'target': dict_data['y'][0].cpu(),
                        'pred': pred[0].detach().cpu(),
                        'confidence': confidence[0].cpu(),
                        'margin': -margin[0].cpu(),  # minus sign is to draw smaller margin part brighter
                        'entropy': entropy[0].cpu()}

        self.vis(dict_tensors, fp=f"{self.dir_checkpoints}/{epoch}_val.png")
        # self.vis(dict_tensors, fp=f"{self.dir_checkpoints}/{self.nth_query}_query/{epoch}_val.png")
        return

    @staticmethod
    def _query(prob, query_strategy):
        # prob: b x n_classes x h x w
        if query_strategy == "least_confidence":
            query = 1.0 - prob.max(dim=1)[0]  # b x h x w

        elif query_strategy == "margin_sampling":
            query = prob.topk(k=2, dim=1).values  # b x k x h x w
            query = (query[:, 0, :, :] - query[:, 1, :, :]).abs()  # b x h x w

        elif query_strategy == "entropy":
            query = (-prob * torch.log(prob)).sum(dim=1)  # b x h x w

        elif query_strategy == "random":
            b, _, h, w = prob.shape
            query = torch.rand((b, h, w))

        return query

    def _reset_meters(self):
        self.running_miou.reset()
        self.running_pixel_acc.reset()
        self.running_loss.reset()
