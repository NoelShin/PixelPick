import os
from shutil import rmtree
from math import ceil
from copy import copy, deepcopy

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.metrics import prediction, eval_metrics, RunningScore
from utils.utils import get_dataloader, get_model, get_criterion, get_optimizer, get_lr_scheduler, get_validator
from utils.utils import AverageMeter, write_log, send_file, Visualiser, zip_dir
from al_selectors.query import QuerySelector
from criterions.contrastive_loss import get_dict_label_projection, compute_contrastive_loss


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
        self.dataloader = get_dataloader(deepcopy(args), val=False, query=False,
                                         shuffle=True, batch_size=args.batch_size, n_workers=args.n_workers)

        self.dataloader_query = get_dataloader(deepcopy(args), val=False, query=True,
                                               shuffle=False, batch_size=1, n_workers=args.n_workers)

        self.dataloader_val = get_dataloader(deepcopy(args), val=True, query=False,
                                             shuffle=False, batch_size=1, n_workers=args.n_workers)

        self.model = get_model(args).to(self.device)

        self.criterion = get_criterion(args, self.device)
        self.lr_scheduler_type = args.lr_scheduler_type

        self.vis = Visualiser(args.dataset_name)
        self.query_strategy = args.query_strategy

        self.query_selector = QuerySelector(args, self.dataloader_query)

        # for tracking stats
        self.running_loss = AverageMeter()
        self.running_score = RunningScore(args.n_classes)

        self.w_img_inp = args.w_img_inp

        self.model_name = args.model_name
        self.network_name = args.network_name
        self.n_emb_dims = args.n_emb_dims
        self.n_prototypes = args.n_prototypes

        self.use_scribbles = args.use_scribbles
        self.use_visual_acuity = args.use_visual_acuity
        self.w_visual_acuity = args.w_visual_acuity

        self.use_openset = args.use_openset

        self.use_pseudo_label = args.use_pseudo_label
        self.window_size = args.window_size

        self.use_contrastive_loss = args.use_contrastive_loss
        self.use_region_contrast = args.use_region_contrast
        self.w_contrastive = args.w_contrastive
        self.selection_mode = args.selection_mode
        self.temperature = args.temperature
        self.dict_label_projection = None

        # if active learning
        if self.n_pixels_per_query > 0:
            self.model_0_query = f"{args.network_name}_0_query_{args.seed}.pt"

    def __call__(self):
        # fully-supervised model
        if self.n_pixels_per_query == 0:
            dir_checkpoints = f"{self.dir_checkpoints}/fully_sup"
            self.log_train = f"{dir_checkpoints}/log_train.txt"
            self.log_val = f"{dir_checkpoints}/log_val.txt"

            os.makedirs(f"{dir_checkpoints}", exist_ok=True)
            write_log(f"{self.log_train}", header=["epoch", "mIoU", "pixel_acc", "loss"])
            write_log(f"{self.log_val}", header=["epoch", "mIoU", "pixel_acc"])

            self._train()

            # zip_file = zip_dir(f"{dir_checkpoints}", remove_dir=True)
            # send_file(zip_file, file_name=f"{self.experim_name}", remove_file=True)
        # active learning model
        else:
            if os.path.isfile(self.model_0_query) and False:
                state_dict = torch.load(self.model_0_query)
                model = deepcopy(self.model)
                model.load_state_dict(state_dict["model"])

                for nth_query in range(1, self.max_budget // self.n_pixels_per_query):
                    dir_checkpoints = f"{self.dir_checkpoints}/{nth_query}_query"
                    self.log_train = f"{dir_checkpoints}/log_train.txt"
                    self.log_val = f"{dir_checkpoints}/log_val.txt"

                    os.makedirs(f"{dir_checkpoints}", exist_ok=True)
                    write_log(f"{self.log_train}", header=["epoch", "mIoU", "pixel_acc", "loss"])
                    write_log(f"{self.log_val}", header=["epoch", "mIoU", "pixel_acc"])

                    # select queries using the current model and label them.
                    queries, queries_err = self.query_selector(nth_query, model)
                    self.dataloader.dataset.label_queries(queries, nth_query)
                    if self.dataset_name == "cv" and self.args.simulate_error:
                        self.dataloader.dataset.update_error_queries(queries_err, nth_query)

                    # pseudo-labelling based on the current labels
                    if self.use_pseudo_label:
                        self.dataloader.dataset.update_pseudo_label(model, window_size=self.window_size,
                                                                    nth_query=nth_query)

                    if self.use_contrastive_loss:
                        self.dict_label_projection = get_dict_label_projection(self.dataloader_query, model,
                                                                               arr_masks=self.dataloader.dataset.arr_masks,
                                                                               ignore_index=self.ignore_index,
                                                                               region_contrast=self.use_region_contrast)

                    self.nth_query = nth_query

                    model, prototypes = self._train()

                    # zip_file = zip_dir(f"{dir_checkpoints}", remove_dir=True)
                    # send_file(zip_file, file_name=f"{self.experim_name}_{nth_query}_query", remove_file=True)
                    if nth_query == (self.max_budget // self.n_pixels_per_query) - 1 or self.nth_query == 20:
                        break

            else:
                for nth_query in range(self.max_budget // self.n_pixels_per_query):
                    dir_checkpoints = f"{self.dir_checkpoints}/{nth_query}_query"
                    self.log_train = f"{dir_checkpoints}/log_train.txt"
                    self.log_val = f"{dir_checkpoints}/log_val.txt"

                    os.makedirs(f"{dir_checkpoints}", exist_ok=True)
                    write_log(f"{self.log_train}", header=["epoch", "mIoU", "pixel_acc", "loss"])
                    write_log(f"{self.log_val}", header=["epoch", "mIoU", "pixel_acc"])

                    self.nth_query = nth_query

                    model, prototypes = self._train()

                    if self.use_scribbles:
                        break
                    
                    # select queries using the current model and label them.
                    queries, queries_err = self.query_selector(nth_query, model)
                    self.dataloader.dataset.label_queries(queries, nth_query + 1)

                    if self.dataset_name == "cv" and self.args.simulate_error:
                        self.dataloader.dataset.update_error_queries(queries_err, nth_query)

                    # zip_file = zip_dir(f"{dir_checkpoints}", remove_dir=True)
                    # send_file(zip_file, file_name=f"{self.experim_name}_{nth_query}_query", remove_file=True)
                    if nth_query == (self.max_budget // self.n_pixels_per_query) - 1 or self.n_pixels_per_img == 0:
                        break

                    elif nth_query == 0:
                        torch.save({"model": model.state_dict()}, self.model_0_query)

                    # pseudo-labelling based on the current labels
                    if self.use_pseudo_label:
                        self.dataloader.dataset.update_pseudo_label(model, window_size=self.window_size, nth_query=nth_query+1)

                    if self.use_contrastive_loss:
                        self.dict_label_projection = get_dict_label_projection(self.dataloader_query, model,
                                                                               arr_masks=self.dataloader.dataset.arr_masks,
                                                                               ignore_index=self.ignore_index,
                                                                               region_contrast=self.use_region_contrast)

        # rmtree(f"{self.dir_checkpoints}")
        return

    def _train_epoch(self, epoch, model, optimizer, lr_scheduler, prototypes=None, dict_label_projection=None):
        if self.n_pixels_per_img != 0:
            print(f"training an epoch {epoch} of {self.nth_query}th query ({self.dataloader.dataset.n_pixels_total} labelled pixels)")
            fp = f"{self.dir_checkpoints}/{self.nth_query}_query/{epoch}_train.png"
        else:
            fp = f"{self.dir_checkpoints}/fully_sup/{epoch}_train.png"
        log = f"{self.log_train}"

        model.train()
        dataloader_iter = iter(self.dataloader)
        tbar = tqdm(range(len(self.dataloader)))

        for batch_ind in tbar:
            dict_data = next(dataloader_iter)
            x, y = dict_data['x'].to(self.device), dict_data['y'].to(self.device)

            # if active learning
            if self.n_pixels_per_img != 0:
                mask = dict_data['mask'].to(self.device, torch.bool)
                y.flatten()[~mask.flatten()] = self.ignore_index

                if self.args.simulate_error:
                    mask_err = dict_data['mask_err'].to(self.device, torch.bool)
                    gt_points = y.flatten()[mask_err.flatten()]
                    # replace labels on error points with a random incorrect label.
                    random_error = [np.random.choice(list(set(range(self.n_classes)) - {gt_p.cpu().numpy().item()}), 1)[0] for gt_p in gt_points]
                    y.flatten()[mask_err.flatten()] = torch.tensor(np.array(random_error).astype(np.int64)).to(self.device)
                    assert all(gt_points.cpu().numpy() != y.flatten()[mask_err.flatten()].cpu().numpy())

                if self.use_pseudo_label and self.nth_query > 0:
                    y_pseudo = dict_data["y_pseudo"].to(self.device)

            # forward pass
            dict_outputs = model(x)

            logits = dict_outputs["pred"]
            dict_losses = {"ce": F.cross_entropy(logits, y, ignore_index=self.ignore_index)}
            pred = logits.argmax(dim=1)  # for computing mIoU, pixel acc.
            prob = F.softmax(logits.detach(), dim=1)

            self.running_score.update(y.cpu().numpy(), pred.cpu().numpy())

            if self.use_pseudo_label and self.nth_query > 0:
                dict_losses.update({"ce_pseudo": F.cross_entropy(logits, y_pseudo, ignore_index=self.ignore_index)})

            if self.use_openset:
                emb = dict_outputs['emb_']  # b x n_emb_dims x h x w

                emb = emb / torch.linalg.norm(emb, ord=2, dim=1, keepdim=True)

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
                _, dist = prediction(emb.detach(), prototypes.detach(), return_distance=True)

                prob_ = F.softmax(-dist.detach(), dim=1)
                # pred = (prob + prob_).argmax(dim=1)  # ensemble

                dict_losses.update(self.criterion(dict_outputs, prototypes, labels=y))

            if self.use_contrastive_loss and self.n_pixels_per_img != 0 and self.nth_query > 0:
                projection = dict_outputs["projection"]
                loss_cont = compute_contrastive_loss(projection, y, mask,
                                                     dict_label_projection=self.dict_label_projection,
                                                     selection_mode=self.selection_mode,
                                                     temperature=self.temperature)

                dict_losses.update({"loss_cont": self.w_contrastive * loss_cont})

            loss = torch.tensor(0., dtype=torch.float32).to(self.device)

            fmt = "({:s}) Epoch {:d} | mIoU.: {:.3f} | pixel acc.: {:.3f} | Loss: {:.3f}"
            for loss_k, loss_v in dict_losses.items():
                fmt += " | {:s}: {:.3f}".format(loss_k, loss_v.detach().cpu().item())
                loss += loss_v

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            self.running_loss.update(loss.detach().item())
            scores = self.running_score.get_scores()[0]

            tbar.set_description(fmt.format(self.experim_name,
                                            epoch,
                                            scores["Mean IoU"],
                                            scores["Pixel Acc"],
                                            self.running_loss.avg))

            if self.lr_scheduler_type == "Poly":
                lr_scheduler.step(epoch=epoch-1)

            if self.debug:
                break

        if self.lr_scheduler_type == "MultiStepLR":
            lr_scheduler.step(epoch=epoch - 1)

        write_log(log, list_entities=[epoch, scores["Mean IoU"], scores["Pixel Acc"], self.running_loss.avg])
        self._reset_meters()

        confidence = self._query(prob, 'least_confidence')
        margin = self._query(prob, 'margin_sampling')
        entropy = self._query(prob, 'entropy')

        dict_tensors = {'input': x[0].cpu(),
                        'target': dict_data['y'][0].cpu(),
                        'pred': pred[0].detach().cpu(),
                        'confidence': confidence[0].cpu(),
                        'margin': -margin[0].cpu(),  # minus sign is to draw smaller margin part brighter
                        'entropy': entropy[0].cpu()}

        self.vis(dict_tensors, fp=fp)
        return model, optimizer, lr_scheduler, prototypes

    def _train(self):
        print(f"\n ({self.experim_name}) training...\n")
        model = get_model(self.args).to(self.device)

        prototypes = None

        optimizer = get_optimizer(self.args, model, prototypes=prototypes)
        lr_scheduler = get_lr_scheduler(self.args, optimizer=optimizer, iters_per_epoch=len(self.dataloader))

        for e in range(1, 1 + self.n_epochs):
            model, optimizer, lr_scheduler, prototypes = self._train_epoch(e, model, optimizer, lr_scheduler,
                                                                           prototypes=prototypes,
                                                                           dict_label_projection=self.dict_label_projection)

            self._val(e, model, prototypes)
            if self.debug:
                break

        self.best_miou = -1.0
        return model, prototypes

    def _val(self, epoch, model, prototypes=None):
        model.eval()

        dataloader_iter = iter(self.dataloader_val)
        tbar = tqdm(range(len(self.dataloader_val)))
        fmt = "mIoU: {:.3f} | pixel acc.: {:.3f}"

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

                logits = dict_outputs['pred']
                pred = logits.argmax(dim=1)
                prob = F.softmax(logits.detach(), dim=1)

                if self.use_openset:
                    emb = dict_outputs['emb_']
                    _, dist = prediction(emb, prototypes, return_distance=True)
                    prob_ = F.softmax(-dist, dim=1)

                self.running_score.update(y.cpu().numpy(), pred.cpu().numpy())
                scores = self.running_score.get_scores()[0]

                tbar.set_description(fmt.format(scores["Mean IoU"], scores["Pixel Acc"]))

                if self.debug:
                    break

        if scores["Mean IoU"] > self.best_miou:
            state_dict = dict()
            state_dict_model = model.state_dict()
            state_dict.update({"model": state_dict_model})
            if self.use_openset:
                state_dict.update({"prototypes": prototypes.cpu()})

            if self.n_pixels_per_img != 0:
                torch.save(state_dict, f"{self.dir_checkpoints}/{self.nth_query}_query/best_miou_model.pt")
            else:
                torch.save(state_dict, f"{self.dir_checkpoints}/fully_sup/best_miou_model.pt")
            print("best model has been saved (epoch: {:d} | prev. miou: {:.4f} => new miou: {:.4f})."
                  .format(epoch, self.best_miou, scores["Mean IoU"]))
            self.best_miou = scores["Mean IoU"]

        write_log(self.log_val, list_entities=[epoch, scores["Mean IoU"], scores["Pixel Acc"]])
        print('\n' + '=' * 100)
        print("Experim name:", self.experim_name)
        print("Epoch {:d} | miou: {:.3f} | pixel_acc.: {:.3f}".format(epoch,
                                                                      scores["Mean IoU"],
                                                                      scores["Pixel Acc"]))
        print('=' * 100 + '\n')

        self._reset_meters()

        confidence = self._query(prob, 'least_confidence')
        margin = self._query(prob, 'margin_sampling')
        entropy = self._query(prob, 'entropy')

        dict_tensors = {'input': dict_data['x'][0].cpu(),
                        'target': dict_data['y'][0].cpu(),
                        'pred': pred[0].detach().cpu(),
                        'confidence': confidence[0].cpu(),
                        'margin': -margin[0].cpu(),  # minus sign is to draw smaller margin part brighter
                        'entropy': entropy[0].cpu()}

        if self.n_pixels_per_img != 0:
            self.vis(dict_tensors, fp=f"{self.dir_checkpoints}/{self.nth_query}_query/{epoch}_val.png")
        else:
            self.vis(dict_tensors, fp=f"{self.dir_checkpoints}/fully_sup/{epoch}_val.png")
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

        else:
            raise ValueError
        return query

    def _reset_meters(self):
        self.running_loss.reset()
        self.running_score.reset()
