import os

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from networks.model import FPNSeg
from networks.deeplab import DeepLab
from utils.metrics import prediction, eval_metrics


class QuerySelector:
    def __init__(self, args, dataloader, device=torch.device("cuda:0")):
        self.args = args
        self.dataloader = dataloader
        self.debug = args.debug
        self.device = device
        self.dir_checkpoints = f"{args.dir_root}/checkpoints/{args.experim_name}"
        self.ignore_index = args.ignore_index
        self.mc_n_steps = args.mc_n_steps
        self.model_name = args.model_name
        self.n_classes = args.n_classes
        self.n_emb_dims = args.n_emb_dims
        self.n_prototypes = args.n_prototypes

        self.n_pixels_by_oracle_cb = args.n_pixels_by_oracle_cb
        self.n_pixels_by_us = args.n_pixels_by_us

        self.network_name = args.network_name
        self.uncertainty_sampler = UncertaintySampler(args.query_strategy)
        self.use_softmax = args.use_softmax
        self.query_strategy = args.query_strategy
        self.use_cb_sampling = args.use_cb_sampling
        self.use_mc_dropout = args.use_mc_dropout
        self.vote_type = args.vote_type

    def __call__(self, nth_query, model=None):
        # state_dict = torch.load(f"{self.dir_checkpoints}/{nth_query}_query/best_miou_model.pt")
        # if self.network_name == "FPN":
        #     model = FPNSeg(self.args).to(self.device)
        # else:
        #     model = DeepLab(self.args).to(self.device)
        # model.load_state_dict(state_dict["model"])

        if not self.use_softmax:
            prototypes = state_dict["prototypes"].to(self.device)
        model.eval()
        if self.use_mc_dropout:
            model.turn_on_dropout()

        arr_masks = self.dataloader.dataset.arr_masks

        # extract label histogram info from training dataset
        if self.use_cb_sampling or self.n_pixels_by_oracle_cb > 0:
            dict_label_counts = {l: 0 for l in range(self.n_classes)}
            dict_gt_label_counts = {l: 0 for l in range(self.n_classes)}

            dataloader_iter = iter(self.dataloader)
            tbar = tqdm(range(len(self.dataloader)))
            for batch_ind in tbar:
                dict_data = next(dataloader_iter)
                y = dict_data['y'].squeeze(dim=0).numpy()

                # information from ground truth labels
                y_flatten = y.flatten()
                set_unique_gt_labels = set(y_flatten) - {self.ignore_index}
                for ul in set_unique_gt_labels:
                    dict_gt_label_counts[ul] += (y_flatten == ul).sum()

                # information from masked ground truth labels
                y_masked = y.flatten()[arr_masks[batch_ind].flatten()]
                set_unique_labels = set(y_masked) - {self.ignore_index}

                for ul in set_unique_labels:
                    dict_label_counts[ul] += (y_masked == ul).sum()

        list_quries = list()

        if self.n_pixels_by_oracle_cb > 0:
            assert not self.use_cb_sampling, "args use_cb_sampling and use_oracle_cb are not compatible."
            print("Choosing pixels by a class balance sampling based on oracle")
            n_pixels_prev = arr_masks.sum()
            from copy import deepcopy
            dict_class_n_pixels = get_n_per_class(deepcopy(dict_label_counts), len(self.dataloader.dataset) * self.n_pixels_by_oracle_cb)
            for k in list(dict_class_n_pixels.keys()):
                dict_class_n_pixels[k] -= dict_label_counts[k]

            list_labels = list()
            dataloader_iter = iter(self.dataloader)
            tbar = tqdm(range(len(self.dataloader)))
            for batch_ind in tbar:
                dict_data = next(dataloader_iter)
                y = dict_data['y'].squeeze(dim=0).numpy()
                list_labels.append(y)
            arr_labels = np.array(list_labels)
            arr_labels_flat = arr_labels.flatten()

            # exclude pixels that already have a label.
            arr_labels_flat[arr_masks.flatten()] = self.ignore_index

            n, (h, w) = len(self.dataloader.dataset), y.shape
            grid = np.zeros((n, h, w), dtype=np.bool)
            grid_flat = grid.flatten()
            for l in range(self.n_classes):
                ind = np.where(arr_labels_flat == l)[0]
                ind = np.random.choice(ind, dict_class_n_pixels[l], replace=False)
                grid_flat[ind] = True
            grid_oracle_cb = grid_flat.reshape((n, h, w))
            print(f"{grid_oracle_cb.sum()} labelled pixels are selected by an oracle class balance sampling")

            # save this as a new mask
            # arr_masks = grid
            # print(f"# labelled pixels changed from {n_pixels_prev} to {arr_masks.sum()}")

        if self.n_pixels_by_us > 0:
            print("Choosing pixels by a uncertainty sampling")
            dataloader_iter = iter(self.dataloader)
            tbar = tqdm(range(len(self.dataloader)))
            with torch.no_grad():
                for batch_ind in tbar:
                    dict_data = next(dataloader_iter)
                    x = dict_data['x'].to(self.device)
                    y = dict_data['y'].squeeze(dim=0).numpy()   # 360 x 480
                    mask_void_pixels = (y == 11)  # 360 x 480

                    h, w = x.shape[2:]

                    selected_queries_per_img = torch.zeros((h * w)).to(self.device)

                    # if using vote uncertainty sampling
                    if self.use_mc_dropout:
                        grid = torch.zeros((h, w)).to(self.device)  # (h, w)

                        # repeat for mc_n_steps times - set to 20 as a default
                        for step in range(self.mc_n_steps):
                            dict_outputs = model(x)

                            # get softmax probability
                            if self.use_softmax:
                                prob = F.softmax(dict_outputs["pred"], dim=1)

                            else:
                                emb = dict_outputs['emb']  # b x n_emb_dims x h x w
                                _, dist = prediction(emb.detach(), prototypes.detach(), return_distance=True)
                                prob = F.softmax(-dist, dim=1)

                            # get uncertainty map
                            uncertainty_map = self.uncertainty_sampler(prob).squeeze(dim=0)  # h x w

                            # exclude pixels that are already annotated, belong to the void category, or selected by
                            # the oracle class balance sampling.
                            if self.query_strategy in ["entropy", "least_confidence"]:
                                uncertainty_map[arr_masks[batch_ind]] = 0.0
                                uncertainty_map[mask_void_pixels] = 0.0
                                if self.n_pixels_by_oracle_cb > 0:
                                    uncertainty_map[grid_oracle_cb[batch_ind]] = 0.0
                            else:
                                uncertainty_map[arr_masks[batch_ind]] = 1.0
                                uncertainty_map[mask_void_pixels] = 1.0
                                if self.n_pixels_by_oracle_cb > 0:
                                    uncertainty_map[grid_oracle_cb[batch_ind]] = 1.0

                            if self.vote_type == "hard":
                                uncertainty_map = uncertainty_map.view(-1)  # (h * w)

                                ind_best_queries = torch.topk(uncertainty_map,
                                                              k=self.n_pixels_by_us,
                                                              dim=0,
                                                              largest=self.query_strategy in ["entropy", "least_confidence"]).indices  # k

                                grid = grid.view(-1)  # (h * w)
                                for ind in ind_best_queries:
                                    grid[ind] += 1
                                grid.view(h, w)  # h x w

                            else:
                                grid += uncertainty_map  # h x w

                            if self.debug:
                                break

                        grid = grid / self.mc_n_steps  # note that this line is actually not necessary
                        grid = grid.view(-1)  # (h * w)
                        ind_best_queries = torch.topk(grid,
                                                      k=self.n_pixels_by_us,
                                                      dim=0,
                                                      largest=True if self.vote_type == "hard" else self.query_strategy in ["entropy", "least_confidence"]).indices  # k

                    else:
                        dict_outputs = model(x)

                        if self.use_softmax:
                            prob = F.softmax(dict_outputs["pred"], dim=1)

                        else:
                            emb = dict_outputs['emb']  # b x n_emb_dims x h x w
                            _, dist = prediction(emb.detach(), prototypes.detach(), return_distance=True)
                            prob = F.softmax(-dist, dim=1)

                        # get uncertainty map
                        uncertainty_map = self.uncertainty_sampler(prob).squeeze(dim=0)  # h x w

                        # exclude pixels that are already annotated or belong to void category.
                        if self.query_strategy in ["entropy", "least_confidence"]:
                            uncertainty_map[arr_masks[batch_ind]] = 0.0
                            uncertainty_map[mask_void_pixels] = 0.0
                            if self.n_pixels_by_oracle_cb > 0:
                                uncertainty_map[grid_oracle_cb[batch_ind]] = 0.0
                        else:
                            uncertainty_map[arr_masks[batch_ind]] = 1.0
                            uncertainty_map[mask_void_pixels] = 1.0
                            if self.n_pixels_by_oracle_cb > 0:
                                uncertainty_map[grid_oracle_cb[batch_ind]] = 1.0

                        # get top k pixels
                        uncertainty_map = uncertainty_map.view(-1)  # (h * w)
                        ind_best_queries = torch.topk(uncertainty_map,
                                                      k=self.n_pixels_by_us,
                                                      dim=0,
                                                      largest=self.query_strategy in ["entropy",
                                                                                      "least_confidence"]).indices  # k

                    if self.use_cb_sampling:
                        ind_ignore_index = np.where(y.flatten() == self.ignore_index)[0]
                        y_masked = y.flatten()[arr_masks[batch_ind].flatten()]
                        set_unique_labels = set(y_masked)

                        rarest_label = self.ignore_index
                        cnt = np.inf
                        for ul in set_unique_labels:
                            if dict_label_counts[ul] < cnt:
                                rarest_label = ul
                        assert rarest_label != self.ignore_index
                        y_flat = y.flatten()
                        y_flat[~arr_masks[batch_ind].flatten()] = self.ignore_index
                        mask_rarest_label = (y_flat == rarest_label)  # (h * w)

                        emb = dict_outputs["emb"]  # 1 x n_emb_dims x h / s x w / s
                        emb = F.interpolate(emb, size=(h, w), mode='bilinear', align_corners=True).squeeze(dim=0)

                        n_emb_dims = emb.shape[0]
                        emb = emb.view(n_emb_dims, h * w)
                        emb = emb.transpose(1, 0)  # (h * w) x n_emb_dims
                        emb_rarest = emb[mask_rarest_label]  # m x n_emb_dims
                        emb_mean = emb_rarest.mean(dim=0)  # n_emb_dims

                        l2_dist = (emb - emb_mean.unsqueeze(dim=0)).pow(2).sum(dim=1).sqrt()

                        # compute distance from the closest labelled point
                        grid = np.zeros((h, w))
                        grid.fill(np.inf)
                        grid_flat = grid.flatten()

                        grid_loc = list()
                        for i in range(360 * 480):
                            grid_loc.append([i // w, i % w])
                        grid_loc = np.array(grid_loc)

                        list_ind = np.where(mask_rarest_label.flatten())[0]
                        list_ind_2d = {(ind // w, ind % w) for ind in list_ind}

                        for (i, j) in list_ind_2d:
                            dist = ((grid_loc - np.expand_dims(np.array([i, j]), axis=0)) ** 2).sum(axis=1).squeeze()
                            grid_flat = np.where(dist < grid_flat, dist, grid_flat)
                        grid_flat = grid_flat / np.sqrt(h ** 2 + w ** 2)
                        confidence_map = np.exp(-l2_dist.cpu().numpy() * grid_flat)

                        # exclude the center points and the points where the uncertainty sampling already picked.
                        confidence_map[list_ind] = 0.
                        confidence_map[ind_best_queries.cpu().numpy()] = 0.
                        confidence_map[ind_ignore_index] = 0.

                        confidence_map = torch.tensor(confidence_map)

                        ind_topk = torch.topk(confidence_map, k=self.n_pixels_by_us, largest=True).indices

                        for ind in ind_topk:
                            selected_queries_per_img[ind] += 1

                    for ind in ind_best_queries:
                        selected_queries_per_img[ind] += 1

                    selected_queries_per_img = selected_queries_per_img.view(h, w)
                    list_quries.append(selected_queries_per_img.cpu().numpy())

        if len(list_quries) > 0:
            selected_queries = np.array(list_quries).astype(np.bool)
            print(f"{selected_queries.sum()} labelled pixels  are chosen by {self.query_strategy} strategy")

            if self.n_pixels_by_oracle_cb > 0:
                selected_queries = np.maximum(grid_oracle_cb, selected_queries)

        elif self.n_pixels_by_oracle_cb > 0:
            selected_queries = grid_oracle_cb

        else:
            raise NotImplementedError

        # Update labels for query dataloader. Note that this does not update labels for training dataloader.
        self.dataloader.dataset.label_queries(selected_queries, nth_query + 1)

        # remove model file to save memory
        # os.remove(f"{self.dir_checkpoints}/{nth_query}_query/best_miou_model.pt")
        return selected_queries


class UncertaintySampler:
    def __init__(self, query_strategy):
        self.query_strategy = query_strategy

    @staticmethod
    def _entropy(prob):
        return (-prob * torch.log(prob)).sum(dim=1)  # b x h x w

    @staticmethod
    def _least_confidence(prob):
        return 1.0 - prob.max(dim=1)[0]  # b x h x w

    @staticmethod
    def _margin_sampling(prob):
        top2 = prob.topk(k=2, dim=1).values  # b x k x h x w
        return (top2[:, 0, :, :] - top2[:, 1, :, :]).abs()  # b x h x w

    @staticmethod
    def _random(prob):
        b, _, h, w = prob.shape
        return torch.rand((b, h, w))

    def __call__(self, prob):
        return getattr(self, f"_{self.query_strategy}")(prob)


def get_n_per_class(dict_init, n_new_pixels=3670):
    from copy import deepcopy
    dict_prev = deepcopy(dict_init)
    list_asc_keys = [i[0] for i in sorted(dict_init.items(), key=lambda x: x[1])]
    list_prev_k = list()
    cnt = 0

    for i in range(len(list_asc_keys) - 1):
        sorted_dict = dict(sorted(dict_init.items(), key=lambda x: x[1]))
        v_0 = np.array(list(sorted_dict.values())[:-1])
        v_1 = np.array(list(sorted_dict.values())[1:])

        if ((v_0 - v_1) == 0).all():
            n_pixels_per_class = n_new_pixels // len(dict_init)

            for k in list(dict_init.keys()):
                dict_init[k] += n_pixels_per_class

                cnt += n_pixels_per_class

            return dict_init

        k, k_next = list_asc_keys[i], list_asc_keys[i + 1]

        delta = dict_init[k_next] - dict_init[k]
        if cnt + delta * (i + 1) > n_new_pixels:
            n_pixels_per_class = (n_new_pixels - cnt) // (len(list_prev_k) + 1)  # including k
            a = 0
            for k_prev in list_prev_k:
                dict_init[k_prev] += n_pixels_per_class
                cnt += n_pixels_per_class
                a += n_pixels_per_class

            dict_init[k] += n_pixels_per_class
            return dict_init

        elif i == len(list_asc_keys) - 2:
            for j in range(len(list_asc_keys) - 1):
                dict_init[list_asc_keys[j]] += delta
                cnt += delta

            if cnt < n_new_pixels:
                delta = n_new_pixels - cnt
                n_pixels_per_class = delta // len(dict_init)
                for k in list(dict_init.keys()):
                    dict_init[k] += n_pixels_per_class
                    cnt += n_pixels_per_class

            return dict_init

        else:
            for k_prev in list_prev_k:
                dict_init[k_prev] += delta
            dict_init[k] += delta
            assert dict_init[k] == dict_init[k_next], f"{dict_init[k]}, {dict_init[k_next]}"

            list_prev_k.append(k)
            cnt += delta * (i + 1)


            # n labelled pixels per class for class balance considering the all classes (e.g. 11 for camvid)
            # n_labelled_pixels_per_class_cb = (n_labelled_pixels // self.n_classes)
            # print("n_labelled_pixels", n_labelled_pixels, n_labelled_pixels_per_class_cb)
            #
            # # detect classes that have more number of labelled pixels than it should have to keep class balance
            # list_excessive_classes, list_insufficient_classes = list(), list()
            # dict_label_n_excessive_pixels = {l: 0 for l in range(self.n_classes)}
            # n_labelled_pixels_of_insufficient_classes, n_labelled_pixels_of_excessive_classes = 0, 0
            # for l in range(self.n_classes):
            #     if dict_label_counts[l] > n_labelled_pixels_per_class_cb:
            #         dict_label_n_excessive_pixels[l] += (dict_label_counts[l] - n_labelled_pixels_per_class_cb)
            #         list_excessive_classes.append(l)
            #         n_labelled_pixels_of_excessive_classes += dict_label_counts[l]
            #     else:
            #         list_insufficient_classes.append(l)
            #         n_labelled_pixels_of_insufficient_classes += dict_label_counts[l]
            #
            # n_excessive_classes, n_insufficient_classes = len(list_excessive_classes), len(list_insufficient_classes)
            # assert n_insufficient_classes > 0, "n_insufficient_classes should be larger than 0"
            #
            # # compute the number of pixels for each class which lacks labels excluding the excessive classes
            # n_labelled_pixels_per_class_cb = (n_labelled_pixels_of_insufficient_classes + n_labelled_pixels_per_query)
            # n_labelled_pixels_per_class_cb_quotient = n_labelled_pixels_per_class_cb // n_insufficient_classes
            #
            # # adjust the quotient value by considering classes that have # labelled pixels lower than
            # # # labelled pixels per class (for all classes) but higher than # labelled pixels per class
            # # (for insufficient classes). This is kinda confusing but needs to be done to assure # new pixels for cb
            # # to match n_labelled_pixels_per_query.
            # tmp = 0
            # n_classes_temp = 0
            # for l in list_insufficient_classes:
            #     delta = dict_label_counts[l] - n_labelled_pixels_per_class_cb_quotient
            #     print(delta, dict_label_counts[l], n_labelled_pixels_per_class_cb_quotient)
            #     if delta > 0:
            #         tmp += delta
            #         n_classes_temp += 1
            #
            # n_labelled_pixels_per_class_cb -= tmp
            # n_labelled_pixels_per_class_cb_quotient = n_labelled_pixels_per_class_cb // n_insufficient_classes
            # n_labelled_pixels_per_class_cb_remainder = n_labelled_pixels_per_class_cb % n_insufficient_classes
            # print(n_labelled_pixels_per_class_cb_quotient, n_labelled_pixels_per_class_cb_remainder)
            # print(n_labelled_pixels_per_class_cb, n_labelled_pixels_of_excessive_classes, n_labelled_pixels)
            # print(dict_label_counts)
            # assert n_labelled_pixels_per_class_cb + n_labelled_pixels_of_excessive_classes < n_labelled_pixels
            #
            # dict_n_labelled_pixels_per_class_cb = {l: 0 for l in range(self.n_classes)}
            # for l, n in dict_n_labelled_pixels_per_class_cb.items():
            #     if l not in list_excessive_classes:
            #         n_new_labelled_pixels_per_class = n_labelled_pixels_per_class_cb_quotient - dict_label_counts[l]
            #         if n_new_labelled_pixels_per_class > 0:
            #             dict_n_labelled_pixels_per_class_cb[l] = n_new_labelled_pixels_per_class
            # print(dict_n_labelled_pixels_per_class_cb)
            # print(sum(dict_n_labelled_pixels_per_class_cb.values()))
            # exit(12)
            #
            # print(dict_label_counts)
            # print(dict_label_n_excessive_pixels)
            #
            # # list_labels_with_excessive_pixels = [k for k, v in dict_label_n_excessive_pixels.items() if v > 0]
            # n_excessive_pixels = sum([v for v in dict_label_n_excessive_pixels.values()])
            # n_labelled_pixels_per_class_cb = (n_labelled_pixels - n_excessive_pixels) // self.n_classes
            # print(n_excessive_classes, n_excessive_pixels, n_labelled_pixels_per_class_cb)
            #
            # dict_n_labelled_pixels_per_class_cb = {l: 0 for l in range(self.n_classes)}
            # for l, n in dict_n_labelled_pixels_per_class_cb.items():
            #     if l not in n_excessive_classes:
            #         dict_n_labelled_pixels_per_class_cb[l] = (n_labelled_pixels_per_class_cb - dict_label_counts[l])
            # print(sum(dict_n_labelled_pixels_per_class_cb.values()))
            # exit(12)
