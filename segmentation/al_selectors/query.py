import os
import pickle as pkl

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

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
        self.use_openset = args.use_openset
        self.query_strategy = args.query_strategy
        self.use_cb_sampling = args.use_cb_sampling
        self.use_mc_dropout = args.use_mc_dropout
        self.vote_type = args.vote_type

        self.top_n_percent = args.top_n_percent

        self.query_stats = QueryStats(args)

    def _select_queries(self, uc_map):
        h, w = uc_map.shape[-2:]
        uc_map = uc_map.flatten()
        k = int(h * w * self.top_n_percent) if self.top_n_percent > 0. else self.n_pixels_by_us
        ind_queries = uc_map.topk(k=k,
                                  dim=0,
                                  largest=self.query_strategy in ["entropy", "least_confidence"]).indices.cpu().numpy()
        if self.top_n_percent > 0.:
            ind_queries = np.random.choice(ind_queries, self.n_pixels_by_us, False)

        query = np.zeros((h * w), dtype=np.bool)
        query[ind_queries] = True
        query = query.reshape((h, w))
        return query

    def __call__(self, nth_query, model, prototypes=None):
        arr_masks = self.dataloader.dataset.arr_masks

        model.eval()
        if self.use_mc_dropout:
            model.turn_on_dropout()

        print(f"Choosing pixels by {self.query_strategy}")
        list_queries = list()
        with torch.no_grad():
            for batch_ind, dict_data in tqdm(enumerate(self.dataloader)):
                x = dict_data['x'].to(self.device)
                y = dict_data['y'].squeeze(dim=0).numpy()   # 360 x 480
                mask = arr_masks[batch_ind]
                mask_void = (y == 11)  # 360 x 480
                h, w = x.shape[2:]

                # get uncertainty map
                if self.use_mc_dropout:
                    uc_map = torch.zeros((h, w)).to(self.device)  # (h, w)
                    prob = torch.zeros((x.shape[0], self.n_classes, h, w)).to(self.device)  # b x c x h x w
                    # repeat for mc_n_steps times - set to 20 as a default
                    for step in range(self.mc_n_steps):
                        prob_ = F.softmax(model(x)["pred"], dim=1)

                        uc_map_ = self.uncertainty_sampler(prob_).squeeze(dim=0)  # h x w
                        uc_map += uc_map_
                        prob += prob_

                    prob = prob / self.mc_n_steps

                else:
                    prob = F.softmax(model(x)["pred"], dim=1)

                    uc_map = self.uncertainty_sampler(prob).squeeze(dim=0)  # h x w

                # exclude pixels that are already annotated, belong to the void category
                uc_map[mask] = 0.0 if self.query_strategy in ["entropy", "least_confidence"] else 1.0
                uc_map[mask_void] = 0.0 if self.query_strategy in ["entropy", "least_confidence"] else 1.0

                # select queries
                query = self._select_queries(uc_map)
                list_queries.append(query)

                self.query_stats.update(query, y, prob)

                # if self.debug:
                #     break

        self.query_stats.save(nth_query)

        assert len(list_queries) > 0, f"no queries are chosen!"
        queries = np.stack(list_queries, axis=0)
        print(f"{queries.sum()} labelled pixels  are chosen by {self.query_strategy} strategy")

        # Update labels for query dataloader. Note that this does not update labels for training dataloader.
        self.dataloader.dataset.label_queries(queries, nth_query)

        return queries


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


class QueryStats:
    def __init__(self, args):
        self.dir_checkpoints = f"{args.dir_root}/checkpoints/{args.experim_name}"
        self.list_entropy, self.list_n_unique_labels, self.list_spatial_coverage = list(), list(), list()
        self.dict_label_cnt = {l: 0 for l in range(args.n_classes)}

    def _count_labels(self, query, y):
        for l in y.flatten()[query.flatten()]:
            self.dict_label_cnt[l] += 1

    def _get_entropy(self, query, prob):
        ent_map = (-prob * torch.log(prob)).sum(dim=1).cpu().numpy()  # h x w
        pixel_entropy = ent_map.flatten()[query.flatten()]  # n_pixels_per_img
        return pixel_entropy.tolist()

    def _n_unique_labels(self, query, y):
        return len(set(y.flatten()[query.flatten()]))

    def _spatial_coverage(self, query):
        x_loc, y_loc = np.where(query)
        x_loc, y_loc = np.expand_dims(x_loc, axis=1), np.expand_dims(y_loc, axis=1)
        x_loc_t, y_loc_t = x_loc.transpose(), y_loc.transpose()
        dist = np.sqrt((x_loc - x_loc_t) ** 2 + (y_loc - y_loc_t) ** 2)
        try:
            dist = dist[~np.eye(dist.shape[0], dtype=np.bool)].reshape(dist.shape[0], -1).sum() / 2
        except ValueError:
            return np.NaN
        return dist

    def save(self, nth_query):
        dict_stats = {
            "label_distribution": self.dict_label_cnt,
            "avg_entropy": np.mean(self.list_entropy),
            "avg_n_unique_labels": np.mean(self.list_n_unique_labels),
            "avg_spatial_coverage": np.mean(self.list_spatial_coverage)
        }

        for k, v in dict_stats.items():
            print(f"{k}: {v}")

        pkl.dump(dict_stats, open(f"{self.dir_checkpoints}/{nth_query}_query/query_stats.pkl", "wb"))

    def update(self, query, y, prob):
        # count labels
        self._count_labels(query, y)

        # entropy
        self.list_entropy.extend(self._get_entropy(query, prob))

        # n_unique_labels
        self.list_n_unique_labels.append(self._n_unique_labels(query, y))

        # spatial_coverage
        self.list_spatial_coverage.append(self._spatial_coverage(query))
        return


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

