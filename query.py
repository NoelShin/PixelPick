import os
from typing import Tuple, Dict, List, Union
from pathlib import Path
import pickle as pkl
from math import ceil
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


class QuerySelector:
    def __init__(self, args, dataloader, device=torch.device("cuda:0")):
        self.dataset_name = args.dataset_name
        self.dataloader = dataloader
        self.debug = args.debug
        self.device = device
        self.dir_checkpoints = f"{args.dir_root}/checkpoints/{args.experim_name}"
        self.ignore_index = args.ignore_index
        self.mc_n_steps = args.mc_n_steps
        self.n_classes = args.n_classes
        self.n_pixels_by_us = args.n_pixels_by_us
        self.network_name = args.network_name
        self.query_stats = QueryStats(args)
        self.query_strategy = args.query_strategy
        self.reverse_order = args.reverse_order
        self.stride_total = args.stride_total
        self.top_n_percent = args.top_n_percent
        self.uncertainty_sampler = UncertaintySampler(args.query_strategy)
        self.use_mc_dropout = args.use_mc_dropout
        self.vote_type = args.vote_type

    def _select_queries(self, uc_map) -> np.ndarray:
        h, w = uc_map.shape[-2:]
        uc_map = uc_map.flatten()
        k = int(h * w * self.top_n_percent) if self.top_n_percent > 0. else self.n_pixels_by_us

        if self.reverse_order:
            assert self.top_n_percent > 0.
            ind_queries = np.random.choice(range(h * w), k, False)
            sampling_mask = np.zeros((h * w), dtype=np.bool)
            sampling_mask[ind_queries] = True
            sampling_mask = torch.tensor(sampling_mask, dtype=torch.bool, device=self.device)

            if self.query_strategy in ["entropy", "least_confidence"]:
                uc_map[~sampling_mask] = 0.
            else:
                uc_map[~sampling_mask] = 1.0

            ind_queries = uc_map.topk(
                k=self.n_pixels_by_us,
                dim=0,
                largest=self.query_strategy in ["entropy", "least_confidence"]
            ).indices.cpu().numpy()

        else:
            ind_queries = uc_map.topk(
                k=k,
                dim=0,
                largest=self.query_strategy in ["entropy", "least_confidence"]
            ).indices.cpu().numpy()

            if self.top_n_percent > 0.:
                ind_queries = np.random.choice(ind_queries, self.n_pixels_by_us, False)

        query = np.zeros((h * w), dtype=np.bool)
        query[ind_queries] = True
        query = query.reshape((h, w))
        return query

    @staticmethod
    def encode_query(
            p_img: str,
            size: Tuple[int, int],  # (h, w) of the image
            query: np.ndarray
    ) -> Dict[str, dict]:
        y_coords, x_coords = np.where(query)

        query_info: Dict[str, dict] = {
            p_img: {
                "height": size[0],
                "width": size[1],
                "x_coords": x_coords,
                "y_coords": y_coords
            }
        }
        return query_info

    @staticmethod
    def decode_queries(
            encoded_query: Dict[str, dict],
            ignore_index: int = 255,
            return_as_dict: bool = False
    ) -> Union[List[np.ndarray], Dict[str, np.ndarray]]:
        def decode_query(query_info: dict, ignore_index: int = 255) -> np.ndarray:
            queried_pixels = zip(query_info["y_coords"], query_info["x_coords"])
            labels: List[int] = query_info.get("category_id", None)
            if labels is None:
                query: np.ndarray = np.zeros((query_info["height"], query_info["width"]), dtype=np.bool)
            else:
                query: np.ndarray = ignore_index * np.ones((query_info["height"], query_info["width"]), dtype=np.int64)

            for i, loc in enumerate(queried_pixels):
                query[loc] = labels[i] if labels is not None else True
            return query

        if len(encoded_query) > 1:
            if return_as_dict:
                queries: Dict[str, np.ndarray] = dict()
                for p_img, query_info in sorted(encoded_query.items()):
                    queries.update({
                        p_img: decode_query(query_info, ignore_index)
                    })
            else:
                queries: List[np.ndarray] = list()
                for p_img, query_info in sorted(encoded_query.items()):
                    queries.append(decode_query(query_info, ignore_index))

        elif len(encoded_query) == 1:
            if return_as_dict:
                queries: Dict[str, np.ndarray] = {
                    list(encoded_query.keys())[0]: decode_query(list(encoded_query.values())[0], ignore_index)
                }

            else:
                queries: List[np.ndarray] = [decode_query(list(encoded_query.values())[0], ignore_index)]

        else:
            raise ValueError(len(encoded_query))
            # queries = [decode_query(list(encoded_query.values())[0])]
        # if isinstance(encoded_query, dict):
        #     queries: List[np.ndarray] = list()
        #     for p_img, query_info in encoded_query.items():
        #         queries.append(decode_query(query_info))
        #
        # elif isinstance(encoded_query, dict):
        #     queries = [decode_query(list(encoded_query.values())[0])]
        #     # queries = [decode_query(list(encoded_query.values())[0])]

        # else:
        #     raise TypeError(type(encoded_query))
        return queries

    def __call__(self, nth_query, model, human_labels: bool = False):
        if human_labels:
            prev_queries = self.dataloader.dataset.list_labelled_queries
        else:
            prev_queries = self.dataloader.dataset.queries

        model.eval()
        if self.use_mc_dropout:
            model.turn_on_dropout()

        print(f"Choosing pixels by {self.query_strategy}")
        list_queries, n_pixels = list(), 0
        dict_queries: dict = dict()

        with torch.no_grad():
            for batch_ind, dict_data in tqdm(enumerate(self.dataloader)):
                x = dict_data['x'].to(self.device)
                y = dict_data.get('y', None)
                mask = prev_queries[batch_ind]  # h x w

                if y is not None:
                    y = y.squeeze(dim=0).numpy()  # h x w
                    mask_void = (y == self.ignore_index)  # h x w

                h, w = x.shape[2:]

                # voc
                if self.dataset_name == "voc":
                    pad_h = ceil(h / self.stride_total) * self.stride_total - h  # x.shape[2]
                    pad_w = ceil(w / self.stride_total) * self.stride_total - w  # x.shape[3]
                    x = F.pad(x, pad=(0, pad_w, 0, pad_h), mode='reflect')

                # get uncertainty map
                if self.use_mc_dropout:
                    uc_map = torch.zeros((h, w)).to(self.device)  # (h, w)
                    prob = torch.zeros((x.shape[0], self.n_classes, h, w)).to(self.device)  # b x c x h x w
                    # repeat for mc_n_steps times - set to 20 as a default
                    for step in range(self.mc_n_steps):
                        prob_ = F.softmax(model(x)["pred"], dim=1)[:, :, :h, :w]
                        uc_map_ = self.uncertainty_sampler(prob_).squeeze(dim=0)  # h x w
                        uc_map += uc_map_
                        prob += prob_
                    up_map = up_map / self.mc_n_steps
                    prob = prob / self.mc_n_steps

                else:
                    prob = F.softmax(model(x)["pred"][:, :, :h, :w], dim=1)

                    uc_map = self.uncertainty_sampler(prob).squeeze(dim=0)  # h x w

                # exclude pixels that are already annotated, belong to the void category
                if human_labels:
                    uc_map[mask != self.ignore_index] = 0.0 if self.query_strategy in ["entropy", "least_confidence"] else 1.0
                else:
                    uc_map[mask] = 0.0 if self.query_strategy in ["entropy", "least_confidence"] else 1.0

                if y is not None:
                    uc_map[mask_void] = 0.0 if self.query_strategy in ["entropy", "least_confidence"] else 1.0

                # select queries
                query: np.ndarray = self._select_queries(uc_map)
                list_queries.append(query)
                n_pixels += query.sum()

                if not human_labels and y is not None:
                    self.query_stats.update(query, y, prob)
                query_info: dict = self.encode_query(dict_data["p_img"][0], size=(h, w), query=query)

                dict_queries.update(query_info)

        assert len(list_queries) > 0, f"no queries are chosen!"
        if not human_labels and y is not None:
            self.query_stats.save(nth_query)
            print(f"{n_pixels} labelled pixels  are chosen by {self.query_strategy} strategy")

            # Update labels for query dataloader. Note that this does not update labels for training dataloader.
            self.dataloader.dataset.label_queries(dict_queries, nth_query)
        return dict_queries


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

    @staticmethod
    def _get_entropy(query, prob):
        ent_map = (-prob * torch.log(prob)).sum(dim=1).cpu().numpy()  # h x w
        pixel_entropy = ent_map.flatten()[query.flatten()]  # n_pixels_per_img
        return pixel_entropy.tolist()

    @staticmethod
    def _n_unique_labels(query, y):
        return len(set(y.flatten()[query.flatten()]))

    @staticmethod
    def _spatial_coverage(query):
        x_loc, y_loc = np.where(query)
        x_loc, y_loc = np.expand_dims(x_loc, axis=1), np.expand_dims(y_loc, axis=1)
        x_loc_t, y_loc_t = x_loc.transpose(), y_loc.transpose()
        dist = np.sqrt((x_loc - x_loc_t) ** 2 + (y_loc - y_loc_t) ** 2)
        try:
            dist = dist[~np.eye(dist.shape[0], dtype=np.bool)].reshape(dist.shape[0], -1).mean()
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

        os.makedirs(f"{self.dir_checkpoints}/{nth_query}_query", exist_ok=True)
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


def gather_previous_query_files(dir_base: str, ext="pkl") -> List[str]:
    list_pkl_files = [str(p) for p in Path(dir_base).rglob(f"*/queries.{ext}" if ext is not None else "*")]
    return list_pkl_files


def merge_previous_query_files(
        list_previous_query_files: List[str],
        ignore_index: int,
        verbose: bool = True
) -> Dict[str, np.ndarray]:

    list_prev_queries: List[Dict[str, np.ndarray]] = list()
    for p_prev_query_file in list_previous_query_files:
        prev_query_file: dict = pkl.load(open(p_prev_query_file, "rb"))
        img_path_to_queries: Dict[str, np.ndarray] = QuerySelector.decode_queries(
            prev_query_file, ignore_index=ignore_index, return_as_dict=True
        )
        list_prev_queries.append(img_path_to_queries)

    # collect queries across different previous query files with their path as a key.
    all_img_path_to_queries: Dict[str, List[np.ndarray]] = dict()
    for img_path_to_queries in list_prev_queries:
        for img_path, queries in img_path_to_queries.items():
            try:
                all_img_path_to_queries[img_path].append(queries)
            except KeyError:
                all_img_path_to_queries.update({img_path: [queries]})
    # merge the annotations for each image such that each image has only one corresponding query file.
    cnt = 0
    img_path_to_merged_queries: Dict[str, np.ndarray] = dict()
    for p_img, list_queries in all_img_path_to_queries.items():
        assert p_img not in img_path_to_merged_queries, f"{p_img} already exists in img_path_to_merged_queries file."
        merged_query: np.ndarray = ignore_index * np.ones_like(list_queries[0], dtype=np.int64)
        for query in list_queries:
            merged_query[query != ignore_index] = query[query != ignore_index]
            cnt += (query != ignore_index).sum()
        img_path_to_merged_queries.update({p_img: merged_query})

    if verbose:
        print(f"# merged pixels: {cnt}")
    return img_path_to_merged_queries


if __name__ == '__main__':
    from argparse import Namespace
    from copy import deepcopy
    import random
    import yaml
    from torch.utils.data import DataLoader
    from args import Arguments
    from utils.utils import get_dataloader, get_model

    arguments_parser = Arguments()
    arguments_parser.parser.add_argument("--p_state_dict", type=str, default='', help="path to a state_dict file")
    arguments_parser.parser.add_argument(
        "--p_dataset_config", '-pdc', type=str, default="/Users/noel/Desktop/pixelpick/datasets/configs/custom.yaml"
    )
    args = arguments_parser.parse_args(verbose=True)

    # dataset_config = yaml.safe_load(open(f"{args.p_dataset_config}", 'r'))
    # args: dict = vars(args)
    # args.update(dataset_config)
    # args: Namespace = Namespace(**args)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.p_state_dict != '':
        model = get_model(args).to(device)
        state_dict: dict = torch.load(args.p_state_dict)["model"]
        model.load_state_dict(state_dict)
        print(f"pretrained model is loaded from {args.p_state_dict}")

        list_prev_query_files: List[str] = gather_previous_query_files(args.dir_checkpoints)
        img_path_to_merged_query: Dict[str, np.ndarray] = merge_previous_query_files(
            list_prev_query_files, ignore_index=args.ignore_index
        )

        # change the path
        list_inputs: List[str] = list()
        list_merged_queries: List[np.ndarray] = list()
        for p_img, merged_query in sorted(img_path_to_merged_query.items()):
            filename: str = p_img.split('/')[-1]
            p_img = f"{args.dir_dataset}/train/{filename}"
            assert os.path.exists(p_img)
            list_inputs.append(p_img)
            list_merged_queries.append(merged_query)

        # change the images that the dataloader loads depending on the images that have annotation.
        dataset = get_dataloader(
            deepcopy(args),
            query=True,
            val=False,
            generate_init_queries=False,
            shuffle=False,
            batch_size=1,
            n_workers=args.n_workers
        ).dataset

        dataset.list_inputs = list_inputs
        dataset.update_labelled_queries(list_merged_queries)

        dataloader = DataLoader(
            dataset,
            batch_size=1,
            num_workers=args.n_workers,
            shuffle=False
        )

        nth_query: int = len(list_prev_query_files)

        qs = QuerySelector(args, dataloader, device=device)
        dict_queries: Dict[str, dict] = qs(nth_query=nth_query, model=model, human_labels=True)
        os.makedirs(f"{args.dir_checkpoints}/{nth_query}_query", exist_ok=True)
        pkl.dump(dict_queries, open(f"{args.dir_checkpoints}/{nth_query}_query/queries.pkl", "wb"))
        print(f"Queries are saved at {args.dir_checkpoints}/{nth_query}_query/queries.pkl")

    else:
        dataloader = get_dataloader(
            deepcopy(args),
            query=True,
            val=False,
            generate_init_queries=True,
            shuffle=False,
            batch_size=1,
            n_workers=args.n_workers
        )
        nth_query: int = 0
