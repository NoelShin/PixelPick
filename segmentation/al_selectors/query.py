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
        self.n_pixels_per_query = args.n_pixels_per_query
        self.network_name = args.network_name
        self.uncertainty_sampler = UncertaintySampler(args.query_strategy)
        self.use_softmax = args.use_softmax
        self.query_strategy = args.query_strategy
        self.use_mc_dropout = args.use_mc_dropout
        self.vote_type = args.vote_type

    def __call__(self, nth_query):
        state_dict = torch.load(f"{self.dir_checkpoints}/{nth_query}_query/best_miou_model.pt")
        if self.network_name == "FPN":
            model = FPNSeg(self.args).to(self.device)
        else:
            model = DeepLab(self.args).to(self.device)
        model.load_state_dict(state_dict["model"])

        if not self.use_softmax:
            prototypes = state_dict["prototypes"].to(self.device)
        model.eval()
        if self.use_mc_dropout:
            model.turn_on_dropout()

        arr_masks = self.dataloader.dataset.arr_masks

        dataloader_iter = iter(self.dataloader)
        tbar = tqdm(range(len(self.dataloader)))

        list_quries = list()
        with torch.no_grad():
            for batch_ind in tbar:
                dict_data = next(dataloader_iter)
                x = dict_data['x'].to(self.device)
                y = dict_data['y']  # 1 x 360 x 480
                mask_void_pixels = (y == 11).squeeze(dim=0).numpy()  # 360 x 480

                h, w = x.shape[2:]

                selected_queries_per_img = torch.zeros((h * w)).to(self.device)

                # if using vote uncertainty sampling
                if self.use_mc_dropout:
                    # selected_queries = torch.zeros((h * w)).to(self.device)
                    # grid = torch.zeros((self.n_classes, x.shape[2], x.shape[3])).to(self.device)
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

                        # exclude pixels that are already annotated.
                        uncertainty_map[arr_masks[batch_ind]] = 0.0 if self.query_strategy in ["entropy", "least_confidence"] else 1.0

                        # exclude pixels that belong to void category.
                        uncertainty_map[mask_void_pixels] = 0.0 if self.query_strategy in ["entropy", "least_confidence"] else 1.0

                        if self.vote_type == "hard":
                            # h, w = uncertainty_map.shape
                            uncertainty_map = uncertainty_map.view(-1)  # (h * w)

                            ind_best_queries = torch.topk(uncertainty_map,
                                                          k=self.n_pixels_per_query,
                                                          dim=0,
                                                          largest=self.query_strategy == "entropy").indices  # k

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
                                                  k=self.n_pixels_per_query,
                                                  dim=0,
                                                  largest=True if self.vote_type == "hard" else self.query_strategy == "entropy").indices  # k

                else:
                    dict_outputs = model(x)

                    if self.use_softmax:
                        prob = dict_outputs["pred"]

                    else:
                        emb = dict_outputs['emb']  # b x n_emb_dims x h x w
                        _, dist = prediction(emb.detach(), prototypes.detach(), return_distance=True)
                        prob = F.softmax(-dist, dim=1)

                    # get uncertainty map
                    uncertainty_map = self.uncertainty_sampler(prob).squeeze(dim=0)  # h x w

                    # exclude pixels that are already annotated.
                    uncertainty_map[arr_masks[batch_ind]] = 0.0 if self.query_strategy in ["entropy", "least_confidence"] else 1.0

                    # exclude pixels that belong to void category.
                    uncertainty_map[mask_void_pixels] = 0.0 if self.query_strategy in ["entropy", "least_confidence"] else 1.0

                    # get top k pixels
                    uncertainty_map = uncertainty_map.view(-1)
                    ind_best_queries = torch.topk(uncertainty_map,
                                                  k=self.n_pixels_per_query,
                                                  dim=0,
                                                  largest=self.query_strategy in ["entropy", "least_confidence"]).indices  # k
                for ind in ind_best_queries:
                    selected_queries_per_img[ind] += 1

                selected_queries_per_img = selected_queries_per_img.view(h, w)
                list_quries.append(selected_queries_per_img.cpu().numpy())
        selected_queries = np.array(list_quries).astype(np.bool)

        # Update labels for query dataloader. Note that this does not update labels for training dataloader.
        self.dataloader.dataset.label_queries(selected_queries, nth_query + 1)

        # remove model file to save memory
        os.remove(f"{self.dir_checkpoints}/{nth_query}_query/best_miou_model.pt")
        return selected_queries


class UncertaintySampler:
    def __init__(self, query_strategy):
        self.query_strategy = query_strategy

    def _entropy(self, prob):
        return (-prob * torch.log(prob)).sum(dim=1)  # b x h x w

    def _least_confidence(self, prob):
        return 1.0 - prob.max(dim=1)[0]  # b x h x w

    def _margin_sampling(self, prob):
        top2 = prob.topk(k=2, dim=1).values  # b x k x h x w
        return (top2[:, 0, :, :] - top2[:, 1, :, :]).abs()  # b x h x w

    def _random(self, prob):
        b, _, h, w = prob.shape
        return torch.rand((b, h, w))

    def __call__(self, prob):
        return getattr(self, f"_{self.query_strategy}")(prob)
