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
        self.use_cb_sampling = args.use_cb_sampling
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

        # extract histogram info from training dataset
        if self.use_cb_sampling:
            dict_label_counts = {l: 0 for l in range(self.n_classes)}

            dataloader_iter = iter(self.dataloader)
            tbar = tqdm(range(len(self.dataloader)))
            with torch.no_grad():
                for batch_ind in tbar:
                    dict_data = next(dataloader_iter)
                    y = dict_data['y'].squeeze(dim=0).numpy()
                    y_masked = y.flatten()[arr_masks[batch_ind].flatten()]
                    set_unique_labels = set(y_masked) - {self.ignore_index}

                    for ul in set_unique_labels:
                        dict_label_counts[ul] += (y_masked == ul).sum()

        dataloader_iter = iter(self.dataloader)
        tbar = tqdm(range(len(self.dataloader)))

        list_quries = list()
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
                    k = self.n_pixels_per_query // 2 if self.use_cb_sampling else self.n_pixels_per_query
                    ind_best_queries = torch.topk(grid,
                                                  k=k,
                                                  dim=0,
                                                  largest=True if self.vote_type == "hard" else self.query_strategy in ["entropy", "least_confidence"]).indices  # k

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
                    uncertainty_map = uncertainty_map.view(-1)  # (h * w)
                    k = self.n_pixels_per_query // 2 if self.use_cb_sampling else self.n_pixels_per_query
                    ind_best_queries = torch.topk(uncertainty_map,
                                                  k=k,
                                                  dim=0,
                                                  largest=self.query_strategy in ["entropy", "least_confidence"]).indices  # k

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
                    # print((y_flat == rarest_label).sum())
                    y_flat[~arr_masks[batch_ind].flatten()] = self.ignore_index
                    # print("r", rarest_label, dict_label_counts[rarest_label])
                    mask_rarest_label = (y_flat == rarest_label)  # (h * w)
                    # print("mrl", mask_rarest_label.sum())

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
                    # print('d', dist.min(), dist.max())
                    grid_flat = grid_flat / np.sqrt(h ** 2 + w ** 2)
                    confidence_map = np.exp(-l2_dist.cpu().numpy() * grid_flat)
                    # confidence_map_2 = np.exp(-l2_dist.cpu().numpy())

                    # exclude the center points and the points where the uncertainty sampling already picked.
                    confidence_map[list_ind] = 0.
                    confidence_map[ind_best_queries.cpu().numpy()] = 0.
                    confidence_map[ind_ignore_index] = 0.

                    confidence_map = torch.tensor(confidence_map)

                    ind_topk = torch.topk(confidence_map, k=self.n_pixels_per_query // 2, largest=True).indices
                    # assert (ind_topk != ind_best_queries.cpu().numpy())

                    for ind in ind_topk:
                        selected_queries_per_img[ind] += 1

                    # confidence_map_2[list_ind] = 0.
                    #
                    # confidence_map = torch.tensor(confidence_map)
                    # confidence_map_2 = torch.tensor(confidence_map_2)
                    #
                    # ind_topk = torch.topk(confidence_map, k=100).indices
                    # ind_topk_2 = torch.topk(confidence_map_2, k=100).indices
                    # gt = y.flatten()[ind_topk]
                    # gt_2 = y.flatten()[ind_topk_2]
                    #
                    # print((gt == rarest_label).sum() / 100)
                    # print((gt_2 == rarest_label).sum() / 100)
                    #
                    # exit(12)
                    # connp.zeros_like(confidence_map.numpy())
                    # confidence_map[ind_topk] = 1.0

                    # confidence_map_2[torch.topk(confidence_map_2, k=100).indices]

                    # confidence_map_2 = confidence_map_2.reshape(h, w) * 255
                    # confidence_map_2 = confidence_map_2.astype(np.uint8)

                    # grid = confidence_map.reshape(h, w) * 255
                    # grid = grid.astype(np.uint8)
                    from PIL import Image

                    # Image.fromarray()
                    # x = x.cpu().numpy().squeeze()
                    # x -= x.min()
                    # x = x / x.max()
                    # x *= 255.0
                    # x = x.astype(np.uint8)
                    # x = np.transpose(x, (1, 2, 0))
                    #
                    # Image.fromarray(x).show()
                    # Image.fromarray(confidence_map_2).show()
                    # Image.fromarray(grid).show()
                    # exit(12)
                    # np.random.seed(0)
                    #
                    # h, w = 360, 480
                    # grid = np.zeros((h, w))
                    # grid.fill(np.inf)
                    # grid_flat = grid.flatten()
                    #
                    # grid_loc = list()
                    # for i in range(360 * 480):
                    #     grid_loc.append([i // w, i % w])
                    # grid_loc = np.array(grid_loc)
                    #
                    # N = 100
                    # list_ind = np.random.choice(range(len(grid_flat)), N, replace=False)
                    # list_ind_2d = {(ind // w, ind % w) for ind in list_ind}
                    #
                    # for ind, (i, j) in tqdm(enumerate(list_ind_2d)):
                    #     dist = ((grid_loc - np.expand_dims(np.array([i, j]), axis=0)) ** 2).sum(axis=1).squeeze()
                    #     grid_flat = np.where(dist < grid_flat, dist, grid_flat)
                    #
                    # print(np.exp(- 0.00001 * grid_flat).min())
                    # grid = np.exp(- 0.1 * grid_flat).reshape(h, w) * 255
                    # grid = grid.astype(np.uint8)
                    # Image.fromarray(grid).show()

                    # emb = emb.transpose(1, 0).view(emb, h, w)
                    # print(mask_rarest_label.sum())
                    # print(dict_label_counts, set_unique_labels, rarest_label)
                    # exit(12)

                for ind in ind_best_queries:
                    selected_queries_per_img[ind] += 1

                selected_queries_per_img = selected_queries_per_img.view(h, w)
                list_quries.append(selected_queries_per_img.cpu().numpy())
        selected_queries = np.array(list_quries).astype(np.bool)

        # Update labels for query dataloader. Note that this does not update labels for training dataloader.
        self.dataloader.dataset.label_queries(selected_queries, nth_query + 1)
        # self.dataloader.dataset.dict_label_counts = dict_label_counts

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
