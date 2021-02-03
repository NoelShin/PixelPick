import torch.nn.functional as F


class LocalSimilarity:
    def __init__(self, args):
        self.window_size = 63  # args.window_size
        self.ignore_index = args.ignore_index
        self.h = 360
        self.w = 480

    def _compute_metrics(self):
        return

    def _draw_hist(self):
        return

    def _compute_local_similarity(self, emb, mask):
        k = self.window_size // 2

        for loc in zip(*torch.where(mask)):
            (i, j) = loc
            point_feature = emb[:, i, j]
            # print(emb.shape, k, loc)
            # if i is NOT at the bottommost and j is NOT at the rightmost
            if i < (self.h - k) and j < (self.w - k):
                window = emb[:, max(i - k, 0): i + (k + 1), max(j - k, 0): j + (k + 1)]

            # if i is at the bottommost and j is NOT at the rightmost
            elif i >= (self.h - k) and j < (self.w - k):
                window = emb[:, i - k: min(i + (k + 1), self.h), max(j - k, 0): j + (k + 1)]

            # if j is at the rightmost and i is NOT at the bottommost
            elif i < (self.h - k) and j >= (self.w - k):
                window = emb[:, max(i - k, 0): i + (k + 1), j - k: min(j + (k + 1), self.w)]

            print(window.shape)
            point_feature = point_feature.view((-1, 1, 1)).repeat((1, *window.shape[1:]))
            sim = F.cosine_similarity(point_feature, window, dim=0)
            print(sim.shape, sim.min(), sim.max())
            sim_flat = sim.flatten()
            sim_flat *= 100
            sim_flat = torch.round(sim_flat)  # .to(dtype=torch.int32)
            # print(a_flat.min(), a_flat.max())
            n_bins = 20
            list_bins = torch.arange(-1., 1., n_bins / 200)
            # print(list_bins)
            hist = torch.histc(sim_flat, bins=n_bins, min=-100, max=100)
            # print(hist)
            # print(hist.shape)
            # print(hist[1:] - hist[:-1])
            diff = hist[1:] - hist[:-1]
            threshold = list_bins[torch.where((diff <= 0))[0][-1]]
            # print(threshold)

            # hist_arr = hist.cpu().numpy()
            # import matplotlib.pyplot as plt
            # plt.bar(range(-100, 100, 10), hist_arr)
            # plt.show()
            # plt.close()

            break

        return

    def __call__(self, emb, masked_y):
        batch_masks = (masked_y != self.ignore_index)  # h x w
        emb = F.interpolate(emb, size=(self.h, self.w), mode='bilinear', align_corners=True)
        for b in range(batch_masks.shape[0]):

            self._compute_local_similarity(emb[b], batch_masks[b])

            break

        return


if __name__ == '__main__':
    from sys import path
    path.append("../")
    import numpy as np
    import torch
    import torch.nn.functional as F
    from tqdm import tqdm

    from args import Arguments
    from utils.utils import get_dataloader
    from networks.deeplab import DeepLab

    args = Arguments().parse_args()

    args.use_softmax = True
    args.query_strategy = "random"
    args.n_pixels_by_us = 10

    dataloader = get_dataloader(args, val=False, query=True,
                                shuffle=False, batch_size=args.batch_size, n_workers=args.n_workers)

    device = torch.device("cuda:0")

    torch.manual_seed(0)
    model = DeepLab(args).to(device)
    # state_dict = torch.load("best_miou_model.pt")
    # model.load_state_dict(state_dict["model"])

    # exit(12);
    pseudo_labelling = LocalSimilarity(args)
    masks = torch.tensor(np.load("rand_20.npy")).to(device)

    with torch.no_grad():
        for batch_ind, dict_data in tqdm(enumerate(dataloader)):
            x, y = dict_data['x'].to(device), dict_data['y'].to(device)
            mask = masks[batch_ind: (batch_ind + 1) * args.batch_size]
            y.flatten()[~mask.flatten()] = args.ignore_index
            print(masks.sum(), y.sum(), mask.sum())

            dict_output = model(x)
            emb = dict_output["emb"]

            pseudo_labelling(emb, y)
            break
