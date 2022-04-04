import os
from typing import Dict, List, Optional
from glob import glob
import pickle as pkl
from random import random, randint, uniform
import cv2
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ColorJitter, RandomApply, RandomGrayscale
from torchvision.transforms import InterpolationMode as IM
import torchvision.transforms.functional as TF
from tqdm import tqdm
from query import QuerySelector


class BaseDataset(Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()
        self.list_labelled_queries: Optional[List[np.ndarray]] = None
        self.ignore_index: int

    def label_queries(self, dict_queries: Dict[str, dict], nth_query=None):
        assert len(dict_queries) == len(self.queries), f"{len(dict_queries)} != {len(self.queries)}"

        queries: List[np.ndarray] = QuerySelector.decode_queries(dict_queries)

        previous = self.n_pixels_total

        merged_queries: List[np.ndarray] = list()
        for prev, new in zip(self.queries, queries):
            merged_queries.append(np.logical_or(prev, new))

        new = 0
        for q in merged_queries:
            new += q.sum()

        self.queries = merged_queries
        self.n_pixels_total = new
        print(f"# labelled pixels is changed from {previous} to {new} (delta: {new - previous})")

        if isinstance(nth_query, int):
            os.makedirs(f"{self.dir_checkpoints}/{nth_query}_query", exist_ok=True)
            pkl.dump(dict_queries, open(f"{self.dir_checkpoints}/{nth_query}_query/queries.pkl", "wb"))
            # pkl.dump(self.queries, open(f"{self.dir_checkpoints}/{nth_query}_query/merged_queries.pkl", "wb"))

    def _geometric_augmentations(
            self,
            x: Image.Image,
            y: Optional[Image.Image] = None,
            queries: Optional[torch.Tensor] = None,
            labelled_queries: Optional[torch.Tensor] = None
    ):
        if self.geometric_augmentations["random_scale"]:
            w, h = x.size

            # sampling random scale between 0.5 and 2.0
            rs = uniform(0.5, 2.0)
            w_rs, h_rs = int(w * rs), int(h * rs)

            x = TF.resize(x, (h_rs, w_rs), IM.BILINEAR)

            if y is not None:
                y = TF.resize(y, (h_rs, w_rs), IM.NEAREST)

            if queries is not None:
                queries = TF.resize(queries.unsqueeze(0), (h_rs, w_rs), IM.NEAREST).squeeze(0)

            if labelled_queries is not None:
                labelled_queries = TF.resize(labelled_queries.unsqueeze(0), (h_rs, w_rs), IM.NEAREST).squeeze(0)

        if self.geometric_augmentations["crop"]:
            w, h = x.size
            pad_h, pad_w = max(self.crop_size[0] - h, 0), max(self.crop_size[1] - w, 0)
            self.pad_size = (pad_h, pad_w)

            x = TF.pad(x, (0, 0, pad_w, pad_h), fill=self.mean_val, padding_mode="constant")

            if y is not None:
                y = TF.pad(y, (0, 0, pad_w, pad_h), fill=self.ignore_index, padding_mode="constant")

            if queries is not None:
                queries = TF.pad(queries, (0, 0, pad_w, pad_h), fill=0, padding_mode="constant")

            if labelled_queries is not None:
                labelled_queries = TF.pad(labelled_queries, (0, 0, pad_w, pad_h), fill=self.ignore_index, padding_mode="constant")

            w, h = x.size
            start_h, start_w = randint(0, h - self.crop_size[0]), randint(0, w - self.crop_size[1])

            x = TF.crop(x, top=start_h, left=start_w, height=self.crop_size[0], width=self.crop_size[1])

            if y is not None:
                y = TF.crop(y, top=start_h, left=start_w, height=self.crop_size[0], width=self.crop_size[1])

            if queries is not None:
                queries = TF.crop(queries, top=start_h, left=start_w, height=self.crop_size[0], width=self.crop_size[1])

            if labelled_queries is not None:
                labelled_queries = TF.crop(
                    labelled_queries, top=start_h, left=start_w, height=self.crop_size[0], width=self.crop_size[1]
                )

        if self.geometric_augmentations["random_hflip"]:
            if random() > 0.5:
                x = TF.hflip(x)

                if y is not None:
                    y = TF.hflip(y)

                if queries is not None:
                    queries = TF.hflip(queries)

                if labelled_queries is not None:
                    labelled_queries = TF.hflip(labelled_queries)

        if queries is not None:
            queries = torch.from_numpy(np.asarray(queries, dtype=np.uint8) // 255)
        else:
            queries = torch.tensor(0)

        if labelled_queries is not None:
            labelled_queries = labelled_queries
        else:
            labelled_queries = torch.tensor(-1)
        return x, y, queries, labelled_queries

    def _photometric_augmentations(self, x):
        if self.photometric_augmentations["random_color_jitter"]:
            color_jitter = ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2)
            x = RandomApply([color_jitter], p=0.8)(x)

        if self.photometric_augmentations["random_grayscale"]:
            x = RandomGrayscale(0.2)(x)

        if self.photometric_augmentations["random_gaussian_blur"]:
            w, h = x.size
            smaller_length = min(w, h)
            x = GaussianBlur(kernel_size=int((0.1 * smaller_length // 2 * 2) + 1))(x)
        return x

    def update_labelled_queries(self, labelled_queries: List[np.ndarray]) -> None:
        """
        :param labelled_queries: Human-labelled queries.
        :return:
        """
        self.list_labelled_queries: List[np.ndarray] = labelled_queries
        return

    def __getitem__(self, ind):
        dict_data = dict()
        p_img: str = self.list_inputs[ind]
        x = Image.open(p_img).convert("RGB")
        y = Image.open(self.list_labels[ind])

        # get queries
        if self.queries is not None:
            queries = torch.tensor(self.queries[ind].astype(np.uint8)) * 255  # Image.fromarray(self.queries[ind].astype(np.uint8) * 255)

        else:
            w, h = x.size
            queries = torch.ones((h, w), dtype=torch.uint8) * 255

        # get human-labelled queries
        if self.list_labelled_queries is not None:
            labelled_queries: torch.Tensor = torch.tensor(self.list_labelled_queries[ind])
        else:
            w, h = x.size
            labelled_queries: torch.Tensor = torch.zeros((h, w), dtype=torch.uint8)

        # if val or query dataset, do NOT do data augmentation
        if not self.val and not self.query:
            # data augmentation
            x, y, queries, labelled_queries = self._geometric_augmentations(
                x, y, queries=queries, labelled_queries=labelled_queries
            )
            x = self._photometric_augmentations(x)
        else:
            queries = torch.from_numpy(np.asarray(queries, dtype=np.uint8) // 255)

        dict_data.update({
            'x': TF.normalize(TF.to_tensor(x), self.mean, self.std),
            'y': torch.tensor(np.asarray(y, np.int64), dtype=torch.long),
            "queries": queries,
            "labelled_queries": labelled_queries,
            "p_img": p_img
        })
        return dict_data


class GaussianBlur(object):
    # Implements Gaussian blur as described in the SimCLR paper
    def __init__(self, kernel_size, min=0.1, max=2.0):
        self.min = min
        self.max = max
        # kernel size is set to be 10% of the image height/width
        self.kernel_size = kernel_size

    def __call__(self, sample):
        sample = np.array(sample)

        # blur the image with a 50% chance
        prob = np.random.random_sample()

        if prob < 0.5:
            sigma = (self.max - self.min) * np.random.random_sample() + self.min
            sample = cv2.GaussianBlur(sample, (self.kernel_size, self.kernel_size), sigma)
        return sample