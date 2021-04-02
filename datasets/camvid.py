import os
from glob import glob
from random import random, randint, uniform

import cv2
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ColorJitter, RandomApply, RandomGrayscale
import torchvision.transforms.functional as TF
from tqdm import tqdm


class CamVidDataset(Dataset):
    def __init__(self, args, val=False, query=False):
        super(CamVidDataset, self).__init__()
        assert os.path.isdir(args.dir_dataset), f"{args.dir_dataset} does not exist."
        self.dir_checkpoints = f"{args.dir_root}/checkpoints/{args.experim_name}"
        self.seed = args.seed

        # get data paths
        mode = "test" if val else "train"
        self.list_inputs = sorted(glob(f"{args.dir_dataset}/{mode}/*.png"))
        self.list_labels = sorted(glob(f"{args.dir_dataset}/{mode}annot/*.png"))
        assert len(self.list_inputs) == len(self.list_labels) and len(self.list_inputs) > 0

        # seg variables for data augmentation
        self.geometric_augmentations = args.augmentations["geometric"]
        self.photometric_augmentations = args.augmentations["photometric"]
        self.mean, self.std = args.mean, args.std
        self.crop_size, self.pad_size = (360, 480), (0, 0)
        self.ignore_index = args.ignore_index
        if self.geometric_augmentations["crop"]:
            self.mean_val = tuple((np.array(self.mean) * 255.0).astype(np.uint8).tolist())

        self.queries, self.n_pixels_total = None, -1
        n_pixels_per_img = args.n_pixels_by_us

        if n_pixels_per_img != 0 and not val:
            list_masks = list()
            np.random.seed(self.seed)

            label = Image.open(self.list_labels[0])
            w, h = label.size

            # iterate over labels to get masks
            for i in tqdm(range(len(self.list_labels))):
                label = np.array(Image.open(self.list_labels[i]))  # H x W

                # exclude pixels with ignore_index
                ind_non_void_pixels = np.where(label.flatten() != self.ignore_index)[0]
                ind_chosen_pixels = np.random.choice(ind_non_void_pixels, n_pixels_per_img, replace=False)

                mask_flat = np.zeros((h, w), dtype=np.bool).flatten()
                mask_flat[ind_chosen_pixels] = True
                list_masks.append(mask_flat.reshape((h, w)))

            self.queries = np.array(list_masks, dtype=np.bool)
            self.n_pixels_total = self.queries.sum()

            # save initial labelled pixels for a future reproduction
            os.makedirs(f"{self.dir_checkpoints}/0_query", exist_ok=True)
            np.save(f"{self.dir_checkpoints}/0_query/queries.npy", self.queries)
            print("# labelled pixels used for training:", self.n_pixels_total)

        self.val = val
        self.query = query

    def label_queries(self, queries, nth_query=None):
        assert len(queries) == len(self.queries), f"{queries.shape}, {self.queries.shape}"
        previous = self.queries.sum()
        self.queries = np.logical_or(self.queries, queries)
        self.n_pixels_total = new = self.queries.sum()
        print(f"# labelled pixels is changed from {previous} to {new} (delta: {new - previous})")

        if isinstance(nth_query, int):
            os.makedirs(f"{self.dir_checkpoints}/{nth_query}_query", exist_ok=True)
            np.save(f"{self.dir_checkpoints}/{nth_query}_query/label.npy", self.queries)

    def _geometric_augmentations(self, x, y, queries=None):
        if self.geometric_augmentations["random_scale"]:
            w, h = x.size

            # sampling random scale between 0.5 and 2.0
            rs = uniform(0.5, 2.0)
            w_rs, h_rs = int(w * rs), int(h * rs)

            x, y = TF.resize(x, (h_rs, w_rs), Image.BILINEAR), TF.resize(y, (h_rs, w_rs), Image.NEAREST)

            if queries is not None:
                queries = TF.resize(queries, (h_rs, w_rs), Image.NEAREST)

        if self.geometric_augmentations["crop"]:
            w, h = x.size
            pad_h, pad_w = max(self.crop_size[0] - h, 0), max(self.crop_size[1] - w, 0)
            self.pad_size = (pad_h, pad_w)

            x = TF.pad(x, (0, 0, pad_w, pad_h), fill=self.mean_val, padding_mode="constant")
            y = TF.pad(y, (0, 0, pad_w, pad_h), fill=self.ignore_index, padding_mode="constant")

            if queries is not None:
                queries = TF.pad(queries, (0, 0, pad_w, pad_h), fill=0, padding_mode="constant")

            w, h = x.size
            start_h, start_w = randint(0, h - self.crop_size[0]), randint(0, w - self.crop_size[1])

            x = TF.crop(x, top=start_h, left=start_w, height=self.crop_size[0], width=self.crop_size[1])
            y = TF.crop(y, top=start_h, left=start_w, height=self.crop_size[0], width=self.crop_size[1])

            if queries is not None:
                queries = TF.crop(queries, top=start_h, left=start_w, height=self.crop_size[0], width=self.crop_size[1])

        if self.geometric_augmentations["random_hflip"]:
            if random() > 0.5:
                x, y = TF.hflip(x), TF.hflip(y)

                if queries is not None:
                    queries = TF.hflip(queries)

        if queries is not None:
            queries = torch.from_numpy(np.asarray(queries, dtype=np.uint8) // 255)
        else:
            queries = torch.tensor(0)

        return x, y, queries

    def _photometric_augmentations(self, x):
        if self.photometric_augmentations["random_color_jitter"]:
            color_jitter = ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2)
            x = RandomApply([color_jitter], p=0.8)(x)

        if self.photometric_augmentations["random_grayscale"]:
            x = RandomGrayscale(0.2)(x)

        if self.photometric_augmentations["random_gaussian_blur"]:
            w, h = x.size
            x = GaussianBlur(kernel_size=int((0.1 * min(w, h) // 2 * 2) + 1))(x)
        return x

    def __len__(self):
        return len(self.list_inputs)

    def __getitem__(self, ind):
        dict_data = dict()

        x, y = Image.open(self.list_inputs[ind]).convert("RGB"), Image.open(self.list_labels[ind])

        # if val or query dataset, do NOT do data augmentation
        if not any([self.val, self.query]):
            # get queries
            queries = Image.fromarray(self.queries[ind].astype(np.uint8) * 255) if self.queries is not None else None

            # data augmentation
            x, y, queries = self._geometric_augmentations(x, y, queries=queries)
            x = self._photometric_augmentations(x)

            dict_data.update({'queries': queries})

        x = TF.normalize(TF.to_tensor(x), self.mean, self.std)
        dict_data.update({'x': x, 'y': torch.tensor(np.asarray(y, np.int64), dtype=torch.long)})
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
