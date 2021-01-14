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

from utils.ced import CED


class CamVidDataset(Dataset):
    def __init__(self, args, val=False, query=False):
        super(CamVidDataset, self).__init__()
        self.args = args
        assert os.path.isdir(args.dir_dataset), f"{args.dir_dataset} does not exist."
        self.seed = args.seed

        mode = "test" if val else "train"
        self.list_inputs = sorted(glob(f"{args.dir_dataset}/{mode}/*.png"))
        self.list_labels = sorted(glob(f"{args.dir_dataset}/{mode}annot/*.png"))

        assert len(self.list_inputs) == len(self.list_labels) and len(self.list_inputs) > 0

        self.geometric_augmentations = args.augmentations["geometric"]
        self.photometric_augmentations = args.augmentations["photometric"]

        if self.geometric_augmentations["random_scale"]:
            self.batch_size = args.batch_size
            self.n_samples = 0

        if self.geometric_augmentations["crop"]:
            self.mean = tuple((np.array(args.mean) * 255.0).astype(np.uint8).tolist())
            self.ignore_index = args.ignore_index

        self.crop_size = (360, 480)
        self.pad_size = (0, 0)

        self.arr_masks = None

        path_arr_masks = f"{args.dir_dataset}/{mode}/init_labelled_pixels_{self.seed}.npy"
        if args.n_pixels_per_img != 0 and not val:
            if os.path.isfile(path_arr_masks):
                self.arr_masks = np.load(path_arr_masks)

            else:
                np.random.seed(self.seed)
                label = Image.open(self.list_labels[0]).convert("RGB")

                list_masks = list()

                w, h = label.size
                n_pixels_per_img = h * w if args.n_pixels_per_img == 0 else args.n_pixels_per_img

                for _ in tqdm(range(len(self.list_labels))):
                    mask = np.zeros((h, w), dtype=np.bool)
                    mask_flat = mask.flatten()

                    # pseudo-random numbers
                    ind_chosen_pixels = np.random.choice(range(len(mask_flat)), n_pixels_per_img, replace=False)

                    mask_flat[ind_chosen_pixels] += True

                    mask = mask_flat.reshape((h, w))
                    list_masks.append(mask)

                arr_masks = np.array(list_masks, dtype=np.bool)

                # save initial labelled pixels for a future reproduction
                np.save(path_arr_masks, arr_masks)

                print("# labelled pixels used for training:", arr_masks.astype(np.int32).sum())
                self.arr_masks = arr_masks

        self.use_ced = args.use_ced
        self.use_img_inp = args.use_img_inp

        self.use_visual_acuity = args.use_visual_acuity

        self.val = val
        self.query = query

    def label_queries(self, queries):
        assert len(queries) == len(self.arr_masks), f"{queries.shape}, {self.arr_masks.shape}"
        previous = self.arr_masks.sum()
        self.arr_masks = np.maximum(self.arr_masks, queries)
        print("# labelled pixels is changed from {} to {}".format(previous, self.arr_masks.sum()))

    def _geometric_augmentations(self, x, y, edge=None, merged_mask=None, x_blurred=None):
        if self.geometric_augmentations["random_scale"]:
            w, h = x.size
            rs = uniform(0.5, 2.0)
            w_resized, h_resized = int(w * rs), int(h * rs)

            x = TF.resize(x, (h_resized, w_resized), Image.BILINEAR)
            y = TF.resize(y, (h_resized, w_resized), Image.NEAREST)

            if edge is not None:
                edge = TF.resize(edge, (h_resized, w_resized), Image.NEAREST)

            if merged_mask is not None:
                merged_mask = TF.resize(merged_mask, (h_resized, w_resized), Image.NEAREST)

            if x_blurred is not None:
                x_blurred = TF.resize(x_blurred, (h_resized, w_resized), Image.BILINEAR)

        if self.geometric_augmentations["crop"]:
            w, h = x.size
            pad_h = max(self.crop_size[0] - h, 0)
            pad_w = max(self.crop_size[1] - w, 0)
            self.pad_size = (pad_h, pad_w)

            x = TF.pad(x, (0, 0, pad_w, pad_h), fill=self.mean, padding_mode="constant")
            y = TF.pad(y, (0, 0, pad_w, pad_h), fill=11, padding_mode="constant")
            if edge is not None:
                edge = TF.pad(edge, (0, 0, pad_w, pad_h), fill=0, padding_mode="constant")

            if merged_mask is not None:
                merged_mask = TF.pad(merged_mask, (0, 0, pad_w, pad_h), fill=0, padding_mode="constant")

            if x_blurred is not None:
                x_blurred = TF.pad(x_blurred, (0, 0, pad_w, pad_h), fill=0, padding_mode="constant")

            w, h = x.size
            start_h = randint(0, h - self.crop_size[0])
            start_w = randint(0, w - self.crop_size[1])

            x = TF.crop(x, top=start_h, left=start_w, height=self.crop_size[0], width=self.crop_size[1])
            y = TF.crop(y, top=start_h, left=start_w, height=self.crop_size[0], width=self.crop_size[1])
            if edge is not None:
                edge = TF.crop(edge, top=start_h, left=start_w, height=self.crop_size[0], width=self.crop_size[1])

            if merged_mask is not None:
                merged_mask = TF.crop(merged_mask, top=start_h, left=start_w, height=self.crop_size[0], width=self.crop_size[1])

            if x_blurred is not None:
                x_blurred = TF.crop(x_blurred, top=start_h, left=start_w, height=self.crop_size[0], width=self.crop_size[1])

        if self.geometric_augmentations["random_hflip"]:
            if random() > 0.5:
                x, y = TF.hflip(x), TF.hflip(y)

                if edge is not None:
                    edge = TF.hflip(edge)

                if merged_mask is not None:
                    merged_mask = TF.hflip(merged_mask)

                if x_blurred is not None:
                    x_blurred = TF.hflip(x_blurred)

        if edge is not None:
            edge = torch.from_numpy(np.asarray(edge, dtype=np.uint8) // 255)
        else:
            edge = torch.tensor(0)

        if merged_mask is not None:
            merged_mask = torch.from_numpy(np.asarray(merged_mask, dtype=np.uint8) // 255)
        else:
            merged_mask = torch.tensor(0)

        return x, y, edge, merged_mask, x_blurred

    def _photometric_augmentations(self, x, x_blurred=None):
        if self.photometric_augmentations["random_color_jitter"]:
            color_jitter = ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2)
            x = RandomApply([color_jitter], p=0.8)(x)

            if x_blurred is not None:
                x_blurred = RandomApply([color_jitter], p=0.8)(x_blurred)

        if self.photometric_augmentations["random_grayscale"]:
            x = RandomGrayscale(0.2)(x)

            if x_blurred is not None:
                x_blurred = RandomGrayscale(0.2)(x_blurred)

        if self.photometric_augmentations["random_gaussian_blur"]:
            w, h = x.size
            smaller_length = min(w, h)
            x = GaussianBlur(kernel_size=int((0.1 * smaller_length // 2 * 2) + 1))(x)
        return x

    def _gaussian_blur(self, img, min_val=0.1, max_val=2.0):
        w, h = img.size
        smaller_length = min(w, h)
        kernel_size = int((0.1 * smaller_length // 2 * 2) + 1)

        np_array = np.array(img)
        sigma = (max_val - min_val) * np.random.random_sample() + min_val

        np_array = cv2.GaussianBlur(np_array, (kernel_size, kernel_size), sigma)
        return Image.fromarray(np_array)

    def _ced_mask(self, img):
        from random import randint
        lower_thres = randint(0, 100)
        dilation = 1  # randint(1, 2)
        ced = CED(dilation, lower_thres, lower_thres * 2)(img)
        return ced

    def _sample_region_masks(self, h_img, w_img, divider, n_patches):
        from random import choice, choices
        size_region_mask = (h_img * w_img) // (divider * n_patches)

        list_factors = self._get_factors(size_region_mask)
        list_factors = [f for f in list_factors if f < h_img and size_region_mask // f < w_img]

        list_h = choices(list_factors, k=n_patches)
        list_w = list(size_region_mask // h for h in list_h)

        merged_mask = np.zeros((h_img, w_img), np.uint8)
        list_region_masks_params = list()
        for h, w in zip(list_h, list_w):
            i, j = choice(range(h_img - h)), choice(range(w_img - w))
            merged_mask[i: i + h, j: j + w] = 1
            list_region_masks_params.append((i, j, h, w))

        merged_mask = Image.fromarray(merged_mask * 255)
        return merged_mask, list_region_masks_params

    @staticmethod
    def _get_factors(n):
        list_factors = [1]
        from math import ceil
        for t in range(2, (ceil((n / 2) + 1))):
            if n % t == 0:
                list_factors.append(t)
        list_factors.append(n)
        return list_factors

    def __len__(self):
        return len(self.list_inputs)

    def __getitem__(self, ind):
        dict_data = dict()

        x, y = Image.open(self.list_inputs[ind]).convert("RGB"), Image.open(self.list_labels[ind])

        # if not val nor query dataset, do augmentation
        if not self.val and not self.query:
            if self.arr_masks is not None:
                mask = Image.fromarray(self.arr_masks[ind].astype(np.uint8) * 255)
            else:
                mask = None

            if self.use_visual_acuity:
                x_clean = x
            else:
                x_clean = None

            if self.use_img_inp:
                if self.use_ced:
                    merged_mask = self._ced_mask(x)
                else:
                    w, h = x.size
                    merged_mask, list_region_masks_params = self._sample_region_masks(h, w, divider=8, n_patches=8)

            else:
                merged_mask = None

            x, y, mask, merged_mask, x_clean = self._geometric_augmentations(x, y, mask, merged_mask, x_clean)
            x = self._photometric_augmentations(x)

            if self.geometric_augmentations["random_scale"]:
                dict_data.update({"pad_size": self.pad_size})
            dict_data.update({'mask': mask, 'merged_mask': merged_mask})

            if self.use_visual_acuity:
                dict_data.update({"x_clean": TF.to_tensor(x_clean)})

        dict_data.update({'x': TF.to_tensor(x), 'y': torch.tensor(np.asarray(y, np.int64), dtype=torch.long)})
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
