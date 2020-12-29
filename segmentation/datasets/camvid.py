import os
from glob import glob
from copy import copy
from math import ceil
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
    def __init__(self, args, val=False):
        super(CamVidDataset, self).__init__()
        self.args = args
        assert os.path.isdir(args.dir_dataset), f"{args.dir_dataset} does not exist."
        self.active_learning = args.active_learning
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
        self.dict_label_counts = None

        if args.n_pixels_per_img != 0 and not val:
            dir_labels = f"{args.dir_dataset}/n_pixels_labels"

            path_labels = f"{dir_labels}/arr_labels_{args.n_pixels_per_img}.npy"

            np.random.seed(self.seed)
            os.makedirs(f"{dir_labels}", exist_ok=True)

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

            print("# labelled pixels used for training:", arr_masks.astype(np.int32).sum())
            self.arr_masks = arr_masks

        self.val = val

    def _add_labels(self, dict_img_uncertain_pix):
        assert len(dict_img_uncertain_pix) == len(self.list_labels)
        previous = self.arr_masks.sum()
        for img_ind, list_uncertain_pix in dict_img_uncertain_pix.items():
            for uncertain_pix in list_uncertain_pix:
                self.arr_masks[img_ind[uncertain_pix]] = True
        print("# labelled pixels is changed from {} to {}".format(previous, self.arr_masks.sum()))

    def _geometric_augmentations(self, x, y, edge=None):
        if self.geometric_augmentations["random_scale"]:
            w, h = x.size
            rs = uniform(0.5, 2.0)
            w_resized, h_resized = int(w * rs), int(h * rs)

            x = TF.resize(x, (h_resized, w_resized), Image.BILINEAR)
            y = TF.resize(y, (h_resized, w_resized), Image.NEAREST)

            if edge is not None:
                edge = TF.resize(edge, (h_resized, w_resized), Image.NEAREST)

        if self.geometric_augmentations["crop"]:
            w, h = x.size
            pad_h = max(self.crop_size[0] - h, 0)
            pad_w = max(self.crop_size[1] - w, 0)
            self.pad_size = (pad_h, pad_w)

            x = TF.pad(x, (0, 0, pad_w, pad_h), fill=self.mean, padding_mode="constant")
            y = TF.pad(y, (0, 0, pad_w, pad_h), fill=11, padding_mode="constant")
            if edge is not None:
                edge = TF.pad(edge, (0, 0, pad_w, pad_h), fill=0, padding_mode="constant")

            w, h = x.size
            start_h = randint(0, h - self.crop_size[0])
            start_w = randint(0, w - self.crop_size[1])

            x = TF.crop(x, top=start_h, left=start_w, height=self.crop_size[0], width=self.crop_size[1])
            y = TF.crop(y, top=start_h, left=start_w, height=self.crop_size[0], width=self.crop_size[1])
            if edge is not None:
                edge = TF.crop(edge, top=start_h, left=start_w, height=self.crop_size[0], width=self.crop_size[1])

        if self.geometric_augmentations["random_hflip"]:
            if random() > 0.5:
                x, y = TF.hflip(x), TF.hflip(y)

                if edge is not None:
                    edge = TF.hflip(edge)

        if edge is not None:
            edge = torch.from_numpy(np.asarray(edge, dtype=np.uint8) // 255)
        else:
            edge = torch.tensor(0)

        return x, y, edge

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

    def __len__(self):
        return len(self.list_inputs)

    def __getitem__(self, ind):
        dict_data = dict()

        x, y = Image.open(self.list_inputs[ind]).convert("RGB"), Image.open(self.list_labels[ind])

        if not self.val:
            if self.arr_masks is not None:
                mask = Image.fromarray(self.arr_masks[ind].astype(np.uint8) * 255)
            else:
                mask = None

            x, y, mask = self._geometric_augmentations(x, y, mask)
            x = self._photometric_augmentations(x)

            if self.geometric_augmentations["random_scale"]:
                dict_data.update({"pad_size": self.pad_size})

            dict_data.update({'mask': mask})

        dict_data.update({'x': TF.to_tensor(x), 'y': torch.tensor(np.asarray(y, np.int64), dtype=torch.long)})
        if self.active_learning:
            dict_data.update({'ind': ind})
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
