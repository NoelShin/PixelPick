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

from utils.utils import get_dataloader
from utils.ced import CED


class CamVidDataset(Dataset):
    def __init__(self, args, val=False, query=False):
        super(CamVidDataset, self).__init__()
        self.args = args
        assert os.path.isdir(args.dir_dataset), f"{args.dir_dataset} does not exist."
        self.dir_checkpoints = f"{args.dir_root}/checkpoints/{args.experim_name}"
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu:0")
        self.downsample = args.downsample
        self.seed = args.seed

        mode = "test" if val else "train"
        self.list_inputs = sorted(glob(f"{args.dir_dataset}/{mode}/*.png"))
        self.list_labels = sorted(glob(f"{args.dir_dataset}/{mode}annot/*.png"))

        assert len(self.list_inputs) == len(self.list_labels) and len(self.list_inputs) > 0

        self.geometric_augmentations = args.augmentations["geometric"]
        self.photometric_augmentations = args.augmentations["photometric"]
        self.mean = args.mean
        self.std = args.std

        if self.geometric_augmentations["random_scale"]:
            self.batch_size = args.batch_size
            self.n_samples = 0

        if self.geometric_augmentations["crop"]:
            self.mean_val = tuple((np.array(args.mean) * 255.0).astype(np.uint8).tolist())
            self.ignore_index = args.ignore_index

        self.crop_size = (360 // self.downsample, 480 // self.downsample)
        self.pad_size = (0, 0)

        self.arr_masks, self.n_pixels_total = None, -1

        n_pixels_per_img = args.n_pixels_by_us

        path_arr_masks = f"{args.dir_dataset}/{mode}/init_labelled_pixels_d{self.downsample}_{self.seed}.npy"
        if (args.n_pixels_by_us + args.n_pixels_by_oracle_cb) != 0 and not val:
            if os.path.isfile(path_arr_masks) and False:
                self.arr_masks = np.load(path_arr_masks)
            else:
                np.random.seed(self.seed)
                label = Image.open(self.list_labels[0])

                list_masks = list()

                w, h = label.size
                w, h = w // self.downsample, h // self.downsample
                n_pixels_per_img = h * w if n_pixels_per_img == 0 else n_pixels_per_img

                for i in tqdm(range(len(self.list_labels))):
                    label = Image.open(self.list_labels[i]).resize((w, h), Image.NEAREST)  # H x W
                    label = np.array(label)  # H x W
                    ind_non_void_pixels = np.where(label.flatten() != self.ignore_index)[0]
                    ind_chosen_pixels = np.random.choice(ind_non_void_pixels, n_pixels_per_img, replace=False)

                    mask = np.zeros((h, w), dtype=np.bool)
                    mask_flat = mask.flatten()
                    mask_flat[ind_chosen_pixels] += True
                    mask = mask_flat.reshape((h, w))
                    list_masks.append(mask)

                self.arr_masks = np.array(list_masks, dtype=np.bool)

                # save initial labelled pixels for a future reproduction
                np.save(path_arr_masks, self.arr_masks)

            os.makedirs(f"{self.dir_checkpoints}/0_query", exist_ok=True)
            np.save(f"{self.dir_checkpoints}/0_query/label.npy", self.arr_masks)
            self.n_pixels_total = self.arr_masks.sum()
            print("# labelled pixels used for training:", self.n_pixels_total)

        self.use_ced = args.use_ced
        self.use_img_inp = args.use_img_inp

        self.use_visual_acuity = args.use_visual_acuity

        # pseudo-label
        self.use_pseudo_label = args.use_pseudo_label
        self.pseudo_label_flag = False
        if self.use_pseudo_label:
            from pseudo_labelling.local_sim import LocalSimilarity
            self.local_similarity = LocalSimilarity(args)
            self.pseudo_labels = None

        self.val = val
        self.query = query

    def label_queries(self, queries, nth_query=None):
        assert len(queries) == len(self.arr_masks), f"{queries.shape}, {self.arr_masks.shape}"
        previous = self.arr_masks.sum()

        self.arr_masks = np.maximum(self.arr_masks, queries)

        if isinstance(nth_query, int):
            try:
                np.save(f"{self.dir_checkpoints}/{nth_query}_query/label.npy", self.arr_masks)
            except FileNotFoundError:
                os.makedirs(f"{self.dir_checkpoints}/{nth_query}_query", exist_ok=True)
                np.save(f"{self.dir_checkpoints}/{nth_query}_query/label.npy", self.arr_masks)
        new = self.arr_masks.sum()
        self.n_pixels_total = new
        print("# labelled pixels is changed from {} to {} (delta: {})".format(previous, new, new - previous))

    def _geometric_augmentations(self, x, y, edge=None, ced=None):
        if self.geometric_augmentations["random_scale"]:
            w, h = x.size
            rs = uniform(0.5, 2.0)
            w_resized, h_resized = int(w * rs), int(h * rs)

            x = TF.resize(x, (h_resized, w_resized), Image.BILINEAR)
            y = TF.resize(y, (h_resized, w_resized), Image.NEAREST)

            if edge is not None:
                edge = TF.resize(edge, (h_resized, w_resized), Image.NEAREST)

            if ced is not None:
                ced = TF.resize(ced, (h_resized, w_resized), Image.NEAREST)

        if self.geometric_augmentations["crop"]:
            w, h = x.size
            pad_h = max(self.crop_size[0] - h, 0)
            pad_w = max(self.crop_size[1] - w, 0)
            self.pad_size = (pad_h, pad_w)

            x = TF.pad(x, (0, 0, pad_w, pad_h), fill=self.mean_val, padding_mode="constant")
            y = TF.pad(y, (0, 0, pad_w, pad_h), fill=11, padding_mode="constant")

            if edge is not None:
                edge = TF.pad(edge, (0, 0, pad_w, pad_h), fill=0, padding_mode="constant")

            if ced is not None:
                ced = TF.pad(ced, (0, 0, pad_w, pad_h), fill=0, padding_mode="constant")

            w, h = x.size
            start_h = randint(0, h - self.crop_size[0])
            start_w = randint(0, w - self.crop_size[1])

            x = TF.crop(x, top=start_h, left=start_w, height=self.crop_size[0], width=self.crop_size[1])
            y = TF.crop(y, top=start_h, left=start_w, height=self.crop_size[0], width=self.crop_size[1])
            if edge is not None:
                edge = TF.crop(edge, top=start_h, left=start_w, height=self.crop_size[0], width=self.crop_size[1])
            if ced is not None:
                ced = TF.crop(ced, top=start_h, left=start_w, height=self.crop_size[0], width=self.crop_size[1])

        if self.geometric_augmentations["random_hflip"]:
            if random() > 0.5:
                x, y = TF.hflip(x), TF.hflip(y)

                if edge is not None:
                    edge = TF.hflip(edge)

                if ced is not None:
                    ced = TF.hflip(ced)

        if edge is not None:
            edge = torch.from_numpy(np.asarray(edge, dtype=np.uint8) // 255)
        else:
            edge = torch.tensor(0)

        if ced is None:
            ced = torch.tensor(0)

        return x, y, edge, ced

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

    def _gaussian_blur(self, img, min_val=0.1, max_val=2.0):
        w, h = img.size
        smaller_length = min(w, h)
        kernel_size = int((0.1 * smaller_length // 2 * 2) + 1)

        np_array = np.array(img)
        sigma = (max_val - min_val) * np.random.random_sample() + min_val

        np_array = cv2.GaussianBlur(np_array, (kernel_size, kernel_size), sigma)
        return Image.fromarray(np_array)

    @staticmethod
    def _ced_mask(img):
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

    def update_pseudo_label(self, model, window_size, nth_query):
        print("Updating pseudo-labels...")
        if not self.pseudo_label_flag:
            self.pseudo_label_flag = True

        masks = torch.tensor(self.arr_masks).to(self.device)
        dataloader = get_dataloader(self.args,
                                    val=False,
                                    query=True,
                                    shuffle=False,
                                    batch_size=1,
                                    n_workers=self.args.n_workers)

        list_pseudo_labels = list()

        model.eval()
        with torch.no_grad():
            for batch_ind, dict_data in tqdm(enumerate(dataloader)):
                x, y = dict_data['x'].to(self.device), dict_data['y'].to(self.device)
                mask = masks[batch_ind].unsqueeze(dim=0)
                # try:
                #     mask = masks[batch_ind * self.args.batch_size: (batch_ind + 1) * self.args.batch_size]
                # except IndexError:
                #     mask = masks[batch_ind * self.args.batch_size:]

                dict_output = model(x)
                emb = dict_output["emb"]
                list_pseudo_labels.extend(self.local_similarity(emb, y, mask, window_size))

        self.pseudo_labels = torch.stack(list_pseudo_labels, dim=0).cpu().numpy().astype(np.uint8)
        self.local_similarity.print()
        self.local_similarity.write(fp=f"{self.dir_checkpoints}/{nth_query}_query/pr.txt")
        self.local_similarity.reset_metrics()

    def __getitem__(self, ind):
        dict_data = dict()

        x, y = Image.open(self.list_inputs[ind]).convert("RGB"), Image.open(self.list_labels[ind])

        # if val or query dataset, do NOT do augmentation
        if self.val or self.query:
            x = TF.to_tensor(x)
            x = TF.normalize(x, self.mean, self.std)

        else:
            if self.arr_masks is not None:
                mask = Image.fromarray(self.arr_masks[ind].astype(np.uint8) * 255)
            else:
                mask = None

            x, y, mask, ced = self._geometric_augmentations(x, y, edge=mask)

            x = self._photometric_augmentations(x)

            # if self.geometric_augmentations["random_scale"]:
            #     dict_data.update({"pad_size": self.pad_size})
            dict_data.update({'mask': mask})

            x = TF.to_tensor(x)
            x = TF.normalize(x, self.mean, self.std)

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
