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
from .base_dataset import BaseDataset


class CityscapesDataset(BaseDataset):
    def __init__(
            self,
            args,
            val: bool = False,
            query: bool = False
    ):
        super(CityscapesDataset, self).__init__()
        self.args = args
        if args.downsample > 1 and not val:
            dir_dataset = f"{args.dir_dataset}_d{args.downsample}"
        else:
            dir_dataset = f"{args.dir_dataset}_d2"

        if not os.path.isdir(dir_dataset):
            print("Downsampling Cityscapes images...")
            _make_downsampled_cityscapes(f"{args.dir_dataset}", downsample=args.downsample, val=False)
            _make_downsampled_cityscapes(f"{args.dir_dataset}", downsample=args.downsample, val=True)

        self.dataset_name = "cityscapes"
        self.dir_checkpoints = f"{args.dir_root}/checkpoints/{args.experim_name}"
        self.seed = args.seed

        mode = "val" if val else "train"
        self.list_inputs = sorted(glob(f"{dir_dataset}/leftImg8bit/{mode}/**/*.png"))
        self.list_labels = sorted(glob(f"{dir_dataset}/gtFine/{mode}/**/*_labelIds.png"))

        assert len(self.list_inputs) == len(self.list_labels) and len(self.list_inputs) > 0

        self.geometric_augmentations = args.augmentations["geometric"]
        self.photometric_augmentations = args.augmentations["photometric"]
        self.mean, self.std = args.mean, args.std
        self.n_classes = 19

        if self.geometric_augmentations["crop"]:
            self.mean_val = tuple((np.array(args.mean) * 255.0).astype(np.uint8).tolist())
            self.ignore_index = args.ignore_index

        if args.downsample == 2:
            self.crop_size = (512, 1024)
        elif args.downsample == 4:
            self.crop_size = (256, 512)
        else:
            raise ValueError(f"Invalid downsample {args.downsample}")
        self.pad_size = (0, 0)

        self.queries, self.n_pixels_total = None, -1

        n_pixels_per_img = args.n_pixels_by_us

        path_queries = f"{dir_dataset}/init_labelled_pixels_d{args.downsample}_{self.seed}.npy"
        if args.n_pixels_by_us != 0 and not val:
            if os.path.isfile(path_queries):
                self.queries = np.load(path_queries)
            else:
                np.random.seed(self.seed)
                label = Image.open(self.list_labels[0])

                list_queries = list()

                w, h = label.size
                n_pixels_per_img = h * w if n_pixels_per_img == 0 else n_pixels_per_img

                for i in tqdm(range(len(self.list_labels))):
                    label = np.array(Image.open(self.list_labels[i]))  # H x W
                    ind_non_void_pixels = np.where(label.flatten() != self.ignore_index)[0]
                    ind_chosen_pixels = np.random.choice(ind_non_void_pixels, n_pixels_per_img, replace=False)

                    queries_flat = np.zeros((h * w), dtype=np.bool)
                    queries_flat[ind_chosen_pixels] += True
                    queries = queries_flat.reshape((h, w))
                    list_queries.append(queries)

                self.queries = np.array(list_queries, dtype=np.bool)

                # save initial labelled pixels for a future reproduction
                np.save(path_queries, self.queries)
            self.n_pixels_total = self.queries.sum()

            os.makedirs(f"{self.dir_checkpoints}/0_query", exist_ok=True)
            np.save(f"{self.dir_checkpoints}/0_query/label.npy", self.queries)
            print("# labelled pixels used for training:", self.n_pixels_total)

        self.query = query
        self.val = val

    def __len__(self):
        return len(self.list_inputs)


def _make_downsampled_cityscapes(
        dir_cityscapes: str,
        downsample: int = 4,
        val=False
):
    h, w = 1024, 2048
    h_downsample, w_downsample = h // downsample, w // downsample
    mode = "val" if val else "train"

    list_inputs = sorted(glob(f"{dir_cityscapes}/leftImg8bit/{mode}/**/*.png"))
    list_labels = sorted(glob(f"{dir_cityscapes}/gtFine/{mode}/**/*_labelIds.png"))

    for x, y in tqdm(zip(list_inputs, list_labels)):
        dst_x = os.path.dirname(x).replace("cityscapes", f"cityscapes_d{downsample}")
        dst_y = os.path.dirname(y).replace("cityscapes", f"cityscapes_d{downsample}")
        os.makedirs(dst_x, exist_ok=True)
        os.makedirs(dst_y, exist_ok=True)

        name_x, name_y = x.split('/')[-1], y.split('/')[-1]
        x = Image.open(x).resize((w_downsample, h_downsample), resample=Image.BILINEAR)
        y = Image.open(y).resize((w_downsample, h_downsample), resample=Image.NEAREST)

        # reduce_cityscapes_labels
        y = np.array(y)
        y = _cityscapes_classes_to_labels(y)

        x.save(f"{dst_x}/{name_x}")
        Image.fromarray(y).save(f"{dst_y}/{name_y}")
    return


def _cityscapes_classes_to_labels(label_arr):
    ignore_class_label = 19
    classes_to_labels = {
        0: ignore_class_label,
        1: ignore_class_label,
        2: ignore_class_label,
        3: ignore_class_label,
        4: ignore_class_label,
        5: ignore_class_label,
        6: ignore_class_label,
        7: 0,
        8: 1,
        9: ignore_class_label,
        10: ignore_class_label,
        11: 2,
        12: 3,
        13: 4,
        14: ignore_class_label,
        15: ignore_class_label,
        16: ignore_class_label,
        17: 5,
        18: ignore_class_label,
        19: 6,
        20: 7,
        21: 8,
        22: 9,
        23: 10,
        24: 11,
        25: 12,
        26: 13,
        27: 14,
        28: 15,
        29: ignore_class_label,
        30: ignore_class_label,
        31: 16,
        32: 17,
        33: 18,
        -1: ignore_class_label
    }
    h, w = label_arr.shape
    label_arr = label_arr.flatten()
    for i in range(len(label_arr)):
        label_arr[i] = classes_to_labels[label_arr[i].item()]
    return label_arr.reshape((h, w))
