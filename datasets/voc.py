import os
from glob import glob
from random import random, randint, uniform
import pickle as pkl

import numpy as np
import cv2
from PIL import Image, ImageFile
import torch
from torchvision.datasets import VOCSegmentation
from torchvision.transforms import ColorJitter, RandomApply, RandomGrayscale, CenterCrop, Normalize
import torchvision.transforms.functional as tf
from tqdm import tqdm
ImageFile.LOAD_TRUNCATED_IMAGES = True


class VOC2012Segmentation:
    def __init__(self, args, val=False, query=False):
        super(VOC2012Segmentation, self).__init__()
        self.dir_checkpoints = f"{args.dir_root}/checkpoints/{args.experim_name}"
        self.ignore_index = args.ignore_index
        self.size_base = args.size_base
        self.size_crop = (args.size_crop, args.size_crop)
        self.stride_total = args.stride_total

        if args.use_augmented_dataset and not val:
            self.voc = AugmentedVOC(args.dir_augmented_dataset)
        else:
            self.voc = VOCSegmentation(f"{args.dir_dataset}", image_set='val' if val else 'train', download=True)
        print("# images:", len(self.voc))

        self.geometric_augmentations = args.augmentations["geometric"]
        self.photometric_augmentations = args.augmentations["photometric"]
        self.normalize = Normalize(mean=args.mean, std=args.std)
        if query:
            self.geometric_augmentations["random_scale"] = False
            self.geometric_augmentations["crop"] = False
            self.geometric_augmentations["random_hflip"] = False

        if self.geometric_augmentations["crop"]:
            self.mean = tuple((np.array(args.mean) * 255.0).astype(np.uint8).tolist())

        # generate initial queries
        n_pixels_per_img = args.n_pixels_by_us
        init_n_pixels = args.n_init_pixels if args.n_init_pixels > 0 else n_pixels_per_img

        self.queries, self.n_pixels_total = None, -1
        path_queries = f"{args.dir_dataset}/init_labelled_pixels_{args.seed}.pkl"
        if n_pixels_per_img != 0 and not val:
            os.makedirs(f"{self.dir_checkpoints}/0_query", exist_ok=True)
            n_pixels_total = 0

            list_queries = list()
            for i in tqdm(range(len(self.voc))):
                label = self.voc[i][1]
                w, h = label.size

                if n_pixels_per_img == 0:
                    n_pixels_per_img = h * w
                elif n_pixels_per_img != 0 and init_n_pixels > 0:
                    n_pixels_per_img = init_n_pixels
                else:
                    raise NotImplementedError

                # generate queries whose size is set to base_size (longer side), i.e. 400 as default
                h, w = self._compute_base_size(h, w)

                queries_flat = np.zeros((h * w), dtype=np.bool)

                # filter void pixels - boundary pixels that the original labels have (fyi, 5 pixels thickness)
                label = label.resize((w, h), Image.NEAREST)  # note that downsampling method should be Image.NEAREST
                label = np.asarray(label, dtype=np.int32)

                label_flatten = label.flatten()
                ind_void_pixels = np.where(label_flatten == 255)[0]

                ind_non_void_pixels = np.setdiff1d(range(len(queries_flat)), ind_void_pixels)  # remove void pixels
                assert len(ind_non_void_pixels) <= len(queries_flat)

                # for a very rare case where the number of non_void_pixels is not large enough to sample from
                if len(ind_non_void_pixels) < n_pixels_per_img:
                    n_pixels_per_img = len(ind_non_void_pixels)

                ind_chosen_pixels = np.random.choice(ind_non_void_pixels, n_pixels_per_img, replace=False)

                queries_flat[ind_chosen_pixels] += True
                queries = queries_flat.reshape((h, w))

                list_queries.append(queries)
                n_pixels_total += queries.sum()
            pkl.dump(list_queries, open(f"{path_queries}", 'wb'))

            # Note that images of voc dataset vary from image to image thus can't use np.stack().
            self.queries = list_queries
            pkl.dump(self.queries, open(f"{self.dir_checkpoints}/0_query/label.pkl", 'wb'))

            self.n_pixels_total = n_pixels_total
            print("# labelled pixels used for training:", n_pixels_total)
        self.val, self.query = val, query

    def label_queries(self, queries, nth_query=None):
        assert len(queries) == len(self.queries), f"{queries.shape}, {len(self.queries)}"
        previous = self.n_pixels_total

        list_queries = list()
        n_pixels_total = 0
        for q, m in zip(queries, self.queries):
            new_m = np.logical_or(q, m)
            list_queries.append(new_m)
            n_pixels_total += new_m.sum()
        self.queries, self.n_pixels_total = list_queries, n_pixels_total

        if isinstance(nth_query, int):
            os.makedirs(f"{self.dir_checkpoints}/{nth_query}_query", exist_ok=True)
            pkl.dump(self.queries, open(f"{self.dir_checkpoints}/{nth_query}_query/label.pkl", 'wb'))

        print("# labelled pixels is changed from {} to {} (delta: {})".format(previous, n_pixels_total, n_pixels_total - previous))

    def _compute_base_size(self, h, w):
        if w > h:
            h = int(float(h) / w * self.size_base)
            w = self.size_base
        else:
            w = int(float(w) / h * self.size_base)
            h = self.size_base
        return h, w

    def _geometric_augmentations(self, x, y, queries=None):
        w, h = x.size
        h, w = self._compute_base_size(h, w)

        x, y = tf.resize(x, [h, w], Image.BILINEAR), tf.resize(y, [h, w], Image.NEAREST)

        if self.geometric_augmentations["random_scale"]:
            rs = uniform(0.5, 2.0)

            h, w = int(h * rs), int(w * rs)
            x, y = tf.resize(x, [h, w], Image.BILINEAR), tf.resize(y, [h, w], Image.NEAREST)

            if queries is not None:
                queries = Image.fromarray(queries).resize((w, h), Image.NEAREST)

        if self.geometric_augmentations["crop"]:
            w, h = x.size
            pad_h, pad_w = max(self.size_crop[0] - h, 0), max(self.size_crop[1] - w, 0)
            self.pad_size = (pad_h, pad_w)

            x = tf.pad(x, [0, 0, pad_w, pad_h], fill=self.mean, padding_mode="constant")
            y = tf.pad(y, [0, 0, pad_w, pad_h], fill=self.ignore_index, padding_mode="constant")

            w, h = x.size
            start_h, start_w = randint(0, h - self.size_crop[0]), randint(0, w - self.size_crop[1])

            x = tf.crop(x, top=start_h, left=start_w, height=self.size_crop[0], width=self.size_crop[1])
            y = tf.crop(y, top=start_h, left=start_w, height=self.size_crop[0], width=self.size_crop[1])

            if queries is not None:
                queries = tf.crop(queries, top=start_h, left=start_w, height=self.size_crop[0], width=self.size_crop[1])

        if self.geometric_augmentations["random_hflip"]:
            if random() > 0.5:
                x, y = tf.hflip(x), tf.hflip(y)
                if queries is not None:
                    queries = tf.hflip(queries)
        return x, y, queries

    def _photometric_augmentations(self, x):
        if self.photometric_augmentations["random_color_jitter"]:
            color_jitter = ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)
            x = RandomApply([color_jitter], p=0.8)(x)

        if self.photometric_augmentations["random_grayscale"]:
            x = RandomGrayscale(0.2)(x)

        if self.photometric_augmentations["random_gaussian_blur"]:
            w, h = x.size
            smaller_length = min(w, h)
            x = GaussianBlur(kernel_size=int((0.1 * smaller_length // 2 * 2) + 1))(x)
        return x

    def __len__(self):
        return len(self.voc)

    def __getitem__(self, ind):
        dict_data = dict()
        (x, y), queries = self.voc[ind], self.queries[ind] if self.queries is not None else None

        if self.val:
            dict_data.update({"x": self.normalize(tf.to_tensor(x)), "y": self._image_to_tensor(y)})

        else:
            x, y, queries = self._geometric_augmentations(x, y, queries)

            if not self.query:
                x = self._photometric_augmentations(x)

            dict_data.update({"x": self.normalize(tf.to_tensor(x)),
                              "y": self._image_to_tensor(y),
                              "queries": torch.tensor(np.array(queries, np.bool), dtype=torch.bool)})
        return dict_data

    @staticmethod
    def _image_to_tensor(pil_image):
        return torch.tensor(np.array(pil_image, dtype=np.uint8), dtype=torch.long)


class AugmentedVOC:
    def __init__(self, root):
        assert os.path.isdir(root)
        self.voc = list(zip(sorted(glob(f"{root}/images/*")), sorted(glob(f"{root}/annot/*"))))

    def __len__(self):
        return len(self.voc)

    def __getitem__(self, ind):
        p_img, p_annot = self.voc[ind]
        assert p_img.split('/')[-1].split('.')[0] == p_annot.split('/')[-1].split('.')[0]

        return Image.open(p_img), Image.open(p_annot)


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


if __name__ == '__main__':
    from args import Arguments
    args = Arguments().parse_args()
    VOC2012Segmentation(args)