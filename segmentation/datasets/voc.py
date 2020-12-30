import os
from glob import glob
from random import choice, random, randint, uniform

import numpy as np
import cv2
import torch
from torchvision.datasets import VOCSegmentation
import torchvision.transforms as transforms
from torchvision.transforms import ColorJitter, RandomApply, RandomGrayscale, CenterCrop, Normalize
import torchvision.transforms.functional as tf
from PIL import Image, ImageFile
from tqdm import tqdm
ImageFile.LOAD_TRUNCATED_IMAGES = True


class VOC2012Segmentation:
    def __init__(self, args, val=False):
        super(VOC2012Segmentation, self).__init__()
        self.ignore_bg = args.ignore_bg
        self.ignore_index = args.ignore_index
        self.size_base = args.size_base
        self.size_crop = (args.size_crop, args.size_crop)
        self.stride_total = args.stride_total
        self.use_softmax = args.use_softmax

        if args.use_augmented_dataset and not val:
            self.voc = AugmentedVOC(args.dir_augmented_dataset)

        else:
            self.voc = VOCSegmentation(f"{args.dir_dataset}", image_set='val' if val else 'train', download=False)
        print("len data:", len(self.voc))

        self.geometric_augmentations = args.augmentations["geometric"]
        self.photometric_augmentations = args.augmentations["photometric"]
        self.normalize = Normalize(mean=args.mean, std=args.std)

        if self.geometric_augmentations["random_scale"]:
            self.batch_size = args.batch_size
            self.n_samples = 0

        if self.geometric_augmentations["crop"]:
            self.mean = tuple((np.array(args.mean) * 255.0).astype(np.uint8).tolist())

        # generate masks
        self.list_masks = None
        if args.n_pixels_per_img != 0 and not val:
            np.random.seed(args.seed)  # for future reproduction

            list_masks = list()
            n_pixels_total = 0
            for i in tqdm(range(len(self.voc))):
                label = self.voc[i][1]
                w, h = label.size
                n_pixels_per_img = h * w if args.n_pixels_per_img == 0 else args.n_pixels_per_img

                # generate masks whose size is set to base_size (longer side)
                if w > h:
                    h = int(float(h) / w * self.size_base)
                    w = self.size_base
                else:
                    w = int(float(w) / h * self.size_base)
                    h = self.size_base

                mask = np.zeros((h, w), dtype=np.bool)
                mask_flat = mask.flatten()

                # filter void pixels - boundary pixels that the original labels have (fyi, 5 pixels thickness)
                label = label.resize((w, h), Image.NEAREST)  # note that downsampling method should be Image.NEAREST
                label = np.asarray(label, dtype=np.int32)

                label_flatten = label.flatten()
                ind_void_pixels = np.where(label_flatten == 255)[0]

                ind_non_void_pixels = np.setdiff1d(range(len(mask_flat)), ind_void_pixels)  # remove void pixels
                assert len(ind_non_void_pixels) <= len(mask_flat)

                # a very rare case where the number of ind_non_void_pixels is not large enough to sample from
                if len(ind_non_void_pixels) < args.n_pixels_per_img:
                    n_pixels_per_img = len(ind_non_void_pixels)

                # if using the softmax, we consider the background class (label: 0) as known unknown class and include
                # it as one type of class. However, when using gcpl (open set), we consider the background class as void
                # class, i.e. unknown unknown class and do not sample pixels belonging to that class.
                if args.use_softmax or not args.ignore_bg:
                    ind_chosen_pixels = np.random.choice(ind_non_void_pixels, n_pixels_per_img, replace=False)

                else:
                    ind_bg_pixels = np.where(label_flatten == 0)[0]
                    ind_non_void_bg_pixels = np.setdiff1d(ind_non_void_pixels, ind_bg_pixels)

                    # a very rare case where the number of ind_non_void_bg_pixels is not large enough to sample from
                    if len(ind_non_void_bg_pixels) < args.n_pixels_per_img:
                        n_pixels_per_img = len(ind_non_void_bg_pixels)

                    ind_chosen_pixels = np.random.choice(ind_non_void_bg_pixels, n_pixels_per_img, replace=False)

                mask_flat[ind_chosen_pixels] += True
                mask = mask_flat.reshape((h, w))

                list_masks.append(mask)
                n_pixels_total += mask.sum()

            self.list_masks = list_masks
            print("# labelled pixels used for training:", n_pixels_total)

        self.val = val

    def _geometric_augmentations(self, x, y, mask=None):
        if self.geometric_augmentations["random_scale"]:
            w, h = x.size
            rs = uniform(0.5, 2.0)
            w_resized, h_resized = int(w * rs), int(h * rs)

            x = tf.resize(x, (h_resized, w_resized), Image.BILINEAR)
            y = tf.resize(y, (h_resized, w_resized), Image.NEAREST)

            if mask is not None:
                mask = Image.fromarray(mask).resize((w_resized, h_resized), Image.NEAREST)

        if self.geometric_augmentations["crop"]:
            w, h = x.size
            pad_h = max(self.size_crop[0] - h, 0)
            pad_w = max(self.size_crop[1] - w, 0)
            self.pad_size = (pad_h, pad_w)

            x = tf.pad(x, (0, 0, pad_w, pad_h), fill=self.mean, padding_mode="constant")
            y = tf.pad(y, (0, 0, pad_w, pad_h), fill=self.ignore_index, padding_mode="constant")

            w, h = x.size
            start_h = randint(0, h - self.size_crop[0])
            start_w = randint(0, w - self.size_crop[1])

            x = tf.crop(x, top=start_h, left=start_w, height=self.size_crop[0], width=self.size_crop[1])
            y = tf.crop(y, top=start_h, left=start_w, height=self.size_crop[0], width=self.size_crop[1])

            if mask is not None:
                mask = tf.crop(mask, top=start_h, left=start_w, height=self.size_crop[0], width=self.size_crop[1])

        if self.geometric_augmentations["random_hflip"]:
            if random() > 0.5:
                x, y = tf.hflip(x), tf.hflip(y)
                if mask is not None:
                    mask = tf.hflip(mask)
        return x, y, mask

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

        x, y = self.voc[ind]
        mask = self.list_masks[ind] if self.list_masks is not None else None

        if self.val:
            dict_data.update({"x": self.normalize(tf.to_tensor(x)), "y": self._image_to_tensor(y)})

        else:
            x, y, mask = self._geometric_augmentations(x, y, mask)
            dict_data.update({"x": self.normalize(tf.to_tensor(self._photometric_augmentations(x))),
                              "mask": torch.tensor(np.array(mask, np.bool), dtype=torch.bool)})

            # convert background class value into ingore_index for open set regime
            if not self.use_softmax and self.ignore_bg:
                w, h = y.size
                y_flatten = np.array(y, np.int32).flatten()

                # reduce non-void pixel values by one. This is because we do not count bg class as a class that the open
                # set model should predict along with non-bg classes. By doing so, pixels belonging to the bg class will
                # have a value of -1. Note that we should leave void pixels out of this process.
                y_flatten[(y_flatten != self.ignore_index)] -= 1
                y_flatten[(y_flatten == -1)] = self.ignore_index

                y = y_flatten.reshape((h, w))

                dict_data.update({"y": torch.tensor(y, dtype=torch.long)})

            else:
                dict_data.update({"y": self._image_to_tensor(y)})
            # w, h = x.size
            # y = y.resize((w // self.stride_total, h // self.stride_total), resample=Image.NEAREST)

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