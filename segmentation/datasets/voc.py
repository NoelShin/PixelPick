import os
from glob import glob
from random import choice, random, randint, uniform
import pickle as pkl

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
    def __init__(self, args, val=False, query=False):
        super(VOC2012Segmentation, self).__init__()
        self.dir_checkpoints = f"{args.dir_root}/checkpoints/{args.experim_name}"
        self.ignore_bg = args.ignore_bg
        self.ignore_index = args.ignore_index
        self.size_base = args.size_base
        self.size_crop = (args.size_crop, args.size_crop)
        self.stride_total = args.stride_total
        self.use_softmax = args.use_softmax
        self.use_scribbles = args.use_scribbles

        n_pixels_per_img = args.n_pixels_by_us

        if args.use_augmented_dataset and not val:
            self.voc = AugmentedVOC(args.dir_augmented_dataset)

        else:
            self.voc = VOCSegmentation(f"{args.dir_dataset}", image_set='val' if val else 'train', download=False)
        print("len data:", len(self.voc))

        self.geometric_augmentations = args.augmentations["geometric"]
        self.photometric_augmentations = args.augmentations["photometric"]
        self.normalize = Normalize(mean=args.mean, std=args.std)

        if query:
            self.geometric_augmentations["random_scale"] = False
            self.geometric_augmentations["crop"] = False
            self.geometric_augmentations["random_hflip"] = False

        if self.geometric_augmentations["random_scale"]:
            self.batch_size = args.batch_size
            self.n_samples = 0

        if self.geometric_augmentations["crop"]:
            self.mean = tuple((np.array(args.mean) * 255.0).astype(np.uint8).tolist())

        # generate masks
        self.arr_masks, self.n_pixels_total = None, -1

        if self.use_scribbles:
            list_scribbles = sorted(glob(f"{args.dir_dataset}/scribbles/*.npy"))
            scribbles = list()
            n_pixels_total = 0
            for s in list_scribbles:
                scribble = np.load(s).astype(np.bool)
                scribbles.append(scribble)
                n_pixels_total += scribble.sum()
            self.arr_masks = scribbles
            self.n_pixels_total = n_pixels_total

        else:
            path_arr_masks = f"{args.dir_dataset}/init_labelled_pixels_{args.seed}.pkl"
            if n_pixels_per_img != 0 and not val:
                os.makedirs(f"{self.dir_checkpoints}/0_query", exist_ok=True)
                n_pixels_total = 0
                if os.path.isfile(path_arr_masks) and True:
                    list_masks = pkl.load(open(path_arr_masks, 'rb'))
                    for m in list_masks:
                        n_pixels_total += m.sum()

                else:
                    np.random.seed(args.seed)  # for future reproduction

                    list_masks = list()
                    for i in tqdm(range(len(self.voc))):
                        label = self.voc[i][1]
                        w, h = label.size
                        n_pixels_per_img = h * w if n_pixels_per_img == 0 else n_pixels_per_img

                        # generate masks whose size is set to base_size (longer side), i.e. 400 as default
                        h, w = self._compute_base_size(h, w)

                        mask = np.zeros((h, w), dtype=np.bool)
                        mask_flat = mask.flatten()

                        # filter void pixels - boundary pixels that the original labels have (fyi, 5 pixels thickness)
                        label = label.resize((w, h), Image.NEAREST)  # note that downsampling method should be Image.NEAREST
                        label = np.asarray(label, dtype=np.int32)

                        label_flatten = label.flatten()
                        ind_void_pixels = np.where(label_flatten == 255)[0]

                        ind_non_void_pixels = np.setdiff1d(range(len(mask_flat)), ind_void_pixels)  # remove void pixels
                        assert len(ind_non_void_pixels) <= len(mask_flat)

                        # for a very rare case where the number of non_void_pixels is not large enough to sample from
                        if len(ind_non_void_pixels) < n_pixels_per_img:
                            n_pixels_per_img = len(ind_non_void_pixels)

                        ind_chosen_pixels = np.random.choice(ind_non_void_pixels, n_pixels_per_img, replace=False)

                        mask_flat[ind_chosen_pixels] += True
                        mask = mask_flat.reshape((h, w))

                        list_masks.append(mask)
                        n_pixels_total += mask.sum()
                    pkl.dump(list_masks, open(f"{path_arr_masks}", 'wb'))

                # note that unlike CamVid or Cityscapes where all images share the same spatial size, images of voc dataset
                # varies from image to image thus can't use np.stack().
                self.arr_masks = list_masks
                pkl.dump(self.arr_masks, open(f"{self.dir_checkpoints}/0_query/label.pkl", 'wb'))

                self.n_pixels_total = n_pixels_total
                print("# labelled pixels used for training:", n_pixels_total)

        # total_pixels = 0
        # for m in self.arr_masks:
        #     total_pixels += np.prod(m.shape)
        #     assert m.shape[0] <= 400 and m.shape[1] <= 400, m.shape
        # print(total_pixels)
        # exit(12)

        self.val, self.query = val, query

    def label_queries(self, queries, nth_query=None):
        assert len(queries) == len(self.arr_masks), f"{queries.shape}, {len(self.arr_masks)}"
        previous = self.n_pixels_total

        list_masks = list()
        n_pixels_total = 0
        for q, m in zip(queries, self.arr_masks):
            new_m = np.logical_or(q, m)
            list_masks.append(new_m)
            n_pixels_total += new_m.sum()
        self.arr_masks, self.n_pixels_total = list_masks, n_pixels_total

        if isinstance(nth_query, int):
            os.makedirs(f"{self.dir_checkpoints}/{nth_query}_query", exist_ok=True)
            pkl.dump(self.arr_masks, open(f"{self.dir_checkpoints}/{nth_query}_query/label.pkl", 'wb'))

        print("# labelled pixels is changed from {} to {} (delta: {})".format(previous, n_pixels_total, n_pixels_total - previous))

    def _compute_base_size(self, h, w):
        if w > h:
            h = int(float(h) / w * self.size_base)
            w = self.size_base
        else:
            w = int(float(w) / h * self.size_base)
            h = self.size_base
        return h, w

    def _geometric_augmentations(self, x, y, mask=None):
        w, h = x.size
        h, w = self._compute_base_size(h, w)

        x = tf.resize(x, [h, w], Image.BILINEAR)
        y = tf.resize(y, [h, w], Image.NEAREST)

        if self.geometric_augmentations["random_scale"]:
            rs = uniform(0.5, 2.0)

            h, w = int(h * rs), int(w * rs)

            x = tf.resize(x, [h, w], Image.BILINEAR)
            y = tf.resize(y, [h, w], Image.NEAREST)

            if mask is not None:
                mask = Image.fromarray(mask).resize((w, h), Image.NEAREST)

        if self.geometric_augmentations["crop"]:
            w, h = x.size
            pad_h = max(self.size_crop[0] - h, 0)
            pad_w = max(self.size_crop[1] - w, 0)
            self.pad_size = (pad_h, pad_w)

            x = tf.pad(x, [0, 0, pad_w, pad_h], fill=self.mean, padding_mode="constant")
            y = tf.pad(y, [0, 0, pad_w, pad_h], fill=self.ignore_index, padding_mode="constant")

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

        (x, y), mask = self.voc[ind], self.arr_masks[ind] if self.arr_masks is not None else None

        if self.val:
            dict_data.update({"x": self.normalize(tf.to_tensor(x)), "y": self._image_to_tensor(y)})

        elif self.query:
            x, y, mask = self._geometric_augmentations(x, y, mask)
            dict_data.update({"x": self.normalize(tf.to_tensor(x)),
                              "y": self._image_to_tensor(y),
                              "mask": torch.tensor(np.array(mask, np.bool), dtype=torch.bool)})

        else:
            x, y, mask = self._geometric_augmentations(x, y, mask)
            dict_data.update({"x": self.normalize(tf.to_tensor(self._photometric_augmentations(x))),
                              "y": self._image_to_tensor(y),
                              "mask": torch.tensor(np.array(mask, np.bool), dtype=torch.bool)})
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