import os
from glob import glob
from typing import Dict, List
import pickle as pkl
from PIL import Image
import numpy as np
from tqdm import tqdm
from datasets.base_dataset import BaseDataset
from query import QuerySelector


class CamVidDataset(BaseDataset):
    def __init__(
            self,
            args,
            val=False,
            query=False,
            generate_init_queries: bool = True
    ):
        super(CamVidDataset, self).__init__()
        assert os.path.isdir(args.dir_dataset), f"{args.dir_dataset} does not exist."
        self.dataset_name = "camvid"
        self.dir_checkpoints = f"{args.dir_root}/checkpoints/{args.experim_name}"
        self.seed = args.seed

        # get data paths
        mode = "test" if val else "train"
        self.list_inputs = sorted(glob(f"{args.dir_dataset}/{mode}/*.png"))
        self.list_labels = sorted(glob(f"{args.dir_dataset}/{mode}annot/*.png"))
        assert len(self.list_inputs) > 0
        if mode != "train":
            assert len(self.list_inputs) == len(self.list_labels)

        # seg variables for data augmentation
        self.geometric_augmentations = args.augmentations["geometric"]
        self.photometric_augmentations = args.augmentations["photometric"]
        self.mean, self.std = args.mean, args.std
        self.n_classes = 11
        self.crop_size, self.pad_size = (360, 480), (0, 0)
        self.ignore_index = args.ignore_index
        self.query = query
        self.val = val

        if self.geometric_augmentations["crop"]:
            self.mean_val = tuple((np.array(self.mean) * 255.0).astype(np.uint8).tolist())

        self.queries, self.n_pixels_total = None, -1
        n_pixels_per_img = args.n_pixels_by_us

        if n_pixels_per_img != 0 and not val and generate_init_queries:
            # if os.path.isfile(args.p_query_file):
            #     path_queries = args.p_query_file
            #
            # else:
            path_queries = f"{self.dir_checkpoints}/0_query/queries.pkl"
            if os.path.isfile(path_queries):
                dict_queries: Dict[str, dict] = pkl.load(open(path_queries, "rb"))
                self.queries: List[np.ndarray] = QuerySelector.decode_queries(dict_queries)

            else:
                # generate initial queries
                dict_queries: Dict[str, dict] = dict()
                np.random.seed(self.seed)

                # iterate over labels to get masks
                # for i in tqdm(range(len(self.list_labels))):
                for i in tqdm(range(len(self.list_inputs))):
                    label = np.array(Image.open(self.list_labels[i]))  # H x W
                    h, w = label.shape
                    # h, w = np.array(Image.open(self.list_inputs[i])).shape[:2]

                    # exclude pixels with ignore_index
                    ind_non_void_pixels = np.where(label.flatten() != self.ignore_index)[0]
                    ind_chosen_pixels = np.random.choice(ind_non_void_pixels, n_pixels_per_img, replace=False)

                    # ind_candidate = range(h * w)
                    # ind_chosen_pixels = np.random.choice(ind_candidate, n_pixels_per_img, replace=False)

                    query_flat = np.zeros((h, w), dtype=np.bool).flatten()
                    query_flat[ind_chosen_pixels] = True
                    query = query_flat.reshape((h, w))

                    query_info: dict = QuerySelector.encode_query(p_img=self.list_inputs[i], size=(h, w), query=query)
                    dict_queries.update(query_info)

                self.queries: List[np.ndarray] = QuerySelector.decode_queries(dict_queries)

                # save initial labelled pixels for a future reproduction
                os.makedirs(f"{self.dir_checkpoints}/0_query", exist_ok=True)
                pkl.dump(dict_queries, open(path_queries, "wb"))

            self.n_pixels_total = 0
            for q in self.queries:
                self.n_pixels_total += q.sum()
            print("total number of labelled pixels selected as queries:", self.n_pixels_total)
            print(f"queries are saved at {path_queries}")

    def __len__(self):
        return len(self.list_inputs)
