import os
from glob import glob
from typing import Dict, List
import pickle as pkl
from PIL import Image
import numpy as np
import torch
import torchvision.transforms.functional as TF
from tqdm import tqdm
from datasets.base_dataset import BaseDataset
from query import QuerySelector


class CustomDataset(BaseDataset):
    def __init__(
            self,
            args,
            val=False,
            query=False,
            generate_init_queries: bool = True
    ):
        super(CustomDataset, self).__init__()
        assert os.path.isdir(args.dir_dataset), f"{args.dir_dataset} does not exist."
        self.dataset_name = args.dataset_name
        self.dir_checkpoints = f"{args.dir_root}/checkpoints/{args.experim_name}"
        self.seed = args.seed

        # get data paths
        mode = "test" if val else "train"
        self.list_inputs = sorted(glob(f"{args.dir_dataset}/{mode}/*.{args.img_ext}"))
        assert len(self.list_inputs) > 0, ValueError(len(self.list_inputs))
        if mode != "train":
            self.list_labels = sorted(glob(f"{args.dir_dataset}/{mode}annot/*.{args.img_ext}"))
            assert len(self.list_inputs) == len(self.list_labels)
        else:
            self.list_labels = None

        # seg variables for data augmentation
        self.geometric_augmentations = args.augmentations["geometric"]
        self.photometric_augmentations = args.augmentations["photometric"]
        self.mean, self.std = args.mean, args.std
        self.n_classes = args.n_classes
        self.crop_size, self.pad_size = args.crop_size, (0, 0)
        self.ignore_index = args.ignore_index
        self.query = query
        self.val = val

        if self.geometric_augmentations["crop"]:
            self.mean_val = tuple((np.array(self.mean) * 255.0).astype(np.uint8).tolist())

        self.queries, self.n_pixels_total = None, -1
        n_pixels_per_img = args.n_pixels_by_us

        if n_pixels_per_img != 0 and not val and generate_init_queries:
            path_queries = f"{self.dir_checkpoints}/0_query/queries.pkl"
            if os.path.isfile(path_queries):
                dict_queries: Dict[str, dict] = pkl.load(open(path_queries, "rb"))
                self.queries: List[np.ndarray] = QuerySelector.decode_queries(dict_queries)

            else:
                # generate initial queries
                dict_queries: Dict[str, dict] = dict()
                np.random.seed(self.seed)

                # iterate over images to get masks
                for i in tqdm(range(len(self.list_inputs))):
                    img = np.array(Image.open(self.list_inputs[i]).convert("RGB"))  # H x W x 3
                    h, w = img.shape[:2]

                    # pick random pixel locations as many as specified
                    ind_candidate = range(h * w)
                    ind_chosen_pixels = np.random.choice(ind_candidate, n_pixels_per_img, replace=False)

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

    def __getitem__(self, ind):
        dict_data = dict()
        p_img: str = self.list_inputs[ind]
        x = Image.open(p_img).convert("RGB")
        # y = Image.open(self.list_labels[ind])

        # if a ground-truth label for all pixels within an image is available (esp., for validation)
        if self.list_labels is not None:
            y = Image.open(self.list_labels[ind])
        else:
            y = None

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
                x, y=y, queries=queries, labelled_queries=labelled_queries
            )
            x = self._photometric_augmentations(x)
        else:
            queries = torch.from_numpy(np.asarray(queries, dtype=np.uint8) // 255)

        dict_data.update({
            'x': TF.normalize(TF.to_tensor(x), self.mean, self.std),
            "p_img": p_img
        })
        for k, v in {'y': y, "queries": queries, "labelled_queries": labelled_queries}.items():
            if v is not None:
                if k == 'y':
                    dict_data[k] = torch.tensor(np.asarray(y, np.int64), dtype=torch.long)
                else:
                    dict_data[k] = v

        # dict_data.update({
        #     'x': TF.normalize(TF.to_tensor(x), self.mean, self.std),
        #     'y': torch.tensor(np.asarray(y, np.int64), dtype=torch.long),
        #     "queries": queries,
        #     "labelled_queries": labelled_queries,
        #     "p_img": p_img
        # })
        return dict_data
