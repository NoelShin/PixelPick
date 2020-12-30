import os
from argparse import ArgumentParser
from utils.utils import write_log


class Arguments:
    def __init__(self):
        parser = ArgumentParser("")

        parser.add_argument("--debug", "-d", action="store_true", default=False)
        parser.add_argument("--dir_root", type=str, default="..")
        parser.add_argument("--seed", "-s", type=int, default=0)
        parser.add_argument("--model_name", type=str, default="gcpl_seg", choices=["gcpl_seg", "mp_seg"])
        parser.add_argument("--n_pixels_per_img", type=int, default=0)
        parser.add_argument("--use_aug", action='store_true', default=False)
        parser.add_argument("--use_softmax", action='store_true', default=False)
        parser.add_argument("--suffix", type=str, default=None)

        # active learning
        parser.add_argument("--active_learning", action="store_true", default=False)
        parser.add_argument("--epoch_add_labels", type=int, default=5)
        parser.add_argument("--n_add_labels", type=int, default=5)
        parser.add_argument("--n_pixels_per_add_labels", type=int, default=1)

        # system
        parser.add_argument("--gpu_ids", type=str, nargs='+', default='0')
        parser.add_argument("--n_workers", type=int, default=4)

        # dataset
        parser.add_argument("--dataset_name", type=str, default="voc", choices=["cv", "voc"])
        parser.add_argument("--dir_datasets", type=str, default="/scratch/shared/beegfs/gyungin/datasets")
        parser.add_argument("--use_augmented_dataset", action="store_true", default=False, help="whether to use the augmented dataset for pascal voc")

        # gcpl
        parser.add_argument("--ignore_bg", action="store_true", default=False, help="ignore_bg of voc datasets")
        parser.add_argument("--n_prototypes", type=int, default=1)
        parser.add_argument("--loss_type", type=str, default="dce", choices=["dce"])
        parser.add_argument("--use_pl", action="store_true", default=False, help="prototype loss")
        parser.add_argument("--w_pl", type=float, default=0.001, help="weight for prototype loss")
        parser.add_argument("--use_repl", action="store_true", default=False, help="repulsive loss")
        parser.add_argument("--w_repl", type=float, default=1, help="weight for repulsive loss")
        parser.add_argument("--use_vl", action="store_true", default=False, help="prototype loss")
        parser.add_argument("--w_vl", type=float, default=1, help="weight for prototype loss")

        parser.add_argument("--non_isotropic", action="store_true", default=False)
        parser.add_argument("--n_emb_dims", type=int, default=32)

        # encoder
        parser.add_argument("--weight_type", type=str, default="supervised",
                            choices=["random", "supervised", "moco_v2", "swav", "deepcluster_v2"])
        parser.add_argument("--use_dilated_resnet", type=bool, default=True, help="whether to use dilated resnet")
        self.parser = parser

    def parse_args(self):
        args = self.parser.parse_args()

        args.stride_total = 8 if args.use_dilated_resnet else 32
        if args.dataset_name == "cv":
            args.batch_size = 5
            args.dir_dataset = "/scratch/shared/beegfs/gyungin/datasets/camvid"
            args.ignore_index = 11
            args.mean = [0.41189489566336, 0.4251328133025, 0.4326707089857]
            args.std = [0.27413549931506, 0.28506257482912, 0.28284674400252]
            args.n_classes = 11
            args.n_epochs = 50

            args.optimizer_type = "Adam"
            args.optimizer_params = {
                "lr": 5e-4,
                "betas": (0.9, 0.999),
                "weight_decay": 2e-4,
                "eps": 1e-7
            }

            args.augmentations = {
                "geometric": {
                    "random_scale": args.use_aug,
                    "random_hflip": args.use_aug,
                    "crop": args.use_aug
                },

                "photometric": {
                    "random_color_jitter": args.use_aug,
                    "random_grayscale": args.use_aug,
                    "random_gaussian_blur": args.use_aug
                }
            }

        elif args.dataset_name == "voc":
            args.batch_size = 9
            args.dir_dataset = "/scratch/shared/beegfs/gyungin/datasets/VOC2012"
            args.dir_augmented_dataset = "/scratch/shared/beegfs/gyungin/datasets/VOC2012/VOCdevkit/VOC2012/train_aug"
            args.ignore_index = 255
            args.mean = [0.485, 0.456, 0.406]
            args.std = [0.229, 0.224, 0.225]
            args.n_classes = 21 if args.use_softmax or not args.ignore_bg else 20

            args.n_epochs = 50

            args.size_base = 400
            args.size_crop = 320

            args.optimizer_type = "SGD"
            args.optimizer_params = {
                "lr": 1e-2,
                "weight_decay": 1e-4,
                "momentum": 0.9
            }

            args.augmentations = {
                "geometric": {
                    "random_scale": args.use_aug,
                    "random_hflip": args.use_aug,
                    "crop": args.use_aug
                },

                "photometric": {
                    "random_color_jitter": False,
                    "random_grayscale": False,
                    "random_gaussian_blur": False
                }
            }

        # naming
        list_keywords = list()
        list_keywords.append(args.dataset_name)
        list_keywords.append("aug") if args.use_aug else "None"

        if args.use_softmax:
            list_keywords.append("sm")

        elif args.model_name == "gcpl_seg":
            list_keywords.append("gcpl_seg")
            list_keywords.append(f"k_{args.n_prototypes}")
            list_keywords.append(f"n_emb_dims_{args.n_emb_dims}")

            if args.use_pl:
                list_keywords.append("pl")
                list_keywords.append(str(args.w_pl))

            if args.use_repl:
                list_keywords.append("repl")
                list_keywords.append(str(args.w_repl))

            if args.use_vl:
                list_keywords.append("vl")
                list_keywords.append(str(args.w_vl))

        elif args.model_name == "mp_seg":
            list_keywords.append("mp_seg")
            list_keywords.append(f"n_emb_dims_{args.n_emb_dims}")

            args.momentum_prototypes = 0.99
            args.momentum_radii = 0.99

            if args.use_pl:
                list_keywords.append("pl")
                list_keywords.append(str(args.w_pl))

            if args.use_repl:
                list_keywords.append("repl")
                list_keywords.append(str(args.w_repl))

        list_keywords.append(f"n_pixels_{args.n_pixels_per_img}")
        list_keywords.append(str(args.seed))
        list_keywords.append(args.suffix) if args.suffix is not '' else None
        list_keywords.append("debug") if args.debug else None
        args.experim_name = '_'.join(list_keywords)

        # create dirs
        args.dir_checkpoints = f"{args.dir_root}/checkpoints/{args.experim_name}"
        os.makedirs(f"{args.dir_checkpoints}/train", exist_ok=True)
        os.makedirs(f"{args.dir_checkpoints}/val", exist_ok=True)

        # initialise logs
        args.log_train = f"{args.dir_checkpoints}/log_train.txt"
        args.log_val = f"{args.dir_checkpoints}/log_val.txt"
        write_log(args.log_train, header=["epoch", "mIoU", "pixel_acc", "loss"])
        write_log(args.log_val, header=["epoch", "mIoU", "pixel_acc"])

        return args