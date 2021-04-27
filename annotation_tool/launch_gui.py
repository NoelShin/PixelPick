dataset_to_paths = {
    "camvid": {
        "dir_imgs": "/Users/noel/projects/PixelPick/annotation_tool/camvid/train",  # Directory containing the images
        "dir_gts": "/Users/noel/projects/PixelPick/annotation_tool/camvid/trainannot",  # Directory containing the groundtruth labels
        "path_query": "../query.npy"  # Path to your query file
    }
}

if __name__ == '__main__':
    import os
    from glob import glob
    from argparse import ArgumentParser

    import numpy as np
    from matplotlib.pyplot import rcParams
    from PIL import Image
    from tqdm import tqdm
    from time import time
    from datetime import datetime

    from utils.utils import *

    parser = ArgumentParser("Mouse-free Annotation")

    parser.add_argument("--dataset_name", type=str, default="camvid", choices=["camvid"])
    parser.add_argument("--display_all_queries", '-a', action="store_true", help="if True, display all queried pixels")

    parser.add_argument("--edge_size", '-es', type=float, default=2, help="size of edge for queries")
    parser.add_argument("--marker_size", '-ms', type=float, default=5, help="size of marker for queries")

    parser.add_argument("--n_imgs", type=int, default=-1, help="Data size of random subset. Set -1 to use all data.")

    args = parser.parse_args()

    dict_paths = dataset_to_paths[args.dataset_name]
    dir_imgs, dir_gts, path_query = dict_paths["dir_imgs"], dict_paths["dir_gts"], dict_paths["path_query"]
    assert os.path.isdir(f"{dir_imgs}"), f"There is no '{dir_imgs}' directory."
    assert os.path.isdir(f"{dir_gts}"), f"There is no '{dir_gts}' directory."
    assert os.path.isfile(f"{path_query}"), f"There is no '{path_query}' file."

    # Create a directory for saving a log file
    dir_root = f"{args.dataset_name}"
    dir_log = f"../logs/{args.dataset_name}_{datetime.now().strftime('%b_%d_%H_%M')}"
    os.makedirs(dir_log, exist_ok=True)

    # Get img/label paths
    path_imgs, path_gt_labels = sorted(glob(f"{dir_imgs}/*.png")), sorted(glob(f"{dir_gts}/*.png"))
    assert len(path_imgs) > 0, f"No images (png) are found in '{dir_imgs}'."
    assert len(path_gt_labels) > 0, f"No labels (png) are found in '{dir_gts}'."

    print("Loading images...")
    imgs, gt_labels = list(), list()
    for path_img, path_gt in tqdm(zip(path_imgs, path_gt_labels)):
        imgs.append(np.array(Image.open(path_img)))
        gt_labels.append(np.array(Image.open(path_gt)))

    print("Loading queries...")
    queries = np.load(f"{path_query}").astype(np.bool)

    # Check imgs and queries have the same number of files.
    assert len(imgs) == len(queries), f"# imgs ({len(imgs)}) != # masks ({len(queries)})"

    if args.n_imgs > 0:
        indices = np.random.choice(range(len(imgs)), args.n_imgs, False)
        print(f"Random {args.n_imgs} images are selected...", indices)
    else:
        indices = range(len(imgs))
        print(f"{len(imgs)} images are selected...")

    if args.dataset_name == "camvid":
        label_category, alphabet, alphabet_l = cv_label_category, alphabet_cv, alphabet_l_cv
    else:
        raise ValueError(f"Invalid value for dataset_name argument: {args.dataset_name}")

    # Create a logger
    logger = Logger(dir_log)
    acc, timer = AverageMeter(), AverageMeter()

    rcParams["font.family"] = "serif"

    # Annotate images
    tbar = tqdm(enumerate(indices))
    for i, index in tbar:
        x, y, gt = imgs[index], queries[index], gt_labels[index]
        total_time = 0
        fname = path_imgs[index].split('/')[-1].split('.')[0]
        logger(fname, "loc,label,elapsed_time,total_time\n", 'w')

        loc_points = sorted(list(zip(*np.where(y))), key=lambda v: v[1])

        # display all queries in an image
        if args.display_all_queries:
            for ind, p in enumerate(loc_points):
                x = color_points(x, *p, ms=args.marker_size, es=args.edge_size)

        # iterate over queried pixels in an image
        for ind in range(len(loc_points)):
            if args.display_all_queries:
                if ind == 0:
                    pass
                else:
                    x = color_points(x, *loc_points[ind-1], ms=args.marker_size, es=args.edge_size)

            else:
                # new image instance
                x = imgs[index]  # h x w x 3

            x = color_points(x, *loc_points[ind], fc=(255, 0, 0), ms=args.marker_size, es=args.edge_size)
            p_img = Image.fromarray(x)
            gt_p = gt[loc_points[ind]]

            # create and save a gui window
            tmp = make_gui(p_img, args.dataset_name, figsize=(8, 8))  # (7.2, 4.8))  # (12.20, 10.80))

            now = time()
            label = annotate(tmp, args.dataset_name)
            delta = time() - now
            total_time += delta

            timer.update(delta)
            acc.update(float(alphabet_l.index(label) == gt_p))

            tbar.set_description(f"Image: {i + 1}(/{len(indices)}) | "
                                 f"Entered label: {label_category[alphabet_l.index(label)].lower()} | "
                                 f"GT label: {label_category[gt_p].lower()} | "
                                 f"Avg acc.: {acc.avg * 100:.2f}% | "
                                 f"Avg time: {timer.avg:.2f} sec. | "
                                 f"Avg time per img: {(timer.avg * 10):.2f} sec.")

            logger(fname,
                   f"{loc_points[ind]},{label_category[alphabet_l.index(label)].lower()},{delta},{total_time}\n",
                   'a')
