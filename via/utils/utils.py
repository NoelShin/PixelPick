from typing import Tuple
import re
from pathlib import Path
import io
import numpy as np
import cv2
from matplotlib import rcParams
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, DrawingArea, HPacker, VPacker


def combine_files(dir_root: str, pattern: re.Pattern, ext="txt"):
    list_file_paths: list = [str(p.resolve()) for p in Path(dir_root).rglob(f"{ext}")]
    list_matching_file_paths: list = list()
    for fp in list_file_paths:
        if pattern.match(fp) is not None:
            list_matching_file_paths.append(fp)
    return list_matching_file_paths


def annotate(arr, keyword_category_mapping: dict):
    cv2.namedWindow('image')
    cv2.imshow('image', arr)

    predefined_keywords: list = list(keyword_category_mapping.keys())
    while True:
        k = cv2.waitKey()
        if chr(k) not in predefined_keywords:
            print(f"\nYou've clicked a wrong character: {chr(k)}\n")

        elif chr(k) in predefined_keywords:
            break

        elif k == 27:
            break

    cv2.destroyAllWindows()
    return chr(k)


def color_points(x, loc_y, loc_x, fc=(0, 128, 192), ec=(255, 255, 255), ms=5, es=2):
    x = cv2.circle(x, (loc_x, loc_y), ms + es, color=ec, thickness=-1)
    x = cv2.circle(x, (loc_x, loc_y), ms, color=fc, thickness=-1)
    return x


def coords_to_grid(size: Tuple[int, int], x_coords, y_coords) -> np.ndarray:
    grid = np.zeros(size, dtype=np.bool)
    for x, y in zip(x_coords, y_coords):
        grid[y, x] = True
    return grid


def make_gui(img: np.ndarray, keyword_category_mapping: dict, **kwargs):
    fig = plt.figure(figsize=kwargs["figsize"])
    rcParams['axes.linewidth'] = 2
    rcParams['axes.edgecolor'] = 'k'

    plt.imshow(img)

    vpacker_children = [
        TextArea("{} - {}".format(k, cat), textprops={"weight": 'bold', "size": 10})
        for k, cat in keyword_category_mapping.items()
    ]
    box = VPacker(children=vpacker_children, align="left", pad=5, sep=5)

    # Display the texts on the right side of image
    anchored_box = AnchoredOffsetbox(
        loc="center left",
        child=box,
        pad=0.,
        frameon=True,
        bbox_to_anchor=(1.04, 0.5),
        bbox_transform=plt.gca().transAxes,
        borderpad=0.
    )
    anchored_box.patch.set_linewidth(2)
    anchored_box.patch.set_facecolor('gray')
    anchored_box.patch.set_alpha(0.2)

    anchored_box.patch.set_boxstyle("round,pad=0.5, rounding_size=0.2")
    plt.gca().add_artist(anchored_box)

    # create texts for "Enter a label for the current marker"
    box1 = TextArea("Enter a label for the current marker",
                    textprops={"weight": 'bold', "size": 12})
    box2 = DrawingArea(5, 10, 0, 0)
    box2.add_artist(mpatches.Circle((5, 5), radius=5, fc=np.array((1, 0, 0)), edgecolor="k", lw=1.5))
    box = HPacker(children=[box1, box2], align="center", pad=5, sep=5)

    # anchored_box creates the text box outside of the plot
    anchored_box = AnchoredOffsetbox(
        loc="lower center",
        child=box,
        pad=0.,
        frameon=False,
        bbox_to_anchor=(0.5, -0.1),  # ( 0.5, -0.1)
        bbox_transform=plt.gca().transAxes,
        borderpad=0.
    )
    plt.gca().add_artist(anchored_box)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout(pad=2)

    buf = io.BytesIO()
    fig.savefig(buf, format="jpg", dpi=80)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    im = cv2.imdecode(img_arr, 1)
    plt.close()
    return im


class Logger:
    def __init__(self, dir_log):
        self.dir_log = dir_log

    def __call__(self, fname, line, mode):
        assert mode in ['a', 'r', 'w'], f"Invalid mode: {mode}"
        with open(f"{self.dir_log}/{fname}.txt", mode) as f:
            f.write(line)
            f.close()


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.initialized = False
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.initialized = False
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = np.multiply(val, weight)
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum = np.add(self.sum, np.multiply(val, weight))
        self.count = self.count + weight
        self.avg = self.sum / self.count

    @property
    def value(self):
        return self.val

    @property
    def average(self):
        return np.round(self.avg, 5)