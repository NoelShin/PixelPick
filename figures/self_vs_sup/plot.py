import os
from pathlib import Path
from glob import glob
import re
from csv import reader
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams


def sort_by(list_files, list_keywords):
    dict_files = dict()
    for k in list_keywords:
        for f in list_files:
            if f.find(k) != -1:
                try:
                    dict_files[k].append(f)
                except KeyError:
                    dict_files.update({k: [f]})
    return dict_files


def get_best_miou(txt_file):
    list_mious = list()
    with open(txt_file, 'r') as f:
        csv_reader = reader(f)
        for i, line in enumerate(csv_reader):
            if i == 0:
                try:
                    ind_miou = line.index("mIoU")
                except ValueError:
                    try:
                        ind_miou = line.index("miou")
                    except ValueError:
                        ind_miou = 0
                        list_mious.append(float(line[ind_miou]))
                continue
            list_mious.append(float(line[ind_miou]))
    try:
        best_miou = np.max(list_mious)
    except ValueError:
        print(f"{txt_file} has no max value.")
        best_miou = np.NaN
    return best_miou


def compute_avg_miou(dict_files):
    dict_miou = dict()
    for k, list_files in dict_files.items():
        list_mious = list()
        for f in list_files:
            list_mious.append(get_best_miou(f))
        dict_miou.update({k: [np.mean(list_mious), np.std(list_mious)]})
    return dict_miou


def get_files(dir_root, fname):
    """
    Get a list of paths of files whose name contains fname from dir_root.
    """
    assert os.path.isdir(dir_root)
    list_files = list()
    for p in Path(dir_root).rglob(fname):
        list_files.append(os.path.abspath(p))
    assert len(list_files) > 0, f"There is no file name matched with {fname} in {dir_root}"
    return sorted(list_files)


def plot_miou(dict_mious,
              xticks=None,
              ls="-", color="tab:blue", label=None, marker='^', unit=10):
    if xticks is None:
        xticks = [unit * (i + 1) for i in range(len(dict_mious.keys()))]
    yticks = [avg[0] for avg in dict_mious.values()]
    yerr = [avg[1] for avg in dict_mious.values()]

    # plot an errorbar
    # plt.errorbar(xticks, yticks, yerr, capsize=0, color=color, ls=ls, label=label)
    plt.plot(xticks, yticks, color=color, ls=ls, label=label, marker=marker)
    # optional: fill area between error ranges.
    data = {
        'x': xticks,
        "y1": [yticks[i] - yerr[i] for i in range(len(yticks))],
        "y2": [yticks[i] + yerr[i] for i in range(len(yticks))]
    }
    plt.fill_between(**data, alpha=0.15, color=color)
    plt.ylabel("mIoU")
    plt.legend(loc="lower right", fancybox=False, framealpha=1., edgecolor='black')


def plot_model(model,
               fname="log_val.txt",
               list_keywords=[f"_{i}_query" for i in range(10)],
               xticks=None,
               ls='-', color="tab:blue", marker='^', unit=10):
    list_files = get_files(f"{model}", fname=fname)
    dict_files = sort_by(list_files, list_keywords=list_keywords)
    dict_mious = compute_avg_miou(dict_files)

    plot_miou(dict_mious, xticks=xticks,
              label=f"{model.split('/')[-1]}", ls=ls, color=color, unit=unit, marker=marker)


if __name__ == '__main__':
    # plt.plot(figsize=(7, 5))
    rcParams["font.family"] = "serif"
    rcParams["grid.linestyle"] = ':'

    DATASET = "cv"

    if DATASET == "cv":
        title = "CamVid"
        unit = 10
        yrange = () # (0.4, 0.75)

        plot_model(f"{DATASET}/moco_v2",
                   fname="val_log.txt",
                   list_keywords=[f"-{i}-" for i in [*list(range(1, 10)), 20, 50, 100, 1000, 10000]],
                   xticks=[*list(range(1, 10)), 20, 50, 100, 1000, 10000], ls="--", color="b", unit=unit, marker='v')

        plot_model(f"{DATASET}/sup",
                   fname="val_log.txt",
                   list_keywords=[f"-{i}-" for i in [*list(range(1, 10)), 20, 50, 100, 1000, 10000]],
                   xticks=[*list(range(1, 10)), 20, 50, 100, 1000, 10000], ls="--", color="r", unit=unit, marker='v')

    elif DATASET == "cs_d4":
        title = "Cityscapes"
        unit = 1
        yrange = ()

        plot_model(f"{DATASET}/moco_v2",
                   fname="log_val.txt",
                   list_keywords=[f"random_{i}_" for i in [*list(range(1, 11)), 100, 1000, 10000]],
                   xticks=[*list(range(1, 11)), 100, 1000, 10000], ls="--", color="b", unit=unit, marker='v')

        plot_model(f"{DATASET}/sup",
                   fname="log_val.txt",
                   list_keywords=[f"random_{i}_" for i in [*list(range(1, 11)), 100, 1000, 10000]],
                   xticks=[*list(range(1, 11)), 100, 1000, 10000], ls="--", color="r", unit=unit, marker='^')

    elif DATASET == "voc":
        title = "PASCAL VOC 2012"
        unit = 1
        yrange = ()

        plot_model(f"{DATASET}/moco_v2",
                   fname="val_log.txt",
                   list_keywords=[f"-{i}-" for i in [*list(range(1, 10)), 20, 50, 100, 1000, 10000]],
                   xticks=[*list(range(1, 10)), 20, 50, 100, 1000, 10000], ls="--", color="b", unit=unit, marker='v')

        plot_model(f"{DATASET}/sup",
                   fname="val_log.txt",
                   list_keywords=[f"-{i}-" for i in [*list(range(1, 10)), 20, 50, 100, 1000, 10000]],
                   xticks=[*list(range(1, 10)), 20, 50, 100, 1000, 10000], ls="--", color="r", unit=unit, marker='v')

    # plot_model(f"{DATASET}/sup", fname="val_log.txt", color="r", unit=unit)

    # vote baselines
    # plt.title(f"{title}")
    plt.xlabel("# pixels per img")

    plt.xscale('log')
    plt.grid()
    plt.ylim(*yrange)

    plt.tight_layout()
    plt.savefig(f"self_vs_sup_{title}.png")
    plt.show()

