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
              ls="-", color="tab:blue", label=None, marker='^', unit=10, ms=1.0,
              fill_between=True):
    if xticks is None:
        xticks = [unit * (i + 1) for i in range(len(dict_mious.keys()))]
    yticks = [avg[0] for avg in dict_mious.values()]
    yerr = [avg[1] for avg in dict_mious.values()]

    # plot an errorbar
    # plt.errorbar(xticks, yticks, yerr, capsize=0, color=color, ls=ls, label=label)
    plt.plot(xticks, yticks, color=color, ls=ls, label=label, marker=marker, ms=ms)
    # optional: fill area between error ranges.
    if fill_between:
        data = {
            'x': xticks,
            "y1": [yticks[i] - yerr[i] for i in range(len(yticks))],
            "y2": [yticks[i] + yerr[i] for i in range(len(yticks))]
        }
        plt.fill_between(**data, alpha=0.15, color=color)
    plt.ylabel("mIoU")
    plt.legend(loc="lower right", fancybox=False, framealpha=1., edgecolor='black')


def plot_model(model,
               xticks=None,
               list_keywords=None,
               **kwargs):
    list_files = get_files(f"{model}", fname="log_val.txt")

    if list_keywords is not None:
        dict_files = sort_by(list_files, list_keywords=list_keywords)
    else:
        dict_files = sort_by(list_files, list_keywords=[f"_{i}_query" for i in range(10)])
    dict_mious = compute_avg_miou(dict_files)

    plot_miou(dict_mious,
              xticks=xticks,
              **kwargs)


def draw_hline(dir_root, model_name, xmin, xmax, c='b', ls='--', alpha=1.0):
    list_txt_files = sorted(glob(os.path.join(dir_root, '**/log_val.txt'), recursive=True))
    dict_model_values = {model_name: list()}
    for txt_file in list_txt_files:
        list_mious = list()
        with open(txt_file, 'r') as f:
            csv_reader = reader(f)
            for i, line in enumerate(csv_reader):
                if i == 0:
                    try:
                        ind_miou = line.index("mIoU")
                    except ValueError:
                        ind_miou = line.index("miou")
                    assert ind_miou != -1
                    continue

                list_mious.append(float(line[ind_miou]))

            best_miou = np.max(list_mious)
            dict_model_values[model_name].append(best_miou)
    plt.hlines(np.mean(dict_model_values[model_name]), xmin=xmin, xmax=xmax, color=c, ls=ls, label=model_name, alpha=alpha)


if __name__ == '__main__':
    # plt.plot(figsize=(7, 5))
    rcParams["font.family"] = "serif"
    rcParams["grid.linestyle"] = ':'

    DATASET = "cv"
    mode = 2

    if mode == 0:
        title = "CamVid"
        unit = 10
        yrange = ()
        # yrange = (0.35, 0.65)

        # plot_model(f"{DATASET}/deeplab_v3/rand",
        #            xticks=[i / (360 * 480) * 100 for i in range(10, 110, 10)],
        #            color="gray", unit=unit, marker='o')

        plt.figure(figsize=(7, 5))



        plot_model(f"{DATASET}/deeplab_v3/ms",
                   label="PixelPick (Ours, MobileNetv2)",
                   # label="PixelPick (DeepLabv3+)",
                   xticks=[i / (360 * 480) * 100 for i in range(10, 110, 10)],
                   color="b", unit=unit, ls='--', marker='o', ms=3.5)

        plot_model(f"{DATASET}/fpn50/ms",
                   label="PixelPick (Ours, ResNet50)",
                   # label="PixelPick (FPN50)",
                   xticks=[i / (360 * 480) * 100 for i in range(10, 110, 10)],
                   color="darkblue", ls='-', unit=unit, marker='o', ms=3.5)

        plot_model(f"{DATASET}/deeplab_v3/deal",
                   list_keywords=[f"_runs_{i:03d}" for i in range(1, 11)],
                   label="DEAL (MobileNetv2)",
                   # label="DEAL (DeepLabv3+)",
                   xticks=[i for i in range(1, 11)],
                   color="y", unit=unit, marker='o', ms=3.5)

        draw_hline(f"{DATASET}/fpn50/full", "Fully-sup (ResNet50)", xmin=5e-3, xmax=40, alpha=1.0, c='k', ls="-.")
        # 5e-3
        plt.plot([8, 12, 16, 20, 24], [0.567, 0.6182, 0.6274, 0.6341, 0.6385],
                 marker='o', ls='--', label="EquAL (ResNet50)", color='g', ms=3.5)
        plt.plot([8, 12, 16, 20, 24], [0.6213, 0.6446, 0.6492, 0.656, 0.663], marker='o', label="EquAL+ (ResNet50)", color='darkgreen', ms=3.5)

        plt.plot([50 / 367 * 100, 100 / 367 * 100], [0.537, 0.557], label="CCT (ResNet50)", marker='o', color="darkred", ms=3.5)
        # plt.xscale('log')

        gca = plt.gca()
        gca.tick_params(direction='in', which='both')
        plt.legend(loc="lower left", fancybox=False, framealpha=1., edgecolor='black')
        # plt.legend(loc='lower right', fancybox=False, framealpha=1., edgecolor='black')
        # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fancybox=False, framealpha=1., edgecolor='black')
        # plt.xlim(-2, 25)
        plt.xscale("log")
        plt.xlabel("% annotation")

    elif mode == 1:
        title = "CamVid"
        unit = 10
        yrange = ()
        # yrange = (0.35, 0.65)

        # plot_model(f"{DATASET}/deeplab_v3/rand",
        #            xticks=[i for i in range(10, 110, 10)],
        #            color="gray", unit=unit, marker='o')

        plt.figure(figsize=(7, 5))
        ms = 4
        plot_model(f"{DATASET}/deeplab_v3/ms",
                   label="MS",
                   xticks=[i for i in range(10, 110, 10)],
                   color="blue", unit=unit, ls='-', marker='o', ms=ms, fill_between=False)

        plot_model(f"{DATASET}/deeplab_v3/ent",
                   label="ENT",
                   xticks=[i for i in range(10, 110, 10)],
                   color="red", unit=unit, marker='^', ms=ms, fill_between=False)

        plot_model(f"{DATASET}/deeplab_v3/lc",
                   label="LC",
                   xticks=[i for i in range(10, 110, 10)],
                   color="green", unit=unit, marker='v', ms=ms, fill_between=False)

        plot_model(f"{DATASET}/deeplab_v3/ms_soft",
                   label="QBC (MS)",
                   xticks=[i for i in range(10, 110, 10)],
                   color="tab:blue", unit=unit, ls='--', marker='o', ms=ms, fill_between=False)

        plot_model(f"{DATASET}/deeplab_v3/ent_soft",
                   label="QBC (ENT)",
                   xticks=[i for i in range(10, 110, 10)],
                   color="tab:red", unit=unit, ls='--', marker='^', ms=ms, fill_between=False)

        plot_model(f"{DATASET}/deeplab_v3/lc_soft",
                   label="QBC (LC)",
                   xticks=[i for i in range(10, 110, 10)],
                   color="tab:green", unit=unit, ls='--', marker='v', ms=ms, fill_between=False)

        plt.xticks([i for i in range(10, 110, 10)], )

        gca = plt.gca()
        gca.tick_params(direction='in', which='both')
        # plt.legend(loc="lower left", fancybox=False, framealpha=1., edgecolor='black')
        plt.legend(loc='lower right', fancybox=False, framealpha=1., edgecolor='black')
        # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fancybox=False, framealpha=1., edgecolor='black')

        plt.xlabel("% annotation")

    elif mode == 2:
        title = "cs"

        unit = 1
        yrange = ()

        plt.figure(figsize=(7, 5))
        plot_model(f"cs_d4/deeplab_v3/ms",
                   label="PixelPick (Ours, MobileNetv2)",
                   xticks=[i / (256 * 512) * 100 for i in range(1, 11, 1)],
                   color="blue", unit=unit, ls='--', marker='o', ms=3.5)

        plot_model(f"cs_d4/fpn50/ms",
                   label="PixelPick (Ours, ResNet50)",
                   xticks=[i / (256 * 512) * 100 for i in range(1, 8, 1)],
                   color="darkblue", ls='-', unit=unit, marker='o', ms=3.5)

        plt.plot([1, 2, 3, 6, 9, 12, 15], [0.4242, 0.4880, 0.5184, 0.5713, 0.5890, 0.6065, 0.6185],
                 marker='o', ls='--', label="EquAL (ResNet50)", color='green', ms=3.5)
        plt.plot([1, 2, 3, 6, 9, 12, 15], [0.4885, 0.5590, 0.5890, 0.6224, 0.6352, 0.6446, 0.6562],
                 marker='o', label="EquAL+ (ResNet50)", color='green', ms=3.5)

        plt.plot([50 / 2975 * 100, 100 / 2975 * 100], [0.35, 0.401], label="CCT (ResNet50)", marker='o', color="red", ms=3.5)

        plt.legend()
        plt.xscale("log")
        plt.xlabel("% annotation")

    elif DATASET == "voc":
        title = "PASCAL VOC 2012"
        unit = 1
        yrange = ()

        plot_model(f"{DATASET}/deeplab_v3/rand", color="tab:blue", unit=unit, marker='o')
        plot_model(f"{DATASET}/deeplab_v3/ms", color="tab:green", unit=unit, marker='^')
        plot_model(f"{DATASET}/deeplab_v3/ent", color="tab:red", unit=unit, marker='s')
        plot_model(f"{DATASET}/deeplab_v3/lc", color="tab:purple", unit=unit, marker='p')
        plt.xlabel("# pixels per img")

    # plt.title(f"{title}")

    plt.grid()
    plt.ylim(*yrange)

    plt.tight_layout()
    plt.savefig(f"sota_{title}.png")
    plt.show()

