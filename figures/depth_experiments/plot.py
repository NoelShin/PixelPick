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
            print(k, f, re.search(k, f))
            if re.search(k, f) is not None:  # f.find(k) != -1:
                try:
                    dict_files[k].append(f)
                except KeyError:
                    dict_files.update({k: [f]})
    # for k, v in dict_files.items():
    #     print(k, v)
    # exit(12)
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


def plot_miou(dict_mious, xticks=None, ls="-", color="tab:blue", label=None, marker=None, unit=10, ms=1.0):
    if xticks is None:
        xticks = [unit * (i + 1) for i in range(len(dict_mious.keys()))]

    yticks = [avg[0] for avg in dict_mious.values()]
    yerr = [avg[1] for avg in dict_mious.values()]

    # plot an errorbar
    # plt.errorbar(xticks, yticks, yerr, capsize=0, color=color, ls=ls, label=label)
    plt.plot(xticks, yticks, color=color, ls=ls, label=label, marker=marker, ms=ms)

    # optional: fill area between error ranges.
    data = {
        'x': xticks,
        "y1": [yticks[i] - yerr[i] for i in range(len(yticks))],
        "y2": [yticks[i] + yerr[i] for i in range(len(yticks))]
    }
    plt.fill_between(**data, alpha=0.15, color=color)
    plt.ylabel("mIoU")
    plt.legend(loc="lower right", fancybox=False, framealpha=1., edgecolor='black')


def plot_model(model, keywords, **kwargs):
    list_files = get_files(f"{model}", fname="log_val.txt")
    dict_files = sort_by(list_files, list_keywords=keywords)

    dict_mious = compute_avg_miou(dict_files)

    plot_miou(dict_mious, **kwargs)


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
            print(list_mious)
            best_miou = np.max(list_mious)
            dict_model_values[model_name].append(best_miou)
    print("fully sup:", np.mean(dict_model_values[model_name]))
    plt.hlines(np.mean(dict_model_values[model_name]), xmin=xmin, xmax=xmax, color=c, ls=ls, label=model_name, alpha=alpha)


if __name__ == '__main__':
    # plt.plot(figsize=(7, 5))
    rcParams["font.family"] = "serif"
    rcParams["grid.linestyle"] = ':'
    rcParams["font.size"] = 16

    DATASET = "voc"

    if DATASET == "cv":
        title = "CamVid"
        unit = 10
        yrange = (0., 0.7)
        draw_hline(f"cv/FPN50_full", "ResNet50 (100% annot.)", xmin=0, xmax=100, alpha=1.0, c='k', ls="--")

        # plot_model(f"{DATASET}/FPN18", keywords=[f"random_{i}_" for i in range(10)],
        #            color="r", unit=unit, marker='o')
        # plot_model(f"{DATASET}/FPN34", keywords=[f"random_{i}_" for i in range(10)], color="y", unit=unit, marker='^')
        # plot_model(f"{DATASET}/FPN50", keywords=[f"random_{i}_" for i in range(10)], color="g", unit=unit, marker='s')
        plot_model(f"{DATASET}/FPN18",
                   xticks=[*range(1, 10), *range(10, 110, 10)],
                   keywords=[re.compile(r"random_1_\d_{:d}_query".format(i)) for i in range(9)] + [
                       re.compile(r"random_10_\d_{:d}_query".format(i)) for i in range(10)],
                   label="ResNet18",
                   color="r", unit=unit)

        plot_model(f"{DATASET}/FPN34",
                   xticks=[*range(1, 10), *range(10, 110, 10)],
                   keywords=[re.compile(r"random_1_\d_{:d}_query".format(i)) for i in range(9)] + [
                       re.compile(r"random_10_\d_{:d}_query".format(i)) for i in range(10)],
                   label="ResNet34",
                   color="y", unit=unit)

        plot_model(f"{DATASET}/FPN50",
                   xticks=[*range(1, 10), *range(10, 110, 10)],
                   keywords=[re.compile(r"random_1_\d_{:d}_query".format(i)) for i in range(9)] + [
                       re.compile(r"random_10_\d_{:d}_query".format(i)) for i in range(10)],
                   label="ResNet50",
                   color="g", unit=unit)

        plot_model(f"{DATASET}/FPN101",
                   xticks=[*range(1, 10), *range(10, 110, 10)],
                   keywords=[re.compile(r"random_1_\d_{:d}_query".format(i)) for i in range(9)] + [re.compile(r"random_10_\d_{:d}_query".format(i)) for i in range(10)], # [f"_{i}_query" for i in range(10)], #[f"random_{i}_" for i in range(1, 11)],
                   label="ResNet101",
                   color="b", unit=unit)

        gca = plt.gca().tick_params(which='both', direction="in")

    elif DATASET == "cs_d4":
        title = "Cityscapes"
        unit = 1
        yrange = ()

        # plot_model(f"{DATASET}/FPN18", color="r", unit=unit, marker='o')
        # plot_model(f"{DATASET}/FPN34", color="y", unit=unit, marker='^')
        # plot_model(f"{DATASET}/FPN34", color="y", unit=unit, marker='^')

        plot_model(f"{DATASET}/FPN18",
                   xticks=[*range(1, 11)] + [100, 1000, 10000],
                   keywords=[re.compile(r"random_1_\d_{:d}_query".format(i)) for i in range(10)] + [
                       re.compile(r"random_{:d}_\d_0_query".format(i)) for i in [100, 1000, 10000]],
                   label="ResNet18",
                   color="r", unit=unit)

        plot_model(f"{DATASET}/FPN34",
                   xticks=[*range(1, 11)] + [100, 1000, 10000],
                   keywords=[re.compile(r"random_1_\d_{:d}_query".format(i)) for i in range(10)] + [
                       re.compile(r"random_{:d}_\d_0_query".format(i)) for i in [100, 1000, 10000]],
                   label="ResNet34",
                   color="y", unit=unit)

        plot_model(f"{DATASET}/FPN50",
                   xticks=[*range(1, 11)] + [100, 1000, 10000],
                   keywords=[re.compile(r"random_{:d}_\d_0_query".format(i)) for i in range(1, 11)] + [
                       re.compile(r"random_{:d}_\d_0_query".format(i)) for i in [100, 1000, 10000]],
                   label="ResNet50",
                   color="g", unit=unit)
        plt.xscale('log')

    elif DATASET == "voc":
        title = "PASCAL VOC 2012"
        unit = 1
        yrange = (0, 0.75)

        # plot_model(f"{DATASET}/FPN18", keywords=[f"random_{i}_" for i in range(10)],
        #            color="r", unit=unit, marker='o')
        # plot_model(f"{DATASET}/FPN34", keywords=[f"random_{i}_" for i in range(10)],
        #            color="y", unit=unit, marker='^')
        draw_hline(f"voc/FPN50_full", "ResNet50 (100% annot.)", xmin=1, xmax=1000, alpha=1.0, c='k', ls="--")

        plot_model(f"{DATASET}/FPN18",
                   keywords=[f"random_{i}_" for i in [*range(1, 11), 100, 1000]], #[f"random_1_2_{i}" for i in range(10)] + [f"random_{i}_2_0_query" for i in [100, 1000]],
                   xticks=[i for i in [*range(1, 11), 100, 1000]],
                   label="ResNet18",
                   color="r", unit=unit)

        plot_model(f"{DATASET}/FPN34",
                   keywords=[f"random_{i}_" for i in [*range(1, 11), 100, 1000]], # [f"random_1_2_{i}" for i in range(10)] + [f"random_{i}_2_0_query" for i in [100, 1000]],
                   xticks=[i for i in [*range(1, 11), 100, 1000]],
                   label="ResNet34",
                   color="y", unit=unit)

        plot_model(f"{DATASET}/FPN50",
                   keywords=[f"random_{i}_" for i in [*range(1, 11), 100, 1000]],  # + [f"random_1_2_{i}" for i in range(10)]
                   xticks=[i for i in [*range(1, 11), 100, 1000]],
                   label="ResNet50",
                   color="g", unit=unit)

        plot_model(f"{DATASET}/FPN101",
                   keywords=[f"random_{i}_" for i in [*range(1, 11), 100, 1000]],
                   # + [f"random_1_2_{i}" for i in range(10)]
                   xticks=[i for i in [*range(1, 11), 100, 1000]],
                   label="ResNet101",
                   color="b", unit=unit)

        plt.xscale("log")
        gca = plt.gca().tick_params(which='both', direction="in")

    # plt.title(f"{title}")
    plt.xlabel("# labelled pixels per image")

    plt.grid()
    plt.ylim(*yrange)

    plt.tight_layout(pad=0.1)
    plt.savefig(f"depth_experim_{title}.pdf")
    plt.show()

