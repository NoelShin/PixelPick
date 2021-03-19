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
              fill_between=True, scatter=False):
    if xticks is None:
        xticks = [unit * (i + 1) for i in range(len(dict_mious.keys()))]
    yticks = [avg[0] for avg in dict_mious.values()]
    yerr = [avg[1] for avg in dict_mious.values()]
    print(label, yticks)

    # plot an errorbar
    # plt.errorbar(xticks, yticks, yerr, capsize=0, color=color, ls=ls, label=label)
    if scatter:
        plt.scatter(xticks, yticks, color=color, label=label, marker=marker, s=ms)

    else:
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
            print(list_mious)
            best_miou = np.max(list_mious)
            dict_model_values[model_name].append(best_miou)
    print("fully sup:", np.mean(dict_model_values[model_name]))
    plt.hlines(np.mean(dict_model_values[model_name]), xmin=xmin, xmax=xmax, color=c, ls=ls, label=model_name, alpha=alpha)


if __name__ == '__main__':
    # plt.plot(figsize=(7, 5))
    rcParams["font.family"] = "serif"
    rcParams["grid.linestyle"] = ':'
    rcParams["font.size"] = 12
    xlabel_size = 16
    rcParams["axes.labelsize"] = 16
    rcParams["xtick.labelsize"] = 16
    rcParams["ytick.labelsize"] = 16

    DATASET = "voc"
    mode = 2
    if mode == 0:
        title = "sota_CamVid"
        unit = 10
        yrange = ()
        yrange = (0.0, 0.70)

        plt.figure(figsize=(7, 5))

        plot_model(f"cv/deeplab_v3/ms",
                   label="PixelPick (Ours, MobileNetv2)",
                   # label="PixelPick (DeepLabv3+)",
                   xticks=[i / (360 * 480) * 100 for i in range(10, 110, 10)],
                   color="cyan", unit=unit, ls='--', marker='o', ms=3.5)

        plot_model(f"cv/fpn50/ms",
                   label="PixelPick (Ours, ResNet50)",
                   # label="PixelPick (FPN50)",
                   xticks=[i / (360 * 480) * 100 for i in range(10, 110, 10)],
                   color="blue", ls='-', unit=unit, marker='o', ms=3.5)

        # DEAL original
        plt.plot([10, 15, 20, 25, 30, 35, 40], [0.516, 0.557, 0.576, 0.587, 0.6, 0.613, 0.616],
                 marker='o', ls='-', label="DEAL (MobileNetv2)", color='r', ms=3.5)

        # VAAL from DEAL paper
        plt.plot([10, 15, 20, 25, 30, 35, 40], [0.514, 0.547, 0.569, 0.582, 0.597, 0.607, 0.612],
                 marker='o', ls='-', label="VAAL (MobileNetv2)", color='darkviolet', ms=3.5)

        # Core-set from DEAL paper
        plt.plot([10, 15, 20, 25, 30, 35, 40], [0.518, 0.546, 0.566, 0.576, 0.590, 0.598, 0.608],
                 marker='o', ls='-', label="Core-set (MobileNetv2)", color='y', ms=3.5)

        draw_hline(f"cv/fpn50/full", "Fully-sup (100% annot., ResNet50)", xmin=5e-3, xmax=40, alpha=1.0, c='k', ls="--")
        # 5e-3
        plt.plot([8, 12, 16, 20, 24], [0.567, 0.6182, 0.6274, 0.6341, 0.6385],
                 marker='o', ls='--', label="EquAL (ResNet50)", color='g', ms=3.5)
        plt.plot([8, 12, 16, 20, 24], [0.6213, 0.6446, 0.6492, 0.656, 0.663], marker='o', label="EquAL+ (ResNet50)", color='darkgreen', ms=3.5)

        plt.plot([50 / 367 * 100, 100 / 367 * 100], [0.537, 0.557], label="CCT (ResNet50)", marker='o', color="darkred", ms=3.5)

        gca = plt.gca()
        gca.tick_params(direction='in', which='both')
        # gca.ticklabel_format(axis='x', style='plain')
        plt.legend(loc="lower right", fancybox=False, framealpha=1., edgecolor='black')
        # plt.legend(loc='lower right', fancybox=False, framealpha=1., edgecolor='black')
        # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fancybox=False, framealpha=1., edgecolor='black')
        # plt.xlim(-2, 25)
        plt.xscale("log")
        plt.xticks([1e-2, 1e-1, 1, 10], ["0.01", "0.1", "1", "10"])
        plt.xlabel("% annotation (log-scale)")

    elif mode == 1:
        title = "deeplab_comp_cv"
        unit = 10
        yrange = (0.3, 0.59)
        # figsize=(7, 5)
        plt.figure(figsize=(33 * 0.2, 19 * 0.2))
        rcParams["font.size"] = 16

        xticks = [i for i in range(10, 110, 10)]
        ms = 4
        fill_between = True
        plot_model(f"cv/deeplab_v3/rand",
                   label="RAND",
                   xticks=xticks,
                   color="gray", unit=unit, ls='-', marker='s', ms=ms, fill_between=fill_between)

        plot_model(f"cv/deeplab_v3/ms",
                   label="MS",
                   xticks=xticks,
                   color="blue", unit=unit, ls='-', marker='o', ms=ms, fill_between=fill_between)

        plot_model(f"cv/deeplab_v3/ent",
                   label="ENT",
                   xticks=xticks,
                   color="red", unit=unit, marker='^', ms=ms, fill_between=fill_between)

        plot_model(f"cv/deeplab_v3/lc",
                   label="LC",
                   xticks=xticks,
                   color="green", unit=unit, marker='v', ms=ms, fill_between=fill_between)

        plot_model(f"cv/deeplab_v3/ms_vote",
                   label="vote MS",
                   xticks=xticks,
                   color="tab:blue", unit=unit, ls='--', marker='o', ms=ms, fill_between=fill_between)

        plot_model(f"cv/deeplab_v3/ent_vote",
                   label="vote ENT",
                   xticks=xticks,
                   color="tab:red", unit=unit, ls='--', marker='^', ms=ms, fill_between=fill_between)

        plot_model(f"cv/deeplab_v3/lc_vote",
                   label="vote LC",
                   xticks=xticks,
                   color="tab:green", unit=unit, ls='--', marker='v', ms=ms, fill_between=fill_between)

        plt.xticks([i for i in range(10, 110, 10)])

        gca = plt.gca()
        gca.tick_params(direction='in', which='both')
        # plt.legend(loc="lower left", fancybox=False, framealpha=1., edgecolor='black')
        # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fancybox=False, framealpha=1., edgecolor='black')
        draw_hline(f"cv/deeplab_v3/full", model_name="Fully-sup", xmin=5, xmax=105, alpha=1.0, c='k', ls="--")
        plt.legend(loc='lower right', fancybox=False, framealpha=1., edgecolor='black', fontsize=12)
        plt.yticks(np.arange(0.3, 0.60, 0.05))  #np.arange(0.42, 0.60, 0.02))
        plt.xlabel("# labelled pixels per image")

    elif mode == 2:
        title = "sota_cs"

        unit = 1
        yrange = (0, 0.7)

        plt.figure(figsize=(6, 5.2))
        draw_hline(f"cs_d4/fpn50/full", "Fully-sup (100% annot., ResNet50)", xmin=5e-3, xmax=40, alpha=1.0, c='k', ls="--")

        plot_model(f"cs_d4/deeplab_v3/ms_10",
                   label="PixelPick (Ours, MobileNetv2)",
                   xticks=[i * 10 / (256 * 512) * 100 for i in range(1, 11, 1)],
                   color="cyan", unit=unit, ls='--', marker='o', ms=3.5)

        plot_model(f"cs_d4/fpn50/ms_10",
                   label="PixelPick (Ours, ResNet50)",
                   xticks=[i * 10 / (256 * 512) * 100 for i in range(1, 11, 1)],
                   color="blue", ls='-', unit=unit, marker='o', ms=3.5)

        plt.plot([10, 15, 20, 25, 30, 35, 40], [0.509, 0.552, 0.572, 0.588, 0.6, 0.613, 0.620],
                 marker='o', ls='--', label="DEAL (MobileNetv2)", color='r', ms=3.5)

        # VAAL from DEAL paper
        plt.plot([10, 15, 20, 25, 30, 35, 40], [0.509, 0.542, 0.559, 0.579, 0.586, 0.602, 0.609],
                 marker='o', ls='-', label="VAAL (MobileNetv2)", color='darkviolet', ms=3.5)

        # Core-set from DEAL paper
        plt.plot([10, 15, 20, 25, 30, 35, 40], [0.505, 0.535, 0.555, 0.573, 0.584, 0.593, 0.607],
                 marker='o', ls='-', label="Core-set (MobileNetv2)", color='y', ms=3.5)

        plt.plot([3.356, 6.644, 9.932, 11.575, 13.185, 14.760, 16.370, 17.877, 19.418],
                 [0.492, 0.534, 0.559, 0.583, 0.583, 0.590, 0.592, 0.594, 0.599],
                 marker='o', ls='--', label="CEREAL (ResNet50)", color='gray', ms=3.5)

        plt.plot([1, 2, 3, 6, 9, 12, 15], [0.4242, 0.4880, 0.5184, 0.5713, 0.5890, 0.6065, 0.6185],
                 marker='o', ls='--', label="EquAL (ResNet50)", color='green', ms=3.5)
        plt.plot([1, 2, 3, 6, 9, 12, 15], [0.4885, 0.5590, 0.5890, 0.6224, 0.6352, 0.6446, 0.6562],
                 marker='o', label="EquAL+ (ResNet50)", color='green', ms=3.5)

        plt.plot([50 / 2975 * 100, 100 / 2975 * 100], [0.35, 0.401], label="CCT (ResNet50)", marker='o', color="red", ms=3.5)
        gca = plt.gca()
        gca.tick_params(direction='in', which='both')

        plt.ylim(-0.2, 0.7)
        plt.yticks(np.arange(0., 0.7, 0.1))
        plt.legend(loc='lower left', fancybox=False, framealpha=1., edgecolor='black', fontsize=10.8)
        plt.xscale("log")
        plt.xlabel("% annotation (log-scale)")
        plt.xlim()
        plt.xticks([1e-2, 1e-1, 1, 10], ["0.01", "0.1", "1", "10"])

    elif mode == 3:
        title = "sota_PASCAL VOC 2012"
        unit = 5
        yrange = (0., 0.73)
        draw_hline(f"voc/fpn50/full", "Fully-sup (100% annot., ResNet50)", xmin=5e-3, xmax=100, alpha=1.0, c='k', ls="--")

        plot_model(f"voc/deeplab_v3/ms_5",
                   xticks=[i * 1464 / 169787200 * 100 for i in range(5, 55, 5)],
                   color="cyan", unit=unit, marker='o', ms=3.5,
                   label="PixelPick (Ours, MobileNetv2)")

        plot_model(f"voc/fpn50/ms_5",
                   xticks=[i * 1464 / 169787200 * 100 for i in range(5, 55, 5)],
                   color="darkblue", unit=unit, marker='o', ms=3.5,
                   label="PixelPick (Ours, ResNet50)")

        plt.scatter(786982 / 262462800 * 100, 0.631, label=r"ScribbleSup (VGG16)",  # $^\dag$
                    marker='^', color="darkred", s=24)

        # voc total pixels used in CCT: 169787200
        plt.plot([(2 / 3) * 100, 100], [0.64, 0.694], label=r"CCT (ResNet50)", marker='s', color="red", ms=3.5)  # $^\ddag$
        # plt.scatter(100, 0.732, label=r"CCT+ (ResNet50)$^\ddag$", marker='s', color="darkred", s=24)
        plt.scatter(100, 0.646, label=r"WSSL (VGG16)", marker='*', s=24)  # $^\ddag$
        plt.scatter(100, 0.605, label=r"GAIN (VGG16)", marker='p',  s=24)  # $^\ddag$
        plt.scatter(100, 0.657, label=r"MDC (VGG16)", marker='+',  s=24)  # $^\ddag$
        plt.scatter(100, 0.643, label=r"DSRG (VGG16)", marker='v',  s=24)  # $^\ddag$
        plt.scatter(100, 0.658, label=r"FickleNet (VGG16)", marker='h', s=24)  # $^\ddag$

        plt.legend(fancybox=False, framealpha=1.0, edgecolor='k', fontsize=11)
        # plot_model(f"{DATASET}/ScribbleSup/",
        #            xticks=[786982 / 1464],
        #            color="purple",
        #            unit=unit,
        #            scatter=True,
        #            marker='^', ms=20, label="Scribble (ResNet50)")
        gca = plt.gca()
        gca.tick_params(direction='in', which='both')
        plt.xscale("log")
        plt.xlabel("% annotation (log-scale)")
        plt.xticks([1e-2, 1e-1, 1, 10, 100], ["0.01", "0.1", "1", "10", "100"])

    elif mode == 4:
        title = "deeplab_comp"
        unit = 10
        yrange = ()
        # yrange = (0.35, 0.65)
        plt.figure(figsize=(7, 5))

        plot_model(f"{DATASET}/deeplab_v3/rand",
                   label="RAND",
                   fill_between=True,
                   xticks=[i / (360 * 480) * 100 for i in range(10, 110, 10)],
                   color="gray", unit=unit, marker='o')

        plot_model(f"{DATASET}/deeplab_v3/ms",
                   label="MS",
                   fill_between=True,
                   xticks=[i / (360 * 480) * 100 for i in range(10, 110, 10)],
                   color="b", unit=unit, ls='-', marker='o', ms=3.5)

        plot_model(f"{DATASET}/deeplab_v3/ent",
                   label="ENT",
                   fill_between=True,
                   xticks=[i / (360 * 480) * 100 for i in range(10, 110, 10)],
                   color="r", unit=unit, ls='-', marker='o', ms=3.5)

        plot_model(f"{DATASET}/deeplab_v3/lc",
                   label="LC",
                   fill_between=True,
                   xticks=[i / (360 * 480) * 100 for i in range(10, 110, 10)],
                   color="g", unit=unit, ls='-', marker='o', ms=3.5)

        plot_model(f"{DATASET}/deeplab_v3/ms_vote",
                   label="vote MS",
                   fill_between=True,
                   xticks=[i / (360 * 480) * 100 for i in range(10, 110, 10)],
                   color="b", unit=unit, ls='--', marker='o', ms=3.5)

        plot_model(f"{DATASET}/deeplab_v3/ent_vote",
                   label="vote ENT",
                   fill_between=True,
                   xticks=[i / (360 * 480) * 100 for i in range(10, 110, 10)],
                   color="r", unit=unit, ls='--', marker='o', ms=3.5)

        plot_model(f"{DATASET}/deeplab_v3/lc_vote",
                   label="vote LC",
                   fill_between=True,
                   xticks=[i / (360 * 480) * 100 for i in range(10, 110, 10)],
                   color="g", unit=unit, ls='--', marker='o', ms=3.5)

        draw_hline(f"{DATASET}/deeplab_v3/full", "Fully-sup", xmin=5e-3, xmax=0.06, alpha=1.0, c='k', ls="--")

        gca = plt.gca()
        gca.tick_params(direction='in', which='both')
        plt.legend(loc="lower right", fancybox=False, framealpha=1., edgecolor='black')
        # plt.xscale("log")
        plt.xlabel("% annotation")

    elif mode == 5:
        title = "voc_deeplab_comp"
        unit = 1
        yrange = ()
        # yrange = (0.35, 0.65)
        plt.figure(figsize=(7, 5))

        plot_model(f"voc/deeplab_v3/rand",
                   label="RAND",
                   fill_between=False,
                   xticks=[i * 1464 / 169787200 * 100 for i in range(1, 11, 1)],
                   color="gray", unit=unit, marker='o')

        plot_model(f"voc/deeplab_v3/ms",
                   label="MS",
                   fill_between=False,
                   xticks=[i * 1464 / 169787200 * 100 for i in range(1, 11, 1)],
                   color="b", unit=unit, ls='-', marker='o', ms=3.5)

        plot_model(f"voc/deeplab_v3/ent",
                   label="ENT",
                   fill_between=False,
                   xticks=[i * 1464 / 169787200 * 100 for i in range(1, 11, 1)],
                   color="r", unit=unit, ls='-', marker='o', ms=3.5)

        plot_model(f"voc/deeplab_v3/lc",
                   label="LC",
                   fill_between=False,
                   xticks=[i * 1464 / 169787200 * 100 for i in range(1, 11, 1)],
                   color="g", unit=unit, ls='-', marker='o', ms=3.5)

        draw_hline(f"voc/deeplab_v3/full", "Fully-sup", xmin=5e-4, xmax=0.01, alpha=1.0, c='k', ls="--")

        gca = plt.gca()
        gca.tick_params(direction='in', which='both')
        gca.ticklabel_format(axis='x', style="sci", scilimits=(0, 0))
        plt.legend(loc="lower right", fancybox=False, framealpha=1., edgecolor='black')
        # plt.xscale("log")
        plt.xlabel("% annotation")

    elif mode == 6:
        title = "cs_deeplab_comp"
        unit = 1
        yrange = ()
        # yrange = (0.35, 0.65)
        plt.figure(figsize=(7, 5))

        plot_model(f"cs_d4/deeplab_v3/rand",
                   label="RAND",
                   fill_between=False,
                   xticks=[i * 1464 / 169787200 * 100 for i in range(1, 11, 1)],
                   color="gray", unit=unit, marker='o')

        plot_model(f"cs_d4/deeplab_v3/ms",
                   label="MS",
                   fill_between=False,
                   xticks=[i * 1464 / 169787200 * 100 for i in range(1, 11, 1)],
                   color="b", unit=unit, ls='-', marker='o', ms=3.5)

        plot_model(f"cs_d4/deeplab_v3/ent",
                   label="ENT",
                   fill_between=False,
                   xticks=[i * 1464 / 169787200 * 100 for i in range(1, 11, 1)],
                   color="r", unit=unit, ls='-', marker='o', ms=3.5)

        plot_model(f"cs_d4/deeplab_v3/lc",
                   label="LC",
                   fill_between=False,
                   xticks=[i * 1464 / 169787200 * 100 for i in range(1, 11, 1)],
                   color="g", unit=unit, ls='-', marker='o', ms=3.5)

        # draw_hline(f"{DATASET}/deeplab_v3/full", "Fully-sup", xmin=5e-3, xmax=0.06, alpha=1.0, c='k', ls="-.")

        gca = plt.gca()
        gca.tick_params(direction='in', which='both')
        gca.ticklabel_format(axis='x', style="sci", scilimits=(0, 0))
        plt.legend(loc="lower right", fancybox=False, framealpha=1., edgecolor='black')
        # plt.xscale("log")
        plt.xlabel("% annotation")

    plt.grid()
    plt.ylim(*yrange)

    plt.tight_layout(pad=0.1)
    plt.savefig(f"{title}.pdf")
    plt.show()

