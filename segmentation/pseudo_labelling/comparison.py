import os
import re
from glob import glob
from csv import reader
import numpy as np
import matplotlib.pyplot as plt


def get_files(model_name):
    list_files = os.listdir('.')
    pattern = re.compile("[1-9]{1,2}" + f"_{model_name}" + r"\.txt")
    return [f for f in list_files if pattern.search(f) is not None]


def read(txt_file):
    dict_values = dict()
    with open(txt_file, 'r') as f:
        csv_reader = reader(f, delimiter=',')
        for i, line in enumerate(csv_reader):
            k = "mean" if line[0] == "mean" else int(line[0])
            try:
                dict_values.update({k: {
                    'prec': float(line[1]),
                    'recall': float(line[2]),
                    'f1': float(line[3]),
                    'prec_g': float(line[4]),
                    'recall_g': float(line[5]),
                    'f1_g': float(line[6])
                }})
            except IndexError:
                raise IndexError(dict_values, txt_file)
        f.close()
    return dict_values


def plot_model(model_name, c="tab:blue"):
    list_files = get_files(model_name)
    for i, f in enumerate(list_files):
        window_size = int(f.replace(f"_{model_name}.txt", ''))
        alpha = 0.4 + 0.6 * (window_size // 2) / 15
        d = read(f)

        plt.scatter(d['mean']['recall_g'], d['mean']['prec_g'],
                    marker='o', c=c, alpha=alpha, label=model_name if alpha == 1 else None)


if __name__ == '__main__':
    dict_m_c = {"baseline": "tab:blue", "local_sim": "tab:orange", "partition_local_sim": "tab:green"}
    for model_name in ["baseline", "partition_local_sim", "local_sim"]:
        plot_model(model_name, c=dict_m_c[model_name])

    plt.ylabel("precision")
    plt.xlabel("recall")
    plt.tight_layout()
    plt.legend()
    plt.show()
    plt.close()
