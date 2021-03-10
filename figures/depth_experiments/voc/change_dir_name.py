import os
from glob import glob
import re
from shutil import move, copy


def change_name(prev, new):
    return


if __name__ == '__main__':
    DIR_ROOT = "FPN50"
    list_files = sorted(glob(f"{DIR_ROOT}/**/*", recursive=True))

    for fname in list_files:
        if os.path.isdir(fname) or os.path.splitext(fname)[-1] == ".zip":
            continue
        else:
            dir_par = fname.split('/')[-2]
            if dir_par.find("_0_query") != -1:
                continue
            else:
                name_prev = dir_par
                nth_query = name_prev[re.search(r"\d_query$", name_prev).span()[0]]

                name_new = re.sub(r"_\d_query", "_0_query", name_prev)
                name_new = re.sub("random_1", f"random_{1 + int(nth_query)}", name_new)

        dir_new = f"{DIR_ROOT}/{name_new}"
        os.makedirs(dir_new, exist_ok=True)
        copy(fname, dir_new)
