import os
from glob import glob
from shutil import copy, move
if __name__ == '__main__':
    list_logs = sorted(glob("cv/deeplab_v3/deal/**/log_val.txt", recursive=True))
    for log in list_logs:
        name_dir_pp, name_dir_p = os.path.join(*log.split('/')[:-2]), log.split('/')[-2]
        if "runs_" in name_dir_p:
            loc = name_dir_pp.find("pam_")
            dst = name_dir_pp[:loc + 5].replace("deal_", '') + '_' + name_dir_p

            os.makedirs(dst, exist_ok=True)
            move(log, dst)
