from glob import glob
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    inputs = [Image.open(p) for p in sorted(glob("cct_examples/*.jpg"))]
    gts = [Image.open(p) for p in sorted(glob("cct_examples/*.png"))]
    sup_vis = [np.array(Image.open(p))[11:-11, 11:-11] for p in sorted(glob("prev_vis/resnet50*.jpg"))]
    mocov2_vis = [np.array(Image.open(p))[11:-11, 11:-11] for p in sorted(glob("prev_vis/moco_v2*.jpg"))]
    dcv2_vis = [np.array(Image.open(p))[11:-11, 11:-11] for p in sorted(glob("prev_vis/deepcluster*.jpg"))]
    swav_vis = [np.array(Image.open(p))[11:-11, 11:-11] for p in sorted(glob("prev_vis/swav*.jpg"))]

    # x = inputs[0]
    # m = mocov2_vis[0]
    # print(np.array(x).shape, np.array(m).shape)
    #
    # exit(12)
    labelpad = 15
    fig, ax = plt.subplots(nrows=len(inputs), ncols=6, figsize=(12, 6))
    for i, (x, gt, sup, moco, dc, swav) in enumerate(zip(inputs, gts, sup_vis, mocov2_vis, dcv2_vis, swav_vis)):
        print(x.size)

        ax[i, 0].imshow(x)
        ax[i, 1].imshow(gt)
        ax[i, 2].imshow(sup, cmap='gray')
        ax[i, 3].imshow(moco, cmap='gray')
        ax[i, 4].imshow(dc, cmap='gray')
        ax[i, 5].imshow(swav, cmap='gray')

        for j in range(6):
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])

        if i == len(inputs) - 1:
            ax[i, 0].set_xlabel("input", labelpad=labelpad)
            ax[i, 1].set_xlabel("ground-truth", labelpad=labelpad)
            ax[i, 2].set_xlabel("supervised", labelpad=labelpad)
            ax[i, 3].set_xlabel("MoCo v2", labelpad=labelpad)
            ax[i, 4].set_xlabel("DeepCluster v2", labelpad=labelpad)
            ax[i, 5].set_xlabel("SwAV", labelpad=labelpad)

    plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=0)
    plt.savefig("combined_imgs.png")
    plt.show()


