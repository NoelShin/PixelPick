import io
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, DrawingArea, HPacker, VPacker

alphabet = ['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P', 'A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L', 'Z', 'X']
alphabet_cv = ['K', 'B', 'P', 'R', 'V', 'T', 'S', 'F', 'C', 'D', 'I']
alphabet_voc = ['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P', 'A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L', 'Z', 'X']

alphabet_l = [a.lower() for a in alphabet]
alphabet_l_cv, alphabet_l_voc = [a.lower() for a in alphabet_cv], [a.lower() for a in alphabet_voc]


cv_label_category = {
    0: "sKy",
    1: "Building",
    2: "Pole",
    3: "Road",
    4: "paVement",
    5: "Tree",
    6: "Sign symbol",
    7: "Fence",
    8: "Car",
    9: "peDestrian",
    10: "bIcyclist",
}

voc_label_category = {
    0: "background",
    1: "aeroplane",
    2: "bicycle",
    3: "bird",
    4: "boat",
    5: "bottle",
    6: "bus",
    7: "car",
    8: "cat",
    9: "chair",
    10: "cow",
    11: "diningtable",
    12: "dog",
    13: "horse",
    14: "motorbike",
    15: "person",
    16: "pottedplant",
    17: "sheep",
    18: "sofa",
    19: "train",
    20: "tvmonitor",
    # 255: "void"
}


def annotate(arr, dataset_name):
    global alphabet_l
    cv2.namedWindow('image')
    cv2.imshow('image', arr)
    n_classes = 11 if dataset_name == "camvid" else 21
    alphabet_l = alphabet_l_cv if dataset_name == "camvid" else alphabet_l_voc
    while True:
        k = cv2.waitKey()
        if chr(k) not in alphabet_l[:n_classes]:
            print(f"\nYou've clicked a wrong character: {chr(k)}\n")

        elif chr(k) in alphabet_l[:n_classes]:
            break

        elif k == 27:
            break

    cv2.destroyAllWindows()
    return chr(k)


def color_points(x, loc_y, loc_x, fc=(0, 128, 192), ec=(255, 255, 255), ms=5, es=2):
    x = cv2.circle(x, (loc_x, loc_y), ms + es, color=ec, thickness=-1)
    x = cv2.circle(x, (loc_x, loc_y), ms, color=fc, thickness=-1)
    return x


def make_gui(img, dataset_name, **kwargs):
    global alphabet
    fig = plt.figure(figsize=kwargs["figsize"])
    from matplotlib import rcParams
    rcParams['axes.linewidth'] = 2
    rcParams['axes.edgecolor'] = 'k'

    plt.imshow(img)

    label_category = cv_label_category if dataset_name == "camvid" else voc_label_category
    alphabet = alphabet_cv if dataset_name == "camvid" else alphabet_voc

    vpacker_children = [TextArea("{} - {}".format(alphabet[l], cat), textprops={"weight": 'bold', "size": 10})
                        for l, cat in sorted(label_category.items(), key=lambda x: x[1])]
    box = VPacker(children=vpacker_children, align="left", pad=5, sep=5)

    # display the texts on the right side of image
    anchored_box = AnchoredOffsetbox(loc="center left",
                                     child=box,
                                     pad=0.,
                                     frameon=True,
                                     bbox_to_anchor=(1.04, 0.5),
                                     bbox_transform=plt.gca().transAxes,
                                     borderpad=0.)
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
    anchored_box = AnchoredOffsetbox(loc="lower center",
                                     child=box,
                                     pad=0.,
                                     frameon=False,
                                     bbox_to_anchor=(0.5, -0.1),  # ( 0.5, -0.1)
                                     bbox_transform=plt.gca().transAxes,
                                     borderpad=0.)
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