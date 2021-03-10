import os
from csv import reader
from glob import glob
from xml.dom import minidom
import xml.etree.ElementTree as ET
from tqdm import tqdm
import pickle


def get_scribble_summary():
    DIR_VOC = "/scratch/shared/beegfs/gyungin/datasets/VOC2012/datasets/pascal/VOCdevkit_2012/VOC2012"
    DIR_SCRIBBLES = f"{DIR_VOC}/ScribbleAnnotation/pascal_2012"
    LIST_TRAIN = f"{DIR_VOC}/ImageSets/Segmentation/train.txt"
    list_train = []
    with open(LIST_TRAIN, 'r') as f:
        csv_reader = reader(f)
        for i, line in enumerate(csv_reader):
            list_train.append(line[0])
        f.close()
    print("# train images:", len(list_train))

    scribbles_xml = sorted(glob(f"{DIR_SCRIBBLES}/*.xml"))
    tree = ET.parse(scribbles_xml[0])
    root = tree.getroot()

    list_data = []
    for xml_file in tqdm(scribbles_xml):
        root = ET.parse(xml_file).getroot()
        fname = root.findall("filename")[0].text
        if not fname.split('.')[0] in list_train:
            continue

        height, width = root.findall("*/height")[0].text, root.findall("*/width")[0].text

        x, y = [], []
        for elem in root:
            if elem.tag == "size":
                for subelem in elem:
                    if subelem.tag == "height":
                        height = int(subelem.text)

                    elif subelem.tag == "width":
                        width = int(subelem.text)

            elif elem.tag == "polygon":
                for subelem in elem:
                    if subelem.tag == "point":
                        loc_x, loc_y = int(subelem.findall("X")[0].text), int(subelem.findall("Y")[0].text)
                        if loc_x < width and loc_y < height:
                            x.append(loc_x)
                            y.append(loc_y)
                        else:
                            continue

        assert len(x) == len(y), f"{len(x)} != {len(y)}"
        dict_data = {"filename": fname, "h": height, "w": width, "points": list(zip(y, x))}
        list_data.append(dict_data)
    print("# scribble (train):", len(list_data))
    pickle.dump(list_data, open("scribble_annot.pkl", "wb"))
    return list_data


if __name__ == '__main__':
    DST = "scribbles"
    os.makedirs(DST, exist_ok=True)
    fp = "scribble_annot.pkl"
    if os.path.exists(fp):
        data = pickle.load(open(fp, "rb"))

    else:
        data = get_scribble_summary()

    import numpy as np
    for datum in tqdm(data):
        fname, h, w, points = datum["filename"], datum["h"], datum["w"], datum["points"]
        grid = np.zeros((h, w), dtype=np.bool)
        for p in set(points):  # there are redundant points...
            try:
                grid[p] = True
            except IndexError:
                raise IndexError(fname, h, w, p)
        np.save(f"{DST}/{fname.split('.')[0]}.npy", grid)
