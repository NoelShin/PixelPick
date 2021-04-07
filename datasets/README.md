# Dataset description
For ease of future reimplmentation, we detail the information on each dataset in this directory, e.g. which categories/images are considered for training/validation. For the image file names considered for training/validation, please refer to the `{camvid, cityscapes, voc}_{train, val}.txt` file.

### CamVid
We use 11 classes version of this dataset (it is worth noting that it originally contains 32 categories), following the common practice for segmentation. The considered labels are
```Python
{
  0: "sky",
  1: "building",
  2: "pole",
  3: "road",
  4: "pavement",
  5: "tree",
  6: "sign symbol",
  7: "fence",
  8: "car",
  9: "pedestrian",
  10: "bicyclist",
  11: ignore_index
}
```

We use train/test splits which consists of 367 and 233 images for training and validation.

### Cityscapes
We use 19 classes as follows.
```Python
{
  0: "road",
  1: "sidewalk",
  2: "building",
  3: "wall",
  4: "fence",
  5: "pole",
  6: "traffic light",
  7: "traffic sign",
  8: "vegetation",
  9: "terrain",
  10: "sky",
  11: "person",
  12: "rider",
  13: "car",
  14: "truck",
  15: "bus",
  16: "train",
  17: "motorcycle",
  18: "bicycle",
  19: ignore_index
}
```
Note that the original labels provided by the official Cityscapes webpage are not organised in this way (e.g., "road" category has label ID of 7). Instead, we map their default label IDs to this during the data preprocessing step. Please refer to the function `_cityscapes_classes_to_labels` in `cityscapes.py` file for more details on this.

Training/validation is done on train/val splits which are composed of 2975 and 500 images each.

### PASCAL VOC 2012 segmentation
We use 20 classes as follows.
```Python
{
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
  11: "dining table",
  12: "dog",
  13: "horse",
  14: "motorbike",
  15: "person",
  16: "potted plant",
  17: "sheep",
  18: "sofa",
  19: "train",
  20: "tv/monitor",
  255: ignore_index
}
```
Training/validation is done on train/val splits which contain 1464 and 1449 images each.
