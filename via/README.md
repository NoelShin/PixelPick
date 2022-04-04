# PixelPick in VGG Image Annotator (VIA)
We provide the codebase for annotating images using a popular VIA user interface (UI).
Please note that this is not officially integrated into VIA, rather this is to offer a better UI for the interested PixelPick users.

To implement the VIA, please follow the steps below:

1. Move to the `via` directory.
```shell
cd via
```
2. Make a symbolic link to your dataset directory.
```shell
ln -s {YOUR_DATASET_DIRECTORY} datasets
```
It is important to keep your dataset in the following directory structure with each leaf directory containing relevant images:
```
{YOUR_DATASET_DIRECTORY}
├──{YOUR_DATASET_1}
│  ├──train
│  ├──test
│  ├──testannot
├──{YOUR_DATASET_2}
│  ├──train
│  ├──test
│  ├──testannot
...
```
It's worth noting that we assume you have ground-truth labels for test split, but not train split.

3. Configure your annotation setting in the `datasets/configs/custom.yaml`
```yaml
dir_root: "{YOUR_PROJECT_DIR}/PixelPick"  # project directory path

# dataset
dataset_name: "{YOUR_DATASET_NAME}"   # change the name depending on your dataset
dir_dataset: "{YOUR_PROJECT_DIR}/PixelPick/via/datasets/{YOUR_DATASET_NAME}"
ignore_index: 255 # set this for a label value that you would like to ignore
img_ext: "png"  # set this to the extension of images in the dir_dataset directory
n_classes: 11   # set this to the number of classes you are considering

# vgg image annotator (VIA) configuration
# mapping defines the key setting that you will use during annotation
mapping: {
  K: "sKy",
  B: "Building",
  P: "Pole",
  R: "Road",
  V: "paVement",
  T: "Tree",
  S: "Sign symbol",
  F: "Fence",
  C: "Car",
  D: "peDestrian",
  I: "bIcyclist",
}

# k_to_category_id decides how to map the key character to a label value
k_to_category_id: {
  K: 0,
  B: 1,
  P: 2,
  R: 3,
  V: 4,
  T: 5,
  S: 6,
  F: 7,
  C: 8,
  D: 9,
  I: 10
}

mean: [0.41189489566336, 0.4251328133025, 0.4326707089857]  # change this if you want to normalise images with a different mean
std: [0.27413549931506, 0.28506257482912, 0.28284674400252]  # change this if you want to normalise images with a different standard deviation

# augmentation (training time only)
crop_size: [360, 480]  # [H, W], this will be an actual input size of an image

# optimization
batch_size: 4  # a batch size used for training
n_epochs: 50
optimizer_type: "Adam"
optimizer_params: {
  lr: 0.0005,
  betas: [0.9, 0.999],
  weight_decay: 0.0002,
  eps: 0.0000007
}
lr_scheduler_type: "MultiStepLR"  # you can choose between ["Poly", "MultiStepLR"]
```
Please notice that the `mapping` argument contains the information about what keyboard character you will press to label
a queried pixel with some category. In the above example, if you press a character 'K', it means you will label the pixel
with the `sky` class, which will again be mapped to a label value `0` according to the argument `k_to_category_id`.

4. Generate initial queries for your dataset.
Please open the `scripts/query.sh` file which would look like:
```shell
#!/usr/bin/bash
python3 ../query.py \
--p_dataset_config "{YOUR_PROJECT_DIR}/PixelPick/datasets/configs/custom.yaml" \
--n_pixels_by_us 5 # the number of queried pixels per image

# if you have a pre-trained model, you can set
# --p_state_dict "../checkpoints/cv_deeplab_margin_sampling_5_p0.05_0/0_query/best_model.pt"
# -qs "margin_sampling"
```
You need to set the path for your configuration file and also the number of pixels to be queried per image, which is set to 5 in the above example.
The commented part at the bottom is for the case you already trained at least once with (initial) queries and you have pre-trained model to get queries from.

Once you change the `scripts/query.sh` file as you want, run the file:
```shell
cd ../scripts
bash queries.sh
```

This will generate queries for the images of your dataset which will be saved in `checkpoints` directory (automatically created under your project directory).

5. Launch VIA using the saved query file.
Please open the `via/launch-via.sh` file which would look like:
```shell
#!/usr/bin/bash
python3 launch_via.py \
--p_dataset_config "{YOUR_PROJECT_DIR}/PixelPick/datasets/configs/camvid.yaml" \
--p_queries "{YOUR_PROJECT_DIR}/PixelPick/checkpoints/{YOUR_MODEL_NAME}/0_query/queries.pkl"
```

Once you change accordingly to your case, run the file:
```shell
cd ../via
bash launch-via.sh
```

This will launch the VIA and you can start annotating with it.

**Don't forget to save the annotation file once you finish doing it (click a disc icon on the toolbar)**.

6. Convert the annotated labels into a format compatible with the training code.
Once you finish annotation, it'd be saved with a `json` extension, which is not compatible with our training code. To make it compatible,
you need to use the `via/convert-json-to-pkl.sh` file which would look like (we assume you moved the saved annotation file into `via` directory):
```shell
#!/usr/bin/bash
python3 convert_json_to_pkl.py -v \
-vaf "{YOUR_PROJECT_DIR}/PixelPick/via/{YOUR_ANNOTATION_FILE}.json" \
--p_dataset_config "{YOUR_PROJECT_DIR}/PixelPick/datasets/configs/custom.yaml"
```

Once you change accordingly to your case, run the file:
```shell
bash convert-json-to-pkl.sh
```

This will generate an annotation file with the same name but in a `pkl` extension.
Please change this file name to `queries.pkl` and put this to `checkpoints/{YOUR_MODEL_NAME}/0_query`.
You will need to overwrite the `queries.pkl` file which already have been in the directory.

7. Train your model based on the annotated queries.
Please open the file `scripts/train-a-round.sh` which would look like:
```shell
#!/usr/bin/bash
python3 ../train.py \
--n_pixels_by_us 5 \
--dir_checkpoints "{YOUR_PROJECT_DIR}/checkpoints/{YOUR_MODEL_NAME}" \
--p_dataset_config "{YOUR_PROJECT_DIR}/PixelPick/datasets/configs/custom.yaml"
```

Once you change accordingly to your case, run the file:
```shell
cd ../scripts
bash train-a-round.sh
```

Once the training is done, you can repeat from the query generation process (Step 4) to training process (Step 7).
