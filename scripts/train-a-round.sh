#!/usr/bin/bash
python3 ../train.py \
--n_pixels_by_us 5 \
--dir_checkpoints "../checkpoints/camvid_deeplab_margin_sampling_5_p0.05_0" \
--p_dataset_config "/Users/noel/projects/PixelPick/datasets/configs/custom.yaml"