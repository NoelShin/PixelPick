#!/usr/bin/bash
python3 ../query.py \
--p_dataset_config "/Users/noel/projects/PixelPick/datasets/configs/custom.yaml" \
--n_pixels_by_us 5 \
--p_state_dict "../checkpoints/camvid_deeplab_margin_sampling_5_p0.05_0/0_query/best_model.pt" \
-qs "margin_sampling"