#!/home/gishin-temp/bin/sh
python3 train_stage.py  --suffix '' --dataset_name 'cs' --query_strategy 'margin_sampling' --network_name "FPN" --n_pixels_by_us 1 --seed 0 --downsample 2
