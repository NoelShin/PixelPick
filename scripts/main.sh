#!/usr/bin/sh
python3 ../main_al.py --dataset_name 'voc' --network_name 'deeplab' --n_pixels_by_us 10 --seed 0 -qs "margin_sampling"
python3 ../main_al.py --dataset_name 'cs' --network_name 'deeplab' --n_pixels_by_us 10 --seed 0 -qs "margin_sampling"
python3 ../main_al.py --dataset_name 'cv' --network_name 'deeplab' --n_pixels_by_us 10 --seed 0 -qs "margin_sampling"

