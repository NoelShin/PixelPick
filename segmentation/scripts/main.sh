#!/home/gishin-temp/bin/sh
python3 ../main_al.py  --suffix '' --dataset_name 'cv' --n_init_pixels 10 --n_pixels_by_us 10 --seed 0 --max_budget 100 --query_strategy "margin_sampling" --reverse_order
#python3 ../main_al.py  --suffix '' --dataset_name 'voc' --n_init_pixels 1 --n_pixels_by_us 10 --seed 3 --max_budget 30 --query_strategy "margin_sampling"
#python3 ../main_al.py  --suffix '' --dataset_name 'voc' --n_init_pixels 1 --n_pixels_by_us 10 --seed 4 --max_budget 30 --query_strategy "margin_sampling"
#python3 ../main_al.py  --suffix '' --dataset_name 'voc' --n_pixels_by_us 0 --network_name "deeplab" --seed 1
#python3 ../main_al.py  --suffix '' --dataset_name 'voc' --n_pixels_by_us 0 --network_name "deeplab" --seed 2
#python3 ../main_al.py  --suffix '' --dataset_name 'voc' --n_pixels_by_us 0 --network_name "deeplab" --seed 3
#python3 ../main_al.py  --suffix '' --dataset_name 'voc' --n_pixels_by_us 0 --network_name "deeplab" --seed 4
