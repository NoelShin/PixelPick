#!/home/gishin-temp/bin/sh
python3 ../main_al.py  --suffix 'scr' --dataset_name 'voc' --query_strategy 'random' --n_pixels_by_us 10 --network_name "deeplab" --n_layers 50 --seed 0 --use_scribbles
python3 ../main_al.py  --suffix 'scr' --dataset_name 'voc' --query_strategy 'random' --n_pixels_by_us 10 --network_name "deeplab" --n_layers 50 --seed 1 --use_scribbles
python3 ../main_al.py  --suffix 'scr' --dataset_name 'voc' --query_strategy 'random' --n_pixels_by_us 10 --network_name "deeplab" --n_layers 50 --seed 2 --use_scribbles
python3 ../main_al.py  --suffix 'scr' --dataset_name 'voc' --query_strategy 'random' --n_pixels_by_us 10 --network_name "deeplab" --n_layers 50 --seed 3 --use_scribbles
python3 ../main_al.py  --suffix 'scr' --dataset_name 'voc' --query_strategy 'random' --n_pixels_by_us 10 --network_name "deeplab" --n_layers 50 --seed 4 --use_scribbles
