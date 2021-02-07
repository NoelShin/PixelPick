#!/home/gishin-temp/bin/sh
#python3 simple_test.py  --suffix '' --use_aug --dataset_name 'cv' --use_softmax --n_pixels_by_us 10 --query_strategy "random" --seed 0 --use_contrastive_loss --w_contrastive 1.0 --selection_mode "random" --temperature 1.0
#python3 simple_test.py  --suffix '' --use_aug --dataset_name 'cv' --use_softmax --n_pixels_by_us 10 --query_strategy "random" --seed 0 --use_contrastive_loss --w_contrastive 0.1 --selection_mode "random" --temperature 1.0
#
#python3 simple_test.py  --suffix '' --use_aug --dataset_name 'cv' --use_softmax --n_pixels_by_us 10 --query_strategy "random" --seed 0 --use_contrastive_loss --w_contrastive 1.0 --selection_mode "random" --temperature 0.1
#python3 simple_test.py  --suffix '' --use_aug --dataset_name 'cv' --use_softmax --n_pixels_by_us 10 --query_strategy "random" --seed 0 --use_contrastive_loss --w_contrastive 0.1 --selection_mode "random" --temperature 0.1
#
#python3 simple_test.py  --suffix '' --use_aug --dataset_name 'cv' --use_softmax --n_pixels_by_us 10 --query_strategy "random" --seed 0 --use_contrastive_loss --w_contrastive 1.0 --selection_mode "hard" --temperature 1.0
#python3 simple_test.py  --suffix '' --use_aug --dataset_name 'cv' --use_softmax --n_pixels_by_us 10 --query_strategy "random" --seed 0 --use_contrastive_loss --w_contrastive 1.0 --selection_mode "hard" --temperature 0.1
# python3 simple_test.py  --suffix '' --use_aug --dataset_name 'cv' --use_softmax --n_pixels_by_us 10 --query_strategy "random" --seed 0 --use_contrastive_loss --w_contrastive 1.0 --selection_mode "hard" --temperature 1.0
python3 simple_test.py  --suffix 'rand_init' --use_aug --dataset_name 'cv' --use_softmax --n_pixels_by_us 10 --query_strategy "random" --seed 0 --use_contrastive_loss --w_contrastive 0.1 --selection_mode "hard" --temperature 1.0 --use_region_contrast
# python3 simple_test.py  --suffix 'wd' --use_aug --dataset_name 'cv' --use_softmax --n_pixels_by_us 10 --query_strategy "random" --seed 0 --use_contrastive_loss --w_contrastive 0.1 --selection_mode "hard" --temperature 1.0 --use_region_contrast


#python3 simple_test.py  --suffix '' --use_aug --dataset_name 'cv' --use_softmax --n_pixels_by_us 10 --query_strategy "random" --seed 0 --use_contrastive_loss --w_contrastive 1.0 --selection_mode "hard" --temperature 0.1
#python3 simple_test.py  --suffix '' --use_aug --dataset_name 'cv' --use_softmax --n_pixels_by_us 10 --query_strategy "random" --seed 0 --use_contrastive_loss --w_contrastive 0.1 --selection_mode "hard" --temperature 0.1



