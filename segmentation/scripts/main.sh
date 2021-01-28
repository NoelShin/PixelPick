#!/home/gishin-temp/bin/sh
python3 ../main_al.py  --suffix '' --use_aug --use_softmax --query_strategy 'margin_sampling' --n_pixels_by_us 0 --n_pixels_by_oracle_cb 10 --seed 0 --debug
#python3 ../main_al.py --n_pixels_per_img 10 --seed 2  --suffix '' --use_aug --use_softmax --query_strategy 'random' --use_oracle_cb
#python3 ../main_al.py --n_pixels_per_img 10 --seed 3  --suffix '' --use_aug --use_softmax --query_strategy 'random' --use_oracle_cb
#python3 ../main_al.py --n_pixels_per_img 10 --seed 4  --suffix '' --use_aug --use_softmax --query_strategy 'random' --use_oracle_cb


#python3 ../main_al.py --n_pixels_per_img 10 --seed 0  --suffix '' --use_aug --use_softmax --query_strategy 'least_confidence' --use_mc_dropout
#python3 ../main_al.py --n_pixels_per_img 10 --seed 0  --suffix '' --use_aug --use_softmax --query_strategy 'least_confidence' --use_cb_sampling
#python3 ../main_al.py --n_pixels_per_img 10 --seed 0  --suffix '' --use_aug --use_softmax --query_strategy 'least_confidence' --use_mc_dropout --use_cb_sampling
#python3 ../main_al.py --n_pixels_per_img 10 --seed 0  --suffix '' --use_aug --use_softmax --query_strategy 'least_confidence'

#python3 ../main_al.py --n_pixels_per_img 10 --seed 0  --suffix '' --use_aug --use_softmax --query_strategy 'margin_sampling' --use_cb_sampling
#python3 ../main_al.py --n_pixels_per_img 10 --seed 0  --suffix '' --use_aug --use_softmax --query_strategy 'entropy' --use_cb_sampling
#python3 ../main_al.py --n_pixels_per_img 10 --seed 1  --suffix '' --use_aug --use_softmax --query_strategy 'entropy' --use_mc_dropout
#python3 ../main_al.py --n_pixels_per_img 10 --seed 2  --suffix '' --use_aug --use_softmax --query_strategy 'entropy' --use_mc_dropout
#python3 ../main_al.py --n_pixels_per_img 10 --seed 3  --suffix '' --use_aug --use_softmax --query_strategy 'entropy' --use_mc_dropout
#python3 ../main_al.py --n_pixels_per_img 10 --seed 4  --suffix '' --use_aug --use_softmax --query_strategy 'entropy' --use_mc_dropout
#
#python3 ../main_al.py --n_pixels_per_img 10 --seed 0  --suffix '' --use_aug --use_softmax --query_strategy 'least_confidence' --use_mc_dropout
#python3 ../main_al.py --n_pixels_per_img 10 --seed 1  --suffix '' --use_aug --use_softmax --query_strategy 'least_confidence' --use_mc_dropout
#python3 ../main_al.py --n_pixels_per_img 10 --seed 2  --suffix '' --use_aug --use_softmax --query_strategy 'least_confidence' --use_mc_dropout
#python3 ../main_al.py --n_pixels_per_img 10 --seed 3  --suffix '' --use_aug --use_softmax --query_strategy 'least_confidence' --use_mc_dropout
#python3 ../main_al.py --n_pixels_per_img 10 --seed 4  --suffix '' --use_aug --use_softmax --query_strategy 'least_confidence' --use_mc_dropout


#python3 ../main_al.py --n_pixels_per_img 10 --seed 2  --suffix '' --use_aug --query_strategy 'margin_sampling'
#python3 ../main_al.py --n_pixels_per_img 10 --seed 3  --suffix '' --use_aug --query_strategy 'margin_sampling'
#python3 ../main_al.py --n_pixels_per_img 10 --seed 4  --suffix '' --use_aug --query_strategy 'margin_sampling'

#python3 ../main_al.py --n_pixels_per_img 0 --seed 1  --suffix '' --use_aug --use_softmax --query_strategy 'random'
#python3 ../main_al.py --n_pixels_per_img 0 --seed 2  --suffix '' --use_aug --use_softmax --query_strategy 'random'
#python3 ../main_al.py --n_pixels_per_img 0 --seed 3  --suffix '' --use_aug --use_softmax --query_strategy 'random'
#python3 ../main_al.py --n_pixels_per_img 0 --seed 4  --suffix '' --use_aug --use_softmax --query_strategy 'random'
#

# python3 ../main_al.py --n_pixels_per_img 10 --seed 0  --suffix 'whole_only_l2_wo_pho_aug' --use_aug --use_softmax --query_strategy 'random' --use_img_inp --w_img_inp 1
