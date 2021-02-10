#!/home/gishin-temp/bin/sh
python3 ../main_al.py  --suffix '' --use_aug --dataset_name 'cv' --use_softmax --query_strategy 'random' --n_pixels_by_us 10 --seed 0 --n_layers 18 --network_name "FPN" --debug

# python3 ../main_al.py  --suffix '' --use_aug --dataset_name 'cv' --use_softmax --query_strategy 'margin_sampling' --n_pixels_by_us 10 --seed 0 --use_contrastive_loss --w_contrastive 0.01
# python3 ../main_al.py  --suffix '' --use_aug --dataset_name 'cv' --use_softmax --query_strategy 'margin_sampling' --n_pixels_by_us 10 --seed 0 --use_contrastive_loss --temperature 0.1 --selection_mode "random"
# python3 ../main_al.py  --suffix '' --use_aug --dataset_name 'cv' --use_softmax --query_strategy 'margin_sampling' --n_pixels_by_us 10 --seed 0 --use_contrastive_loss --w_contrastive 0.01 --selection_mode "random"

# python3 ../main_al.py  --suffix '' --use_aug --dataset_name 'cv' --use_softmax --query_strategy 'entropy' --n_pixels_by_us 10 --seed 1 --use_pseudo_label --window_size 3
# python3 ../main_al.py  --suffix '' --use_aug --dataset_name 'cv' --use_softmax --query_strategy 'entropy' --n_pixels_by_us 10 --seed 2 --use_pseudo_label --window_size 3
# python3 ../main_al.py  --suffix '' --use_aug --dataset_name 'cv' --use_softmax --query_strategy 'entropy' --n_pixels_by_us 10 --seed 3 --use_pseudo_label --window_size 3
# python3 ../main_al.py  --suffix '' --use_aug --dataset_name 'cv' --use_softmax --query_strategy 'entropy' --n_pixels_by_us 10 --seed 4 --use_pseudo_label --window_size 3


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
