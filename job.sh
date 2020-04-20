#!/bin/bash

# Train StarGAN_VA using the AffectNet dataset
python3 main.py --mode train --lambda_reg 40 \
                --csv_file_train ../AffectNet/Manual_Labels/training3.csv \
                --sample_dir stargan_affectnet/exp14/samples --log_dir stargan_affectnet/exp14/logs \
                --model_save_dir stargan_affectnet/exp14/models --result_dir stargan_affectnet/exp14/results

# # Test StarGAN_VA using the AffectNet dataset
# python3 main.py --mode testpath --label_path_file stargan_affectnet/label_path.txt \
#                 --csv_file_test ../AffectNet/Manual_Labels/validation3.csv \
#                 --batch_size 32 --lambda_reg 40 \
#                 --test_iters 200000 \
#                 --sample_dir stargan_affectnet/exp14/samples --log_dir stargan_affectnet/exp14/logs \
#                 --model_save_dir stargan_affectnet/exp14/models --result_dir stargan_affectnet/exp14/results/200000

echo 'done'

