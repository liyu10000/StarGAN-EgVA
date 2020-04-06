#!/bin/bash

# # Train StarGAN_VA using the AffectNet dataset
# python main.py --mode train --lambda_cls 0.5 --lambda_reg 20 \
#                --csv_file_train ../AffectNet/Manual_Labels/training4.csv \
#                --csv_file_test ../AffectNet/Manual_Labels/validation4.csv \
#                --resume_iters 60000 \
#                --sample_dir stargan_affectnet/exp11/samples --log_dir stargan_affectnet/exp11/logs \
#                --model_save_dir stargan_affectnet/exp11/models --result_dir stargan_affectnet/exp11/results

# # Test StarGAN_VA using the AffectNet dataset
python3 main.py --mode testpath --label_path_file stargan_affectnet/label_path.txt \
                --csv_file_train ../AffectNet/Manual_Labels/training3.csv \
                --csv_file_test ../AffectNet/Manual_Labels/validation3.csv \
                --test_iters 100000 \
                --sample_dir stargan_affectnet/exp12/samples --log_dir stargan_affectnet/exp12/logs \
                --model_save_dir stargan_affectnet/exp12/models --result_dir stargan_affectnet/exp12/results/100000neu2hpy

echo 'done'

