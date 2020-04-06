#!/bin/bash
#SBATCH --job-name="ambergpu"
#SBATCH --output="ambergpu.%j.%N.out"
#SBATCH --no-requeue
#SBATCH -p gpu
#SBATCH --gres=gpu:k80:4
#SBATCH --ntasks-per-node=24
#SBATCH -t 48:00:00

module purge
module load cuda/10.1

source activate py37

# # Train StarGAN using the RaFD dataset
# python main.py --mode train --dataset RaFD --image_size 128 \
#                --c_dim 8 --rafd_image_dir data/RaFD/train \
# #                --resume_iters 67500 \
#                --sample_dir stargan_rafd/samples --log_dir stargan_rafd/logs \
#                --model_save_dir stargan_rafd/models --result_dir stargan_rafd/results

# # Test StarGAN using the RaFD dataset
# python main.py --mode test --dataset RaFD --image_size 128 \
#                --c_dim 8 --rafd_image_dir data/RaFD/test \
#                --test_iters 120000 \
#                --sample_dir stargan_rafd/samples --log_dir stargan_rafd/logs \
#                --model_save_dir stargan_rafd/models --result_dir stargan_rafd/results


# # Train StarGAN using the AffectNet dataset
# python main.py --mode train --dataset RaFD --image_size 128 \
#                --c_dim 8 --rafd_image_dir ../AffectNet/faces \
#                --resume_iters 170000 \
#                --sample_dir stargan_affectnet/samples --log_dir stargan_affectnet/logs \
#                --model_save_dir stargan_affectnet/models --result_dir stargan_affectnet/results

# # Test StarGAN using the AffectNet dataset
# python main.py --mode test --dataset RaFD --image_size 128 \
#                --c_dim 8 --rafd_image_dir ../AffectNet/faces \
#                --test_iters 200000 \
#                --sample_dir stargan_affectnet/samples --log_dir stargan_affectnet/logs \
#                --model_save_dir stargan_affectnet/models --result_dir stargan_affectnet/results


# # Train StarGAN_VA using the AffectNet dataset
# python main.py --mode train --lambda_cls 0.5 --lambda_reg 20 \
#                --csv_file_train ../AffectNet/Manual_Labels/training4.csv \
#                --csv_file_test ../AffectNet/Manual_Labels/validation4.csv \
#                --resume_iters 60000 \
#                --sample_dir stargan_affectnet/exp11/samples --log_dir stargan_affectnet/exp11/logs \
#                --model_save_dir stargan_affectnet/exp11/models --result_dir stargan_affectnet/exp11/results

# # Test StarGAN_VA using the AffectNet dataset
python main.py --mode testpath --label_path_file stargan_affectnet/label_path.txt \
               --csv_file_train ../AffectNet/Manual_Labels/training4.csv \
               --csv_file_test ../AffectNet/Manual_Labels/validation4.csv \
			   --test_iters 100000 \
               --sample_dir stargan_affectnet/exp11/samples --log_dir stargan_affectnet/exp11/logs \
               --model_save_dir stargan_affectnet/exp11/models --result_dir stargan_affectnet/exp11/results

echo 'done'

