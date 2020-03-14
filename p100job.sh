#!/bin/bash
#SBATCH --job-name="ambergpu-shared"
#SBATCH --output="ambergpu-shared.%j.%N.out"
#SBATCH --no-requeue
#SBATCH -p gpu-shared
#SBATCH --gres=gpu:p100:2
#SBATCH --ntasks-per-node=14
#SBATCH -t 24:00:00

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

# Train StarGAN using the AffectNet dataset
python main.py --mode train --dataset RaFD --image_size 128 \
               --c_dim 8 --rafd_image_dir ../AffectNet/faces \
#                --resume_iters 67500 \
               --sample_dir stargan_rafd/samples --log_dir stargan_rafd/logs \
               --model_save_dir stargan_rafd/models --result_dir stargan_rafd/results

echo 'done'

