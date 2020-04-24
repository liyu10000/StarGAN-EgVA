import os
import argparse
from solver import Solver
from data_loader import get_loader
from torch.backends import cudnn


def read_path(label_path_file):
    path = []
    with open(label_path_file, 'r') as f:
        for line in f.readlines():
            tokens = line.strip().split()
            # read till a blank line
            if (not tokens) or (len(tokens) == 0):
                break
            cat = int(tokens[0])
            v = (float(tokens[1]) + 1) / 2
            a = (float(tokens[2]) + 1) / 2
            path.append([cat, v, a])
    return path

def str2bool(v):
    return v.lower() in ('true')

def main(config):
    # For fast training.
    cudnn.benchmark = True

    # Create directories if not exist.
    os.makedirs(config.log_dir, exist_ok=True)
    os.makedirs(config.model_save_dir, exist_ok=True)
    os.makedirs(config.sample_dir, exist_ok=True)
    os.makedirs(config.result_dir, exist_ok=True)

    # Data loader.
    if config.mode == 'train':
        # Data loader.
        train_loader = get_loader(config.csv_file_train, config.image_dir, 
                                  config.batch_size, config.mode, config.num_workers)

        # Solver for training.
        solver = Solver(train_loader, config)
    elif config.mode == 'test' or config.mode == 'testpath':
        # Data loader.
        test_loader  = get_loader(config.csv_file_test, config.image_dir, 
                                  config.batch_size, config.mode, config.num_workers)

        # Solver for testing.
        solver = Solver(test_loader, config)

    if config.mode == 'train':
        solver.train()
    elif config.mode == 'test':
        solver.test()
    elif config.mode == 'testpath':
        path = read_path(config.label_path_file)
        solver.testpath(path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model configuration.
    parser.add_argument('--c_dim', type=int, default=8, help='dimension of emotion categories')
    parser.add_argument('--r_dim', type=int, default=2, help='dimension of va for regression')
    parser.add_argument('--image_size', type=int, default=128, help='image resolution')
    parser.add_argument('--g_conv_dim', type=int, default=64, help='number of conv filters in the first layer of G')
    parser.add_argument('--d_conv_dim', type=int, default=64, help='number of conv filters in the first layer of D')
    parser.add_argument('--g_repeat_num', type=int, default=6, help='number of residual blocks in G')
    parser.add_argument('--d_repeat_num', type=int, default=6, help='number of strided conv layers in D')
    parser.add_argument('--lambda_cls', type=float, default=1, help='weight for domain classification loss')
    parser.add_argument('--lambda_reg', type=float, default=5, help='weight for domain regression loss')
    parser.add_argument('--lambda_rec', type=float, default=10, help='weight for reconstruction loss')
    parser.add_argument('--lambda_gp', type=float, default=10, help='weight for gradient penalty')
    
    # Training configuration.
    parser.add_argument('--batch_size', type=int, default=16, help='mini-batch size')
    parser.add_argument('--num_iters', type=int, default=200000, help='number of total iterations for training D')
    parser.add_argument('--num_iters_decay', type=int, default=50000, help='number of iterations for decaying lr')
    parser.add_argument('--g_lr', type=float, default=0.0001, help='learning rate for G')
    parser.add_argument('--d_lr', type=float, default=0.0001, help='learning rate for D')
    parser.add_argument('--n_critic', type=int, default=5, help='number of D updates per each G update')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
    parser.add_argument('--resume_iters', type=int, default=None, help='resume training from this step')

    # Test configuration.
    parser.add_argument('--test_iters', type=int, default=200000, help='test model from this step')
    # Test path configuration.
    parser.add_argument('--test_1st_batch', type=str2bool, default=True, help='only test the first batch')
    parser.add_argument('--infer_cat', type=str2bool, default=False, help='infer emotion category from va')
    parser.add_argument('--label_path_file', type=str, default='stargan_affectnet/label_path.txt', help='file to store (cat, v, a) values for test')

    # Miscellaneous.
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'testpath'])
    parser.add_argument('--use_tensorboard', type=str2bool, default=True)

    # Directories.
    parser.add_argument('--csv_file_train', type=str, default='../AffectNet/Manual_Labels/training3.csv')
    parser.add_argument('--csv_file_test', type=str, default='../AffectNet/Manual_Labels/validation3.csv')
    parser.add_argument('--image_dir', type=str, default='../AffectNet/faces')
    parser.add_argument('--log_dir', type=str, default='stargan_affectnet/')
    parser.add_argument('--model_save_dir', type=str, default='stargan_affectnet/')
    parser.add_argument('--sample_dir', type=str, default='stargan_affectnet/')
    parser.add_argument('--result_dir', type=str, default='stargan_affectnet/')

    # Step size.
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--sample_step', type=int, default=2000)
    parser.add_argument('--model_save_step', type=int, default=5000)
    parser.add_argument('--lr_update_step', type=int, default=1000)

    config = parser.parse_args()
    print(config)
    main(config)