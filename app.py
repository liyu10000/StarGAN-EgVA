import os
import cv2
import web
import torch
import argparse
import numpy as np
from PIL import Image
from torchvision import transforms as T
from torchvision.utils import save_image
from model import Generator


def get_config():
    parser = argparse.ArgumentParser()

    # Model configuration.
    parser.add_argument('--r_dim', type=int, default=2, help='dimension of va for regression')
    parser.add_argument('--g_conv_dim', type=int, default=64, help='number of conv filters in the first layer of G')
    parser.add_argument('--g_repeat_num', type=int, default=6, help='number of residual blocks in G')

    # Test configuration.
    parser.add_argument('--test_iters', type=int, default=200000, help='test model from this step')

    # Directories.
    parser.add_argument('--csv_file_test', type=str, default='../AffectNet/Manual_Labels/validation3.csv')
    parser.add_argument('--image_dir', type=str, default='../AffectNet/faces')
    parser.add_argument('--model_save_dir', type=str, default='stargan_affectnet/exp14/models')
    parser.add_argument('--result_dir', type=str, default='stargan_affectnet/exp14/results')

    config = parser.parse_args()
    return config

def get_transform():
    transform = []
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    transform = T.Compose(transform)
    return transform

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

class Solver:
    def __init__(self, config):
        # Model configurations.
        self.r_dim = config.r_dim
        self.g_conv_dim = config.g_conv_dim
        self.g_repeat_num = config.g_repeat_num
        self.test_iters = config.test_iters
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_save_dir = config.model_save_dir
        self.result_dir = config.result_dir
        self.transform = get_transform()

        self.build_model()
        self.restore_model()

    def build_model(self):
        """Create a generator and a discriminator."""
        self.G = Generator(self.g_conv_dim, self.r_dim, self.g_repeat_num)
        self.G.to(self.device)
        # self.G.eval() # won't work at small batch size, set track_running_stats = False to solve it.

    def restore_model(self):
        """Restore the trained generator."""
        print('Loading the trained models from step {}...'.format(self.test_iters))
        G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(self.test_iters))
        self.G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))

    def create_path_labels(self, path):
        """Generate target va values for testing.
        :param path: list of [cat, v, a] # note the input still has the info of cat
        """
        r_trg_list = []
        for cat, v, a in path:
            r_trg = torch.tensor([[v, a]])
            r_trg_list.append(r_trg.to(self.device))
        return r_trg_list

    def denorm(self, x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def run(self, image, label_path):
        image = self.transform(image)
        image = image.unsqueeze(0)

        with torch.no_grad():
            # Prepare input images and target domain labels.
            image = image.to(self.device)
            r_trg_list = self.create_path_labels(label_path)

            # Translate images.
            x_fake_list = [image]
            for r_trg in r_trg_list:
                x_fake_list.append(self.G(image, r_trg))

            # Save the translated images.
            x_concat = torch.cat(x_fake_list, dim=3)
            result_path = os.path.join(self.result_dir, '{}-images.jpg'.format(1))
            save_image(self.denorm(x_concat.data.cpu()), result_path, nrow=1, padding=0)
            print('Saved real and fake images into {}...'.format(result_path))


if __name__ == '__main__':
    config = get_config()
    solver = Solver(config)

    while True:
        print()
        name = input("Enter image name:")
        if not name or len(name) == 0:
            name = '6f95a35fc30a9e2135b48b5dff6314175a4758ad78e6ba1ca04bbb99'
        image_name = '../AffectNet/faces/{}.png'.format(name)
        image = Image.open(image_name)
        label_path_file = './stargan_affectnet/label_path.txt'
        label_path = read_path(label_path_file)

        solver.run(image, label_path)

        break