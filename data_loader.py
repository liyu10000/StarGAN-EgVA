from torch.utils import data
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from PIL import Image
import torch
import os
import shutil
import random
import pickle
import numpy as np
import pandas as pd


class CelebA(data.Dataset):
    """Dataset class for the CelebA dataset."""

    def __init__(self, image_dir, attr_path, selected_attrs, transform, mode):
        """Initialize and preprocess the CelebA dataset."""
        self.image_dir = image_dir
        self.attr_path = attr_path
        self.selected_attrs = selected_attrs
        self.transform = transform
        self.mode = mode
        self.train_dataset = []
        self.test_dataset = []
        self.attr2idx = {}
        self.idx2attr = {}
        self.preprocess()

        if mode == 'train':
            self.num_images = len(self.train_dataset)
        else:
            self.num_images = len(self.test_dataset)

    def preprocess(self):
        """Preprocess the CelebA attribute file."""
        lines = [line.rstrip() for line in open(self.attr_path, 'r')]
        all_attr_names = lines[1].split()
        for i, attr_name in enumerate(all_attr_names):
            self.attr2idx[attr_name] = i
            self.idx2attr[i] = attr_name

        lines = lines[2:]
        random.seed(1234)
        random.shuffle(lines)
        for i, line in enumerate(lines):
            split = line.split()
            filename = split[0]
            values = split[1:]

            label = []
            for attr_name in self.selected_attrs:
                idx = self.attr2idx[attr_name]
                label.append(values[idx] == '1')

            if (i+1) < 2000:
                self.test_dataset.append([filename, label])
            else:
                self.train_dataset.append([filename, label])

        print('Finished preprocessing the CelebA dataset...')

    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""
        dataset = self.train_dataset if self.mode == 'train' else self.test_dataset
        filename, label = dataset[index]
        image = Image.open(os.path.join(self.image_dir, filename))
        return self.transform(image), torch.FloatTensor(label)

    def __len__(self):
        """Return the number of images."""
        return self.num_images

    
class AffectNet(data.Dataset):
    def __init__(self, pkl_file, csv_file, root_dir, transform):
        """
        Args:
            pkl_file (string): Path to the pkl file with file names.
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.df = None
        df = pd.read_csv(csv_file)
        df = df[df.expression < 8] # remove noise images
        with open(pkl_file, 'rb') as f:
            names = pickle.load(f)
        names = [name[:-4] for name in names] # remove .png extention
        df_touse = pd.DataFrame(data={'subDirectory_filePath': names})
        df_touse = pd.merge(df_touse, df, how='inner', on=['subDirectory_filePath'])
        self.df = df_touse
        print('# of files to use', self.df.shape[0])
        
    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                self.df.iloc[idx, 0]+'.png') # 0th col: subDirectory_filePath
        image = Image.open(img_name)
        label = self.df.iloc[idx, 6] # 6th col: expression
        return self.transform(image), label
    
    
def get_loader(image_dir, attr_path, selected_attrs, crop_size=178, image_size=128, 
               batch_size=16, dataset='RaFD', mode='train', num_workers=8):
    """Build and return a data loader."""
    transform = []
    if mode == 'train':
        transform.append(T.RandomHorizontalFlip())
#     transform.append(T.CenterCrop(crop_size))
#     transform.append(T.Resize(image_size))
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
#     transform.append(T.Normalize(mean=(0.56, 0.62, 0.74), std=(0.24, 0.21, 0.18)))  # for RaFD
    transform = T.Compose(transform)

    if dataset == 'CelebA':
        dataset = CelebA(image_dir, attr_path, selected_attrs, transform, mode)
    elif dataset == 'RaFD':
#         dataset = ImageFolder(image_dir, transform)
        pkl_file = '../AffectNet/faces_good.pkl'
        csv_file = '../AffectNet/Manual_Labels/training.csv'
#         csv_file = '../AffectNet/Manual_Labels/validation.csv'
        dataset = AffectNet(pkl_file, csv_file, image_dir, transform)

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=(mode=='train'),
                                  num_workers=num_workers)
    return data_loader


if __name__ == '__main__':
#     # test dataset
#     image_dir = '../AffectNet/faces'
#     pkl_file = '../AffectNet/faces_good.pkl'
#     csv_file = '../AffectNet/Manual_Labels/validation.csv'
#     dataset = AffectNet(pkl_file, csv_file, image_dir, None)
#     n = len(dataset)
    
#     # test dataloader
#     image_dir = '../AffectNet/faces'
#     data_loader = get_loader(image_dir, None, None, batch_size=16)
#     data_iter = iter(data_loader)
#     inputs, labels = next(data_iter)
#     print(inputs.shape, labels.shape)
    
#     # test original dataloader
#     image_dir = '../AffectNet/goodbad/val'
#     data_loader = get_loader(image_dir, None, None, batch_size=16)
#     data_iter = iter(data_loader)
#     inputs, labels = next(data_iter)
#     print(inputs.shape, labels.shape)

    # check category counts after removing bad images
    pkl_file = '../AffectNet/faces_good3.pkl'
    csv_file = '../AffectNet/Manual_Labels/training.csv'
    df = pd.read_csv(csv_file)
    df = df[df.expression < 8] # remove noise images
#     print(df.groupby(['expression']).count())
    with open(pkl_file, 'rb') as f:
        names = pickle.load(f)
    names = [name[:-4] for name in names] # remove .png extention
    print('# files read from {}: {}'.format(pkl_file, len(names)))
    
    df_touse = pd.DataFrame(data={'subDirectory_filePath': names})
    df_touse = pd.merge(df_touse, df, how='inner', on=['subDirectory_filePath'])
    print(df_touse.groupby(['expression']).count())

#     names = random.sample(names, 1000)
#     image_dir = '../AffectNet/faces'
#     sample_dir = '../AffectNet/faces_good_samples'
#     for name in names:
#         fin = os.path.join(image_dir, name+'.png')
#         if os.path.isfile(fin):
#             shutil.copy2(fin, sample_dir)
    