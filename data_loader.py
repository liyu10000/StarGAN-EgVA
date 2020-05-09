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

    
class AffectNet(data.Dataset):
    def __init__(self, csv_file, root_dir, transform):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        df = pd.read_csv(csv_file)        
        # Convert the range from [-1, 1] to [0, 1].
        df.valence = (df.valence + 1) / 2
        df.arousal = (df.arousal + 1) / 2
        self.df = df
        
    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                self.df.iloc[idx, 0]+'.png') # 0th col: subDirectory_filePath
        image = Image.open(img_name)
        label = self.df.iloc[idx, 1] # 1th col: expression
        valence = self.df.iloc[idx, 2] # 2th col: valence
        arousal = self.df.iloc[idx, 3] # 3th col: arousal
        return self.transform(image), label, torch.FloatTensor((valence, arousal))
    
    
def get_loader(csv_file, image_dir, batch_size, mode, num_workers):
    """Build and return a data loader."""
    transform = []
    if mode == 'train':
        transform.append(T.RandomHorizontalFlip())
    # transform.append(T.CenterCrop(crop_size))
    # transform.append(T.Resize(image_size))
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    # transform.append(T.Normalize(mean=(0.56, 0.62, 0.74), std=(0.24, 0.21, 0.18)))  # for RaFD
    transform = T.Compose(transform)

    dataset = AffectNet(csv_file, image_dir, transform)
    print('# of files to use', len(dataset))

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=(mode=='train'),
                                  num_workers=num_workers)
    return data_loader


if __name__ == '__main__':
    # test dataset
    image_dir = '../AffectNet/faces'
    pkl_file = '../AffectNet/faces_good4.pkl'
    csv_file = '../AffectNet/Manual_Labels/validation4.csv'
    dataset = AffectNet(csv_file, image_dir, None)
    n = len(dataset)
    
    # test dataloader
    image_dir = '../AffectNet/faces'
    data_loader = get_loader(csv_file, image_dir, batch_size=16, mode='test', num_workers=2)
    data_iter = iter(data_loader)
    inputs, vas, labels = next(data_iter)
    print(inputs.shape, vas.shape, labels.shape)
        