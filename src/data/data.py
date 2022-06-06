import os
import numpy as np
import glob
import PIL.Image as Image
from omegaconf import DictConfig
from tqdm.notebook import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import zipfile
import gdown

class Mikkel_Data(Dataset):
    def __init__(self, train, transform, base_path="./", data_path='data/processed'):
        # 'Initialization'
        self.transform = transform
        data_path = os.path.join(base_path, data_path)
        #image_classes = [os.path.split(d)[1] for d in glob.glob(data_path +'/*') if os.path.isdir(d)]
        #image_classes.sort()
        self.name_to_label = {}
        for filename in os.listdir(data_path):
            if "pos" in filename:
                self.name_to_label[filename] = 1
            elif "neg" in filename:
                self.name_to_label[filename] = 0    
        self.image_paths = glob.glob(data_path + '/*/*.png')
        
        self.targets = np.array([self.name_to_label[os.path.split(os.path.split(image_path)[0])[1]]
                                 for image_path in  self.image_paths])
        
    def __len__(self):
        # 'Returns the total number of samples'
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 'Generates one sample of data'
        image_path = self.image_paths[idx]
        
        image = Image.open(image_path)
        c = os.path.split(os.path.split(image_path)[0])[1]
        y = self.name_to_label[c]
        X = self.transform(image)
        return X, y


def get_data(size, train_augmentation, batch_size, base_path: str = './'):
    train_transform, valid_transform = get_transforms(size, train_augmentation)

    train_set = Hotdog_NotHotdog(train=True, transform=train_transform, base_path=base_path)
    valid_set = Hotdog_NotHotdog(train=False, transform=valid_transform, base_path=base_path)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, valid_loader


def get_transforms(size, train_augmentation):
    norm_mean, norm_std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    train_transform = list()
    if 'random_crop' in train_augmentation:
        train_transform.append(transforms.Resize((int(1.1*size), int(1.1*size))))
        train_transform.append(transforms.RandomCrop((size, size)))
    else:
        train_transform.append(transforms.Resize((size, size)))
    if 'random_horizontal_flip' in train_augmentation:
        train_transform.append(transforms.RandomHorizontalFlip())
    if 'color_jitter' in train_augmentation:
        train_transform.append(transforms.ColorJitter())
    train_transform.append(transforms.ToTensor())
    train_transform.append(transforms.Normalize(norm_mean, norm_std))
    train_transform = transforms.Compose(train_transform)

    valid_transform = [transforms.Resize((size, size)),
                       transforms.ToTensor(),
                       transforms.Normalize(norm_mean, norm_std)]
    valid_transform = transforms.Compose(valid_transform)
    return train_transform, valid_transform

def plot_data(loader):
    images, labels = next(iter(loader))
    plt.figure(figsize=(20,10))
    
    for i in range(21):
        plt.subplot(5,7,i+1)
        plt.imshow(np.swapaxes(np.swapaxes(images[i].numpy(), 0, 2), 0, 1))
        plt.title(['hotdog', 'not hotdog'][labels[i].item()])
        plt.axis('off')
