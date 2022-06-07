import os
import numpy as np
import glob
import PIL.Image as Image
#from omegaconf import DictConfig
from tqdm.notebook import tqdm

import torch
#import torch.nn as nn
import torch.nn.functional as F
#import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

class Mikkel_Data(Dataset):
    def __init__(self, transform="placeholder", base_path="./", data_path='data/processed'):
        # 'Initialization'
        self.transform = transform
        data_path = os.path.join(base_path, data_path)
        self.name_to_label = {}
        for filename in os.listdir(data_path):
            if "pos" in filename:
                self.name_to_label[filename] = 1
            elif "neg" in filename:
                self.name_to_label[filename] = 0    
        self.image_paths = glob.glob(base_path+data_path + '/*jpg')
        
    def __len__(self):
        # 'Returns the total number of samples'
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 'Generates one sample of data'
        image_path = self.image_paths[idx]
        image_name = image_path.split("/")[-1]
        image = Image.open(image_path)

        y = self.name_to_label[image_name]
        #X = self.transform(image)

        norm_mean, norm_std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        transform_to_tensor = transforms.ToTensor()
        X = transform_to_tensor(image)
        normalize = transforms.Normalize(norm_mean, norm_std)
        X = normalize(X)
        return X, y


def get_data(batch_size, base_path: str = './'):
    
    #train_transform = get_transforms(size, train_augmentation)
    dataset = Mikkel_Data(base_path=base_path)
    num_total_samples = len(dataset)
    num_valid_samples = int(num_total_samples * 0.2)
    train_set, valid_set = torch.utils.data.random_split(dataset, (num_total_samples-num_valid_samples, num_valid_samples)) 
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=2)
    
    batch = next(iter(train_loader))
    #print("batch.shape", batch.shape)

    return train_loader, valid_loader


def get_transforms(size, train_augmentation):
    # Skip augmentations for now
    raise NotImplemented
    '''# ImageNet values
    norm_mean, norm_std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    train_transform = list()
    if 'random_crop' in train_augmentation:
        train_transform.append(transforms.Resize((int(1.1*size), int(1.1*size))))
        train_transform.append(transforms.RandomCrop((size, size)))
    else:
        train_transform.append(transforms.Resize((size, size)))
    if 'random_horizontal_flip' in train_augmentation:
        train_transform.append(transforms.RandomHorizontalFlip())
   
    train_transform.append(transforms.ToTensor())
    train_transform.append(transforms.Normalize(norm_mean, norm_std))
    train_transform = transforms.Compose(train_transform)

    valid_transform = [transforms.Resize((size, size)),
                       transforms.ToTensor(),
                       transforms.Normalize(norm_mean, norm_std)]
    valid_transform = transforms.Compose(valid_transform)
    return train_transform, valid_transform
    '''

def plot_data(loader):
    images, labels = next(iter(loader))
    plt.figure(figsize=(20,10))
    
    for i in range(21):
        plt.subplot(5,7,i+1)
        plt.imshow(np.swapaxes(np.swapaxes(images[i].numpy(), 0, 2), 0, 1))
        plt.title(['hotdog', 'not hotdog'][labels[i].item()])
        plt.axis('off')

def test_code():
    train_loader, valid_loader = get_data(batch_size=2, base_path = './')
    print(len(train_loader))
    print(len(valid_loader))
    
    #print(next(iter(train_loader)))
if __name__ == '__main__':
    test_code()
