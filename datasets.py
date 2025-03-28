## dataloader까지 작성

import os 
import torch
import glob
import pydicom
import numpy as np
import random
import matplotlib.pyplot as plt

from PIL import Image
from skimage import exposure

import torchvision
import torch.utils.data as data
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader

# for reproducible result
torch.manual_seed(1)
torch.cuda.manual_seed(1)
torch.backends.cudnn.deterministic = True # deterministic하게 진행한다는 뜻인것 같음
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = False
np.random.seed(1)
random.seed(1)

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

train_img_paths = glob.glob("./data/train/**/*.dcm", recursive=True)
print("train dataset length : ", len(train_img_paths))
train_img_path = train_img_paths[0]
print(train_img_path)

test_img_paths = glob.glob("./data/test/**/*.dcm", recursive=True)
print("test dataset length : ", len(test_img_paths))
test_img_path = test_img_paths[0]
print(test_img_path)


# Dataset
img_size = 256

from torchvision.transforms import Lambda

train_transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

val_transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
])

test_transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
])

class PCDDataset(data.Dataset):
    def __init__(self, split, transform=None):
        self.class_to_int = {'normal' : 0, 'mal' : 1}
        # path
        self.img_paths = glob.glob("./data/train/**/*.dcm", recursive=True)
        self.transform = transform
        self.split = split
        
        from sklearn.model_selection import train_test_split
        train_path, val_path = train_test_split(self.img_paths, train_size=0.9, random_state=1)

        if self.split == 'train':
            self.img_paths = train_path
        elif self.split == 'val':
            self.img_paths = val_path

    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, index):

        img_path = self.img_paths[index]

        if 'normal' in img_path:
            label = np.float32(self.class_to_int['normal'])
        else:
            label = np.float32(self.class_to_int['mal'])

        img_origin = pydicom.dcmread(img_path) # 이제 read_file은 없대
        img = img_origin.pixel_array

        img = exposure.equalize_adapthist(img, clip_limit=1)
        img = (img * 255).astype(np.uint8)
        stack = np.stack((img, img, img), axis=2) # channel방향으로 3개 쌓아주기

        stack_img = Image.fromarray(stack, 'RGB') # transform해주기 위해 skimage.Image로 변환, RGB로도 변환

        # transform
        img_transformed = self.transform(stack_img)

        return img_transformed, label
    
class PCDDataset_test(data.Dataset):
    def __init__(self, transform=None):
        self.class_to_int = {'normal' : 0, 'mal' : 1}
        # path
        self.img_paths = glob.glob("./data/test/**/*.dcm", recursive=True)
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, index):

        img_path = self.img_paths[index]

        if 'normal' in img_path:
            label = np.float32(self.class_to_int['normal'])
        else:
            label = np.float32(self.class_to_int['mal'])

        img_origin = pydicom.dcmread(img_path) # 이제 read_file은 없대
        img = img_origin.pixel_array

        img = exposure.equalize_adapthist(img, clip_limit=1)
        img = (img * 255).astype(np.uint8)
        stack = np.stack((img, img, img), axis=2) # channel방향으로 3개 쌓아주기

        stack_img = Image.fromarray(stack, 'RGB') # transform해주기 위해 skimage.Image로 변환, RGB로도 변환

        # transform
        img_transformed = self.transform(stack_img)

        return img_transformed, label

def dataloader(batch_size):
    train_dataset = PCDDataset('train', transform = train_transform)
    val_dataset = PCDDataset('val', transform=val_transform)

    # train_dataset 데이터 로더 작성
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # validation_dataset 데이터 로더 작성
    validation_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    return train_dataloader, validation_dataloader

def dataloader_test(batch_size):
    test_dataset = PCDDataset_test(transform = test_transform)

    # train_dataset 데이터 로더 작성
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return test_dataloader