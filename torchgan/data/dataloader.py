import os
import cv2
import json
import matplotlib.pyplot as plt
import numpy as np


import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

def load_config():
    with open('config.json') as f:
        config = json.load(f)
    return config

def get_image_paths(dataset_dir):
    image_paths_A = [os.path.join(dataset_dir, 'trainA', filename) for filename in os.listdir(os.path.join(dataset_dir, 'trainA'))]
    image_paths_B = [os.path.join(dataset_dir, 'trainB', filename) for filename in os.listdir(os.path.join(dataset_dir, 'trainB'))]
    return image_paths_A, image_paths_B

class CycleGANDataset(Dataset):
    def __init__(self, dataset_dir, transform=None):
        self.image_paths_A, self.image_paths_B = get_image_paths(dataset_dir)
        self.transform = transform
        self.image_paths_A.sort()
        self.image_paths_B.sort()
        
        
    def __len__(self):
        return max(len(self.image_paths_A), len(self.image_paths_B))

    def __getitem__(self, idx):
        # read images from domain A
        image_A = cv2.imread(self.image_paths_A[idx % len(self.image_paths_A)])
        image_A = cv2.cvtColor(image_A, cv2.COLOR_BGR2RGB)
        
        # read random image from domain B
        image_B = cv2.imread(self.image_paths_B[np.random.randint(0, len(self.image_paths_B))])
        image_B = cv2.cvtColor(image_B, cv2.COLOR_BGR2RGB)
        if self.transform:
            image_A = self.transform(image_A)
            image_B = self.transform(image_B)
        return image_A, image_B
    
def get_dataloader(mode='train'):
    config = load_config()
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((config['model']['image_size'], config['model']['image_size'])),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    if mode == 'train':
        train_dataset = CycleGANDataset(config['data']['dataset_dir'], transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=config['data']['batch_size'], shuffle=True)
        return train_loader
    else:
        test_dataset = CycleGANDataset(config['data']['dataset_dir'], transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
        return test_loader