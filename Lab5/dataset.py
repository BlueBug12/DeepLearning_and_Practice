import torch
import os
import numpy as np
import csv
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

default_transform = transforms.Compose([
    transforms.ToTensor(),
    ])

class dataset(Dataset):
    def __init__(self, args, mode='train', transform=default_transform):
        assert mode == 'train' or mode == 'test' or mode == 'validate'
        raise NotImplementedError
        
    def set_seed(self, seed):
        if not self.seed_is_set:
            self.seed_is_set = True
            np.random.seed(seed)
            
    def __len__(self):
        raise NotImplementedError
        
    def get_seq(self):
        raise NotImplementedError
    
    def get_csv(self):
        raise NotImplementedError
    
    def __getitem__(self, index):
        self.set_seed(index)
        seq = self.get_seq()
        cond =  self.get_csv()
        return seq, cond
