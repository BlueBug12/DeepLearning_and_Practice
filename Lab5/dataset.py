import torch
import os
import numpy as np
import csv
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd

default_transform = transforms.Compose([
    transforms.ToTensor(),
    ])

class bair_robot_pushing_dataset(Dataset):
    def __init__(self, args, mode='train', transform=default_transform):
        assert mode == 'train' or mode == 'test' or mode == 'validate'
        if mode == 'train':
            self.data_dir = os.path.join(args.data_root,"processed_data","train")
            self.ordered = False
        elif mode == 'test':
            self.data_dir = os.path.join(args.data_root,"processed_data","test")
            self.ordered = True
        else:
            self.data_dir = os.path.join(args.data_root,"processed_data","validate")
            self.ordered = True
        self.dirs = []

        for d1 in os.listdir(self.data_dir):
            for d2 in os.listdir(os.path.join(self.data_dir,d1)):
                self.dirs.append(os.path.join(self.data_dir,d1,d2))

        self.seq_len = args.n_past + args.n_future
        self.seed_is_set = False
        self.d = 0
        self.to_tensor = transform

    def set_seed(self, seed):
        if not self.seed_is_set:
            self.seed_is_set = True
            np.random.seed(seed)
            
    def __len__(self):
        return len(self.dirs)
        
    def get_seq(self):
        if not self.ordered:
            self.d = np.random.randint(len(self.dirs))
        d = self.dirs[self.d]
        image_seq = []
        for i in range(self.seq_len):
            fname = '%s/%d.png' % (d, i)
            im = Image.open(fname).convert('RGB')
            im = np.asarray(im).reshape(64, 64, 3)
            im = self.to_tensor(im)
            im = im.reshape(1,3,64,64)
            image_seq.append(im)
        image_seq = np.concatenate(image_seq, axis=0)
        return image_seq
    
    def get_csv(self):
        d = self.dirs[self.d]
        act_df = pd.read_csv(os.path.join(d,"actions.csv"), header=None)
        pos_df = pd.read_csv(os.path.join(d,"endeffector_positions.csv"), header=None)
        act_seq = np.asarray(act_df.iloc[:self.seq_len],dtype='float64')
        pos_seq = np.asarray(pos_df.iloc[:self.seq_len],dtype='float64')
        return torch.from_numpy(np.concatenate([act_seq,pos_seq],axis = 1))
    
    def __getitem__(self, index):
        self.set_seed(index)
        seq = self.get_seq()
        cond =  self.get_csv()
        if self.d == len(self.dirs) - 1:
            self.d = 0
        else:
            self.d+=1
        return seq, cond
