# -*- coding: utf-8 -*-
import os
import json
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset

def get_train_data(root_dir, train_json, obj_json):
    #key: image_name, value: objects_in_image
    data = json.load(open(os.path.join(root_dir, train_json)))

    #index of every object
    obj = json.load(open(os.path.join(root_dir, obj_json)))
    print(train_json)
    label = list(data.values())
    img_name = list(data.keys())
    
    for i in range(len(label)):
        for j in range(len(label[i])):
            #replace object name to object index
            label[i][j] = obj[label[i][j]]
        one_hot = np.zeros(len(obj))

        #if the object exist, replace 1, otherwise 0
        one_hot[label[i]] = 1
        label[i] = one_hot
    return np.asarray(img_name), np.asarray(label)

def get_test_data(root_dir, test_json, obj_json):
    label = json.load(open(os.path.join(root_dir, test_json)))
    obj = json.load(open(os.path.join(root_dir, obj_json)))
    for i in range(len(label)):
        for j in range(len(label[i])):
            label[i][j] = obj[label[i][j]]
        one_hot = np.zeros(len(obj))
        one_hot[label[i]] = 1
        label[i] = one_hot
    return np.asarray(label)

class CLEVRDataset(Dataset):
    def __init__(self, json_dir, dataset_dir, trans, cond=True, mode='train', train_json=None, obj_json=None, test_json=None):
        self.dataset_dir = dataset_dir
        self.trans = trans
        self.cond = cond
        self.mode = mode
        self.num_classes = 24
        if mode == 'train':#root_dir, train_json, obj_json
            self.img_list, self.label_list = get_train_data(json_dir, train_json, obj_json)
            print(f'> Found {len(self.img_list)} images...')
        elif mode == 'test':
            self.label_list = get_test_data(json_dir, test_json, obj_json)
        else:
            raise ValueError('Unknown mode: %s' % mode)

    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, index):
        if self.mode == 'train':
            img = Image.open(os.path.join(self.dataset_dir, self.img_list[index])).convert('RGB')
            img = self.trans(img)
            return img, torch.Tensor(self.label_list[index])
        else:
            cond = self.label_list[index]
            return torch.Tensor(cond)