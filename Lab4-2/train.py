#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import pickle
import os
import torch
from torch.utils import data
import torch.nn as nn
import torchvision.models as models
from tqdm import tqdm
from torchvision import transforms
from PIL import Image
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from argparse import ArgumentParser

# In[2]:

def getTime():
    nowTime = int(time.time()) 
    struct_time = time.localtime(nowTime) 
    timeString = time.strftime("%Y %m %d %I:%M:%S %P", struct_time) 
    return "_".join(timeString.split())

def getData(mode):
    if mode == 'train':
        img = pd.read_csv('train_img.csv')
        label = pd.read_csv('train_label.csv')
        return np.squeeze(img.values), np.squeeze(label.values)
    else:
        img = pd.read_csv('test_img.csv')
        label = pd.read_csv('test_label.csv')
        return np.squeeze(img.values), np.squeeze(label.values)


class RetinopathyDataset(data.Dataset):
    def __init__(self, root, mode):
        """
        Args:
            root (string): Root path of the dataset.
            mode : Indicate procedure status(training or testing)

            self.img_name (string list): String list that store all image names.
            self.label (int or float list): Numerical list that store all ground truth label values.
        """
        self.root = root
        self.img_name, self.label = getData(mode)
        self.mode = mode
        transform = [
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(degrees=25),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)
        ]
        self.to_tensor = transforms.ToTensor()
        self.trans = transforms.RandomOrder(transform)
        print("> Found %d images..." % (len(self.img_name)))

    def __len__(self):
        """'return the size of dataset"""
        return len(self.img_name)

    def __getitem__(self, index):
        """something you should implement here"""

        """
           step1. Get the image path from 'self.img_name' and load it.
                  hint : path = root + self.img_name[index] + '.jpeg'
           
           step2. Get the ground truth label from self.label
                     
           step3. Transform the .jpeg rgb images during the training phase, such as resizing, random flipping, 
                  rotation, cropping, normalization etc. But at the beginning, I suggest you follow the hints. 
                       
                  In the testing phase, if you have a normalization process during the training phase, you only need 
                  to normalize the data. 
                  
                  hints : Convert the pixel value to [0, 1]
                          Transpose the image shape from [H, W, C] to [C, H, W]
                         
            step4. Return processed image and label
        """
        path = self.root + self.img_name[index] + ".jpeg"
        label = self.label[index]
        img = Image.open(path).convert('RGB')
        if self.mode == 'train':
            img = self.trans(img)
        img = self.to_tensor(img)

        return img, label


# In[3]:


def downsample(in_channel, out_channel, stride):
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel_size=(1, 1), stride=stride, bias=False),
        nn.BatchNorm2d(out_channel))

class BasicBlock(nn.Module):
    def __init__(self, in_channel, out_channel, s=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=(3, 3), stride=s, padding=(1, 1), bias=False)
        if s!=1:
            self.identity = downsample(in_channel, out_channel, s)
        else:
            self.identity =  lambda x:x
            
        self.bn1 = nn.BatchNorm2d(out_channel, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out2 = self.identity(x)
        out = self.relu(out+out2)
        return out
    
class Bottleneck(nn.Module):
    def __init__(self, in_channel, out_channel, s=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, mid_channel, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channel, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2 = nn.Conv2d(mid_channel, mid_channel, kernel_size=(3, 3), stride=s, padding=(1, 1), bias=False)
        if s!= 1:
            self.identity = downsample(in_channel, out_channel, s)
        else:
            self.identity = lambda x:x
        self.bn2 = nn.BatchNorm2d(mid_channel, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv3 = nn.Conv2d(mid_channel, out_channel, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(out_channel, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.bn3(out)
        out2 = self.identity(x)
        out = self.relu(out+out2)
        return out


# In[4]:


class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18,self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.bn1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1,ceil_mode=False)
        
        self.layer1 = nn.Sequential(
            BasicBlock(64,64),
            BasicBlock(64,64)
        )
        self.layer2 = nn.Sequential(
            BasicBlock(64,128,2),
            BasicBlock(128,128)
        )
        self.layer3 = nn.Sequential(
            BasicBlock(128,256,2),
            BasicBlock(256,256)
        )
        self.layer4 = nn.Sequential(
            BasicBlock(256,512,2),
            BasicBlock(512,512)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 5, bias=True)
    
    def forward(self,x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.layer1(out)        
        out = self.layer2(out)
        out = self.layer3(out)        
        out = self.layer4(out)        
        out = self.avgpool(out)
        out = self.fc(out.reshape(out.shape[0], -1))
        return out
    


# In[5]:


class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.bn1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1,ceil_mode=False)
        
        self.layer1 = nn.Sequential(
            Bottleneck(64, 64, 256),
            Bottleneck(256, 64, 256),
            Bottleneck(256, 64, 256)
        )
        self.layer2 = nn.Sequential(
            Bottleneck(256, 128, 512, 2),
            Bottleneck(512, 128, 512),
            Bottleneck(512, 128, 512),
            Bottleneck(512, 128, 512)
        )
        self.layer3 = nn.Sequential(
            Bottleneck(512, 256, 1024, 2),
            Bottleneck(1024, 256, 1024),
            Bottleneck(1024, 256, 1024),
            Bottleneck(1024, 256, 1024),
            Bottleneck(1024, 256, 1024),
            Bottleneck(1024, 256, 1024)
        )
        self.layer4 = nn.Sequential(
            Bottleneck(1024, 512, 2048, 2),
            Bottleneck(2048, 512, 2048),
            Bottleneck(2048, 512, 2048)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, 5)
            
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = self.fc(out.reshape(out.shape[0], -1))
        return out


# In[6]:


def evaluate(model, device ,loader, criterion):
    model.eval()
    correct = 0
    loss = 0
    with torch.no_grad():
        #batch_pbar = tqdm(loader)
        for i, (data, label) in enumerate(loader):
            data, label = data.to(device), label.to(device)
            out = model.forward(data)
            loss += criterion(out, label).item()
            out = torch.argmax(out, dim=1)
            correct += (out==label).sum().cpu().item()
            
            #batch_pbar.set_description(f'[eval] [batch: {i+1:>5}/{len(loader)}] acc: {(out==label).sum().item()/out.shape[0]:.4f}')
            
        avg_loss = loss/len(loader)
        avg_acc = correct/len(loader.dataset)
    return avg_loss, avg_acc


# In[7]:


def train(model, device, train_loader, test_loader, optimizer, criterion, epoch_num, logger_path,model_path):
    #epoch_pbar = tqdm(range(1, epoch_num+1))
    train_acc_list = []
    test_acc_list = []
    train_loss_list = []
    test_loss_list = []
    for epoch in range(1, epoch_num+1):
        model.train()
        #batch_pbar = tqdm(train_loader)
        for i, (data,label) in enumerate(train_loader):
            data, label = data.to(device), label.to(device)
            optimizer.zero_grad()
            out = model.forward(data)
            loss = criterion(out,label)
            loss.backward()
            optimizer.step()
            #batch_pbar.set_description(f'[train] [epoch:{epoch:>4}/{epoch_num}] [batch: {i+1:>5}/{len(train_loader)}] loss: {loss.item():.4f}')
            
        torch.save(model.state_dict(), os.path.join(model_path, f'epoch{epoch}.cpt'))
        
        train_loss, train_acc = evaluate(model,device,train_loader,criterion)
        test_loss, test_acc = evaluate(model,device,test_loader,criterion)
        train_acc_list.append(train_acc)
        train_loss_list.append(train_loss)
        test_acc_list.append(test_acc)
        test_loss_list.append(test_loss)
        '''
        logger.add_scalar('train/loss', train_loss, epoch)
        logger.add_scalar('train/acc', train_acc, epoch)
        logger.add_scalar('test/loss', test_loss, epoch)
        logger.add_scalar('test/acc', test_acc, epoch)
        '''
        print(f"[train] [epoch:{epoch:>4}/{epoch_num}] train acc: {train_acc:.4f}, test acc: {test_acc:.4f}")
        #epoch_pbar.set_description(f'[train] [epoch:{epoch:>4}/{epoch_num}] train acc: {train_acc:.4f}, test acc: {test_acc:.4f}')
    result = {'train_acc': train_acc_list,
              'train_loss': train_loss_list,
              'test_acc': test_acc_list,
              'test_loss': test_loss_list}
    with open(f'{getTime()}.pickle', 'wb') as f:
        pickle.dump(result, f)

# In[ ]:
if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument("--model",        type=str,            default="ResNet18")
    parser.add_argument("--batch_size",   type=int,            default=16)
    parser.add_argument("--epoch_num",    type=int,            default=10)
    parser.add_argument("--pretrained",   action="store_true", default=False)
    parser.add_argument("--log_dir",      type=str,            default='logs')
    parser.add_argument("--gpu_index",    type=str,            default="1")
    parser.add_argument("--data_dir",     type=str,            default="data/")
    parser.add_argument("--model_dir",    type=str,            default="models")
    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_index
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    batch_size = args.batch_size
    train_kwargs = {'batch_size': batch_size}
    test_kwargs = {'batch_size': 1}

    if use_cuda:
        print("Use GPU for training...")
        cuda_kwargs = {'num_workers': 32,
                       'pin_memory': True,
                       'shuffle': False}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)
    else:
        print("Use CPU to training...")

    
    train_dataset = RetinopathyDataset(args.data_dir, 'train')
    test_dataset = RetinopathyDataset(args.data_dir, 'test')

    train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)
    model_name = ''
    if args.model == "ResNet18":
        if args.pretrained:
            model = models.resnet18(pretrained=True)
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features,5)
            model_name = 'resnet18_pre'
        else:
            model = ResNet18()
            model_name = 'resnet18'
    elif args.model == "ResNet50":
        if args.pretrained:
            model = models.resnet50(pretrained=True)
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features,5)
            model_name = 'resnet50_pre'
        else:
            model = ResNet50()
            model_name = 'resnet50'
    else:
        print("Unexpected model name")
        exit(1)

    model.to(device)
#optimizer = torch.optim.SGD(model.parameters(),lr=1e-3,momentum=0.9,weight_decay=0)
    optimizer = torch.optim.RAdam(model.parameters(),lr=1e-3,weight_decay=0)

    #writer = SummaryWriter(os.path.join("logs","ResNet18"))

    train(
        model = model,
        device = device,
        train_loader = train_loader,
        test_loader = test_loader,
        optimizer = optimizer,
        criterion = nn.CrossEntropyLoss(),
        epoch_num = 10,
        logger_path = os.path.join(args.log_dir,model_name),
        model_path = os.path.join(args.model_dir,model_name)
    )


# In[ ]:




