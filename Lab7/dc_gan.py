# -*- coding: utf-8 -*-
import torch
import torch.nn as nn


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Generator(nn.Module):
    def __init__(self, args):
        super(Generator, self).__init__()
        self.nz = args.nz
        self.embed_c= nn.Sequential(
            nn.Linear(args.num_conditions, args.nz),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.main = nn.Sequential(
            # input is Z+condition, going into a convolution
            nn.ConvTranspose2d( args.nz * 2, args.ngf * 8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(args.ngf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(args.ngf * 8, args.ngf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(args.ngf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( args.ngf * 4, args.ngf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(args.ngf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( args.ngf * 2, args.ngf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(args.ngf),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( args.ngf, 3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )
    
    def forward(self, z, cond):
        z = z.reshape(-1, self.nz, 1, 1)
        embd_cond = self.embed_c(cond).reshape(-1, self.nz, 1, 1)
        #shape: [128, 100, 1, 1] => [128, 200, 1, 1]
        return self.main(torch.cat((z, embd_cond), dim=1))


class Discriminator(nn.Module):
    def __init__(self, args):
        super(Discriminator, self).__init__()
        self.img_size = args.img_size

        self.embed_c= nn.Sequential(
            nn.Linear(args.num_conditions, args.img_size*args.img_size),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(4, args.ndf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(args.ndf, args.ndf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(args.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(args.ndf * 2, args.ndf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(args.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(args.ndf * 4, args.ndf * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(args.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(args.ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
        
        

    def forward(self, img, cond):
        embd_cond = self.embed_c(cond).reshape(-1, 1, self.img_size, self.img_size)#reshape to [1 x 1 x 64 x 64]
        #shape [1 x 3 x 64 x 64] + [1 x 1 x 64 x 64] = [1 x 4 x 64 x 64]
        return self.main(torch.cat((img, embd_cond), dim=1)).reshape(-1)