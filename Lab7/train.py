# -*- coding: utf-8 -*-
import os
import random
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from evaluator import evaluation_model
from dataset import CLEVRDataset
from dc_gan import Generator, Discriminator, weights_init

def evaluate(g_model, loader, eval_model, n_z, device):
    g_model.eval()
    avg_acc = 0
    gen_images = None
    with torch.no_grad():
        for _, conds in enumerate(loader):
            conds = conds.to(device)
            batch = conds.shape[0]
            #z = sample_z(conds.shape[0], n_z).to(device)
            noise = torch.randn(batch, n_z, 1, 1, device=device)
            fake_images = g_model(noise, conds)
            if gen_images is None:
                gen_images = fake_images
            else:
                gen_images = torch.vstack((gen_images, fake_images))
            acc = eval_model.eval(fake_images, conds)
            avg_acc += acc * batch
    avg_acc /= len(loader.dataset)
    return avg_acc, gen_images


def train( netG, netD, optimizerG, optimizerD, criterion, train_loader,
            test_loader, n_z, num_epochs, eval_interval, log_dir, model_dir, result_dir,device):
    
    logger = SummaryWriter(log_dir)
    # Establish convention for real and fake labels during training
    real_label = 1.
    fake_label = 0.
    
    

    eval_model = evaluation_model()
    best_acc = 0
    batch_done = 0
    pbar_epoch = tqdm(range(num_epochs))
    for epoch in pbar_epoch:
        netG.train()
        netD.train()
        losses_g = 0
        losses_d = 0
        pbar_batch = tqdm(train_loader)
        for batch_idx, (images, conds) in enumerate(pbar_batch):
            images = images.to(device)
            conds = conds.to(device)
            batch = images.shape[0]
            
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            
            netD.zero_grad()
            label = torch.full((batch,), real_label, dtype=torch.float, device=device)
            output = netD(images, conds).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(batch, n_z, 1, 1, device=device)
            # Generate fake image batch with G
            fake = netG(noise, conds)
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = netD(fake.detach(), conds).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            errD_fake.backward()
            #if losses_g < losses_d :
            # Update D
            if batch_idx % 2:
                optimizerD.step()
            D_G_z1 = output.mean().item()
            # Compute error of D as sum over the fake and the real batches
            errD = errD_real + errD_fake
            
            
            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            # fake labels are real for generator cost
            label.fill_(real_label)
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake, conds).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            #if losses_g > losses_d :
                # Update G
            optimizerG.step()
            D_G_z2 = output.mean().item()
            pbar_batch.set_description('[{}/{}][{}/{}][Loss_D={:.4f}][Loss_G={:.4f}][D(x)={:.4f}][D(G(z))={:.4f}/{:.4f}]'
                .format(epoch+1, num_epochs, batch_idx+1, len(train_loader), errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            losses_g += errG.item()
            losses_d += errD.item()

            batch_done += 1

            if batch_done%eval_interval == 0:
                eval_acc, gen_images = evaluate(netG, test_loader, eval_model, n_z, device)
                gen_images = 0.5*gen_images + 0.5
                logger.add_scalar('batch/eval_acc', eval_acc, batch_done)
                if eval_acc > best_acc:
                    best_acc = eval_acc
                    torch.save(
                        netG.state_dict(),
                        os.path.join(model_dir, f'epoch{epoch+1}_iter{batch_done}_eval-acc{eval_acc:.4f}.cpt'))
                #save_image(gen_images, os.path.join(result_dir, f'epoch{epoch+1}_iter{batch_done}.png'), nrow=8)
                #save_image(gen_images, 'gan_current.png', nrow=8)
                netG.train()
                #netD.train()

        avg_loss_g = losses_g / len(train_loader)
        avg_loss_d = losses_d / len(train_loader)
        eval_acc, gen_images = evaluate(netG, test_loader, eval_model, n_z, device)
        gen_images = 0.5*gen_images + 0.5
        pbar_epoch.set_description('[{}/{}][AvgLossD={:.4f}][AvgLossG={:.4f}][EvalAcc={:.4f}]'
            .format(epoch+1, num_epochs, avg_loss_d, avg_loss_g, eval_acc))
        
        logger.add_scalar('epoch/loss_g', avg_loss_g, epoch+1)
        logger.add_scalar('epoch/loss_d', avg_loss_d, epoch+1)
        logger.add_scalar('epoch/eval_acc', eval_acc, epoch+1)
        if eval_acc > best_acc:
            best_acc = eval_acc
            torch.save(
                netG.state_dict(),
                os.path.join(model_dir, f'epoch{epoch+1}_last_eval-acc{eval_acc:.4f}.cpt'))
        save_image(gen_images, os.path.join(result_dir, f'epoch{epoch+1}_last.png'), nrow=8)
        #save_image(gen_images, 'gan_current.png', nrow=8)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--nz', type=int, default=100)
    parser.add_argument('--num_conditions', type=int, default=24)
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--ndf', type=int, default=64)
    parser.add_argument('--img_size', type=int, default=64)
    #parser.add_argument('--add_bias', action='store_true', default=False)
    parser.add_argument('--num_epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--eval_interval', type=int, default=50)
    parser.add_argument('--num_workers', type=int, help='number of data loading workers', default=32)
    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('--model_dir', type=str, default='models')
    parser.add_argument('--result_dir', type=str, default='results')
    parser.add_argument('--dataset_dir',type=str, default='/DATA/2022DLP/LAB7')
    parser.add_argument('--json_dir',type=str, default='./')
    parser.add_argument('--seed',type=int, default=87)
    parser.add_argument('--task_name',type=str,default="test")
    parser.add_argument('--gpu_index',type=str,default='1')

    args = parser.parse_args()

    #fixed random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.manual_seed(args.seed)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_index
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("Training with GPU...")
    else:
        device = torch.device('cpu')
        print("Training with CPU...")
    
    
    model_dir = os.path.join(args.model_dir, args.task_name)
    log_dir = os.path.join(args.log_dir, args.task_name)
    result_dir = os.path.join(args.result_dir, args.task_name)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)


    train_trans = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.CenterCrop((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_dataset = CLEVRDataset(args.json_dir, args.dataset_dir, train_trans, mode='train', train_json='train.json',obj_json='objects.json')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    test_dataset = CLEVRDataset(args.json_dir, args.dataset_dir, None, mode='test', test_json='test.json',obj_json='objects.json')
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Apply the weights_init function to randomly initialize all weights
    # to mean=0, stdev=0.02.
    netG = Generator(args).to(device)
    netG.apply(weights_init)

    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.
    netD = Discriminator(args).to(device)
    netD.apply(weights_init)

    criterion = nn.BCELoss()
    optimizerG = optim.Adam(netG.parameters(), args.lr, betas=(args.beta1, args.beta2))
    optimizerD = optim.Adam(netD.parameters(), args.lr, betas=(args.beta1, args.beta2))
    #optimizerD = optim.SGD(netD.parameters(), args.lr)

    train(
        netG=netG,
        netD=netD,
        optimizerG=optimizerG,
        optimizerD=optimizerD,
        criterion=criterion,
        train_loader=train_loader,
        test_loader=test_loader,
        n_z=args.nz,
        num_epochs=args.num_epochs,
        eval_interval=args.eval_interval,
        log_dir=log_dir,
        model_dir=model_dir,
        result_dir=result_dir,
        device=device)