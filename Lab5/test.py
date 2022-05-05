import argparse
import os
import random
import imageio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
import pickle
from dataset import bair_robot_pushing_dataset
from models.lstm import gaussian_lstm, lstm
from models.vgg_64 import vgg_decoder, vgg_encoder
from utils import init_weights, kl_criterion, plot_pred, finn_eval_seq, tensor_to_img

torch.backends.cudnn.benchmark = True

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=0.002, type=float, help='learning rate')
    parser.add_argument('--beta1', default=0.9, type=float, help='momentum term for adam')
    parser.add_argument('--batch_size', default=16, type=int, help='batch size')
    parser.add_argument('--log_dir', default='./logs/final', help='base directory to save logs')
    parser.add_argument('--model_dir', default='', help='base directory to save logs')
    parser.add_argument('--data_root', default='./data', help='root directory for data')
    parser.add_argument('--optimizer', default='adam', help='optimizer to train with')
    parser.add_argument('--niter', type=int, default=1, help='number of epochs to train for')
    parser.add_argument('--epoch_size', type=int, default=600, help='epoch size')
    parser.add_argument('--tfr', type=float, default=1.0, help='teacher forcing ratio (0 ~ 1)')
    parser.add_argument('--tfr_start_decay_epoch', type=int, default=200, help='The epoch that teacher forcing ratio become decreasing')
    parser.add_argument('--tfr_decay_step', type=float, default=0, help='The decay step size of teacher forcing ratio (0 ~ 1)')
    parser.add_argument('--tfr_lower_bound', type=float, default=0.8, help='The lower bound of teacher forcing ratio for scheduling teacher forcing ratio (0 ~ 1)')
    parser.add_argument('--kl_anneal_cyclical', default=False, action='store_true', help='use cyclical mode')
    parser.add_argument('--kl_anneal_ratio', type=float, default=2, help='The decay ratio of kl annealing')
    parser.add_argument('--kl_anneal_cycle', type=int, default=3, help='The number of cycle for kl annealing (if use cyclical mode)')
    parser.add_argument('--seed', default=1, type=int, help='manual seed')
    parser.add_argument('--n_past', type=int, default=2, help='number of frames to condition on')
    parser.add_argument('--n_future', type=int, default=10, help='number of frames to predict')
    parser.add_argument('--n_eval', type=int, default=12, help='number of frames to predict at eval time')
    parser.add_argument('--rnn_size', type=int, default=256, help='dimensionality of hidden layer')
    parser.add_argument('--posterior_rnn_layers', type=int, default=1, help='number of layers')
    parser.add_argument('--predictor_rnn_layers', type=int, default=2, help='number of layers')
    parser.add_argument('--z_dim', type=int, default=64, help='dimensionality of z_t')
    parser.add_argument('--g_dim', type=int, default=128, help='dimensionality of encoder output vector and decoder input vector')
    parser.add_argument('--beta', type=float, default=0.0001, help='weighting on KL to prior')
    parser.add_argument('--num_workers', type=int, default=32, help='number of data loading threads')
    parser.add_argument('--last_frame_skip', action='store_true', help='if true, skip connections go between frame t and frame t+t rather than last ground truth frame')
    parser.add_argument('--cuda', default=True, action='store_true')  
    parser.add_argument('--gpu_index', type=str, default="1", help="gpu index number")  
    parser.add_argument('--name', type=str)

    args = parser.parse_args()
    return args

def pred(validate_seq, validate_cond, modules, args, device):
    modules['frame_predictor'].eval()
    modules['posterior'].eval()
    #modules['prior'].eval()
    modules['encoder'].eval()
    modules['decoder'].eval()
    mse_criterion = nn.MSELoss()
    
    mse = 0
    kld = 0
    img_seq = validate_seq[:args.n_past]
    with torch.no_grad():
        for i in range(1, args.n_past + args.n_future):
            if i-1 < args.n_past:
                h = modules['encoder'](validate_seq[i-1])
            else:
                h = modules['encoder'](img_seq[i-1])

            if i < args.n_past:
                h_target = modules['encoder'](validate_seq[i])[0]

            if args.last_frame_skip or i < args.n_past:        
                h, skip = h
            else:
                h = h[0]

            if i < args.n_past:
                z_t, mu, logvar = modules['posterior'](h_target)#gaussian_lstm
                h_pred = modules['frame_predictor'](torch.cat([h, z_t, validate_cond[i]], 1))#lstm
                x_pred = modules['decoder']([h_pred, skip])
                #mse += mse_criterion(x_pred, validate_seq[i])
                #kld += kl_criterion(mu, logvar,args)
            
            else:
                z_t = torch.from_numpy(np.random.normal(0,1,size=(args.batch_size,64))).float().to(device)
                h_pred = modules['frame_predictor'](torch.cat([h, z_t, validate_cond[i]], 1))#lstm
                x_pred = modules['decoder']([h_pred, skip])
                #mse += mse_criterion(x_pred, validate_seq[i])
                img_seq.append(x_pred)

    return img_seq


def sequence_input(seq, dtype):
    return [Variable(x.type(dtype)) for x in seq]

def get_batch(data_loader):
    while True:
        for sequence in data_loader:
            sequence[0].transpose_(0,1)
            sequence[1].transpose_(0,1)
            yield sequence_input(sequence[0],torch.cuda.FloatTensor), sequence_input(sequence[1],torch.cuda.FloatTensor)

def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    if args.cuda:
        assert torch.cuda.is_available(), 'CUDA is not available.'
        device = torch.device('cuda')
        print("Testing with GPU...")
    else:
        device = torch.device('cpu')
        print("Testing with CPU...")
    
    
    assert args.n_past + args.n_future <= 30 and args.n_eval <= 30
    assert 0 <= args.tfr and args.tfr <= 1
    assert 0 <= args.tfr_start_decay_epoch 
    assert 0 <= args.tfr_decay_step and args.tfr_decay_step <= 1

    if args.model_dir != '':
        # load model and continue training from checkpoint
        saved_model = torch.load('%s/model.pth' % args.model_dir)
        optimizer = args.optimizer
        model_dir = args.model_dir
        niter = args.niter
        args = saved_model['args']
        args.optimizer = optimizer
        args.model_dir = model_dir
        #args.log_dir = '%s/continued' % args.log_dir
        start_epoch = saved_model['last_epoch']
    else:
        print("Error: empty model name!")
        exit(-1)

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs('%s/test/' % args.log_dir, exist_ok=True)

    print("Random Seed: ", args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    

    if os.path.exists('./{}/test_record.txt'.format(args.log_dir)):
        os.remove('./{}/test_record.txt'.format(args.log_dir))
    
    print(args)

    with open('./{}/test_record.txt'.format(args.log_dir), 'a') as test_record:
        test_record.write('args: {}\n'.format(args))

    # ------------ build the models  --------------

    if args.model_dir != '':
        frame_predictor = saved_model['frame_predictor']
        posterior = saved_model['posterior']
        #prior = saved_model['prior']
    else:
        frame_predictor = lstm(args.g_dim+args.z_dim+7, args.g_dim, args.rnn_size, args.predictor_rnn_layers, args.batch_size, device)
        posterior = gaussian_lstm(args.g_dim, args.z_dim, args.rnn_size, args.posterior_rnn_layers, args.batch_size, device)
        #prior = gaussian_lstm(args.g_dim, args.z_dim, args.rnn_size, args.posterior_rnn_layers, args.batch_size, device)
        frame_predictor.apply(init_weights)
        posterior.apply(init_weights)
        #prior.apply(init_weights)
            
    if args.model_dir != '':
        decoder = saved_model['decoder']
        encoder = saved_model['encoder']
    else:
        encoder = vgg_encoder(args.g_dim)
        decoder = vgg_decoder(args.g_dim)
        encoder.apply(init_weights)
        decoder.apply(init_weights)
    
    # --------- transfer to device ------------------------------------
    frame_predictor.to(device)
    posterior.to(device)
    #prior.to(device)
    encoder.to(device)
    decoder.to(device)

    # --------- load a dataset ------------------------------------
    test_data = bair_robot_pushing_dataset(args, 'test')
    test_loader = DataLoader(test_data,
                            num_workers=args.num_workers,
                            batch_size=args.batch_size,
                            shuffle=False,
                            drop_last=True,
                            pin_memory=True)
    test_iter = get_batch(test_loader)

    # ---------------- optimizers ----------------
    if args.optimizer == 'adam':
        args.optimizer = optim.Adam
    elif args.optimizer == 'rmsprop':
        args.optimizer = optim.RMSprop
    elif args.optimizer == 'sgd':
        args.optimizer = optim.SGD
    else:
        raise ValueError('Unknown optimizer: %s' % args.optimizer)

    params = list(frame_predictor.parameters()) + list(posterior.parameters()) + list(encoder.parameters()) + list(decoder.parameters()) #+ list(prior.parameters())
    #optimizer = args.optimizer(params, lr=args.lr, betas=(args.beta1, 0.999))
    optimizer = args.optimizer(params, lr=args.lr)

    modules = {
        'frame_predictor': frame_predictor,
        'posterior': posterior,
        'encoder': encoder,
        'decoder': decoder,
        #'prior':prior,
    }

    # --------- testing loop ------------------------------------

    progress = tqdm(total=args.niter)
    psnr_list = []
    for epoch in range(start_epoch, start_epoch + niter):

        progress.update(1)
        
        frame_predictor.eval()
        encoder.eval()
        decoder.eval()
        #prior.eval()
        posterior.eval()
        ave_psnr = 0
        
        for n in range(len(test_data) // args.batch_size):
            try:
                test_seq, test_cond = next(test_iter)
            except StopIteration:
                test_iter = get_batch(test_loader)
                test_seq, test_cond = next(test_iter)

            pred_seq = pred(test_seq, test_cond, modules, args, device)
            for i in range(args.batch_size):
                gif = []
                for img,gt in zip(pred_seq,test_seq):
                    gif.append(np.concatenate([tensor_to_img(gt[i]),tensor_to_img(img[i])],axis=1))
                fname = '%s/test/pred_%d.gif' % (args.log_dir, n*args.batch_size+i)
                imageio.mimsave(fname,gif,duration=0.5)

            _, _, psnr = finn_eval_seq(test_seq[args.n_past:], pred_seq[args.n_past-1:])
            psnr_list.append(np.mean(np.concatenate(psnr)))
            for i in range(args.batch_size):
                with open('./{}/test_record.txt'.format(args.log_dir), 'a') as test_record:
                    test_record.write(('====================== psnr of image {}= {:.5f} ========================\n'.format(n*args.batch_size+i,np.mean(psnr[i]))))
            
        ave_psnr = np.mean(psnr_list)
        
        with open('./{}/test_record.txt'.format(args.log_dir), 'a') as test_record:
                test_record.write(('====================== average psnr = {:.5f} ========================\n'.format(ave_psnr)))
if __name__ == '__main__':
    main()
        
