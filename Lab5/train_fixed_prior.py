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
    parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    parser.add_argument('--log_dir', default='./logs/final', help='base directory to save logs')
    parser.add_argument('--model_dir', default='', help='base directory to save logs')
    parser.add_argument('--data_root', default='./data', help='root directory for data')
    parser.add_argument('--optimizer', default='adam', help='optimizer to train with')
    parser.add_argument('--niter', type=int, default=300, help='number of epochs to train for')
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

def train(x, cond, modules, optimizer, kl_anneal, args, epoch,device):
    modules['frame_predictor'].zero_grad()
    modules['posterior'].zero_grad()
    #modules['prior'].zero_grad()
    modules['encoder'].zero_grad()
    modules['decoder'].zero_grad()
    mse_criterion = nn.MSELoss()
    

    # initialize the hidden state.
    modules['frame_predictor'].hidden = modules['frame_predictor'].init_hidden()
    modules['posterior'].hidden = modules['posterior'].init_hidden()
    #modules['prior'].hidden = modules['prior'].init_hidden()
    mse = 0
    kld = 0
    use_teacher_forcing = True if random.random() < args.tfr else False
    pred = x[:args.n_past]
    for i in range(1, args.n_past + args.n_future):
        if i-1 < args.n_past or use_teacher_forcing:
            h = modules['encoder'](x[i-1])
        else:
            h = modules['encoder'](pred[i-1])

        if i < args.n_past or use_teacher_forcing:
            h_target = modules['encoder'](x[i])[0]
        #else:
        #    h_target = modules['encoder'](pred[i-args.n_past],cond[i])[0]

        if args.last_frame_skip or i < args.n_past:        
            h, skip = h
        else:
            h = h[0]

        if i < args.n_past or use_teacher_forcing:
            z_t, mu, logvar = modules['posterior'](h_target)#gaussian_lstm
            h_pred = modules['frame_predictor'](torch.cat([h, z_t, cond[i]], 1))#lstm
            x_pred = modules['decoder']([h_pred, skip])
            mse += mse_criterion(x_pred, x[i])
            kld += kl_criterion(mu, logvar,args)
        else:
            z_t = torch.from_numpy(np.random.normal(0,1,size=(args.batch_size,64))).float().to(device)
            h_pred = modules['frame_predictor'](torch.cat([h, z_t, cond[i]], 1))#lstm
            x_pred = modules['decoder']([h_pred, skip])
            mse += mse_criterion(x_pred, x[i])
            pred.append(x_pred)
            


    beta = kl_anneal.get_beta(epoch)
    loss = mse + kld * beta
    #loss.backward(retain_graph=True)
    loss.backward()

    optimizer.step()

    return loss.detach().cpu().numpy() / (args.n_past + args.n_future), mse.detach().cpu().numpy() / (args.n_past + args.n_future), kld.detach().cpu().numpy() / (args.n_future + args.n_past)

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

class kl_annealing():
    def __init__(self, args):
        super().__init__()
        self.cyclical_flag = args.kl_anneal_cyclical
        self.ratio = args.kl_anneal_ratio
        self.cycle = args.kl_anneal_cycle
        self.period = args.niter // self.cycle
        self.thres = args.niter // 10
        self.slope = 1.0 / self.thres
        self.c_slope = 1.0 / (self.period // 2)
    
    def get_beta(self,epoch):
        if self.cyclical_flag:
            epoch %= self.period
            return min(self.c_slope*epoch, 1.0)
        else:
            return min(self.slope*epoch, 1.0)




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
        print("Training with GPU...")
    else:
        device = torch.device('cpu')
        print("Training with CPU...")
    
    
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
        args.log_dir = '%s/continued' % args.log_dir
        start_epoch = saved_model['last_epoch']
    else:
        #name = 'rnn_size=%d-predictor-posterior-rnn_layers=%d-%d-n_past=%d-n_future=%d-lr=%.4f-g_dim=%d-z_dim=%d-last_frame_skip=%s-beta=%.7f'\
        #    % (args.rnn_size, args.predictor_rnn_layers, args.posterior_rnn_layers, args.n_past, args.n_future, args.lr, args.g_dim, args.z_dim, args.last_frame_skip, args.beta)
        name = args.name
        args.log_dir = '%s/%s' % (args.log_dir, name)
        niter = args.niter
        start_epoch = 0

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs('%s/gen/' % args.log_dir, exist_ok=True)

    print("Random Seed: ", args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    

    if os.path.exists('./{}/train_record.txt'.format(args.log_dir)):
        os.remove('./{}/train_record.txt'.format(args.log_dir))
    
    print(args)

    with open('./{}/train_record.txt'.format(args.log_dir), 'a') as train_record:
        train_record.write('args: {}\n'.format(args))

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
    train_data = bair_robot_pushing_dataset(args, 'train')
    validate_data = bair_robot_pushing_dataset(args, 'test')
    
    train_loader = DataLoader(train_data,
                            num_workers=args.num_workers,
                            batch_size=args.batch_size,
                            shuffle=True,
                            drop_last=True,
                            pin_memory=True)
    train_iter = get_batch(train_loader)

    validate_loader = DataLoader(validate_data,
                            num_workers=args.num_workers,
                            batch_size=args.batch_size,
                            shuffle=True,
                            drop_last=True,
                            pin_memory=True)
    validate_iter = get_batch(validate_loader)

    # ---------------- optimizers ----------------
    if args.optimizer == 'adam':
        args.optimizer = optim.Adam
    elif args.optimizer == 'rmsprop':
        args.optimizer = optim.RMSprop
    elif args.optimizer == 'radam':
        args.optimizer = optim.RAdam
    elif args.optimizer == 'sgd':
        args.optimizer = optim.SGD
    else:
        raise ValueError('Unknown optimizer: %s' % args.optimizer)

    params = list(frame_predictor.parameters()) + list(posterior.parameters()) + list(encoder.parameters()) + list(decoder.parameters()) #+ list(prior.parameters())
    #optimizer = args.optimizer(params, lr=args.lr, betas=(args.beta1, 0.999))
    optimizer = args.optimizer(params, lr=args.lr)
    kl_anneal = kl_annealing(args)

    modules = {
        'frame_predictor': frame_predictor,
        'posterior': posterior,
        'encoder': encoder,
        'decoder': decoder,
        #'prior':prior,
    }

    epoch_list = []
    kld_list = []
    mse_list = []
    beta_list = []
    tf_ratio_list = []
    loss_list = []
    psnr_list = []

    record = {
        'epoch': epoch_list,
        'kld': kld_list,
        'mse': mse_list,
        'beta': beta_list,
        'tf_ratio': tf_ratio_list,
        'loss': loss_list,
        'psnr': psnr_list
    }
    # --------- training loop ------------------------------------

    progress = tqdm(total=args.niter)
    best_val_psnr = 0
    for epoch in range(start_epoch, start_epoch + niter):
        frame_predictor.train()
        posterior.train()
        #prior.train()
        encoder.train()
        decoder.train()

        epoch_loss = 0
        epoch_mse = 0
        epoch_kld = 0

        for _ in range(args.epoch_size):
            try:
                seq, cond = next(train_iter)
            except StopIteration:
                train_iter = get_batch(train_loader)
                seq, cond = next(train_iter)

            loss, mse, kld = train(seq, cond, modules, optimizer, kl_anneal, args,epoch,device)
            epoch_loss += loss
            epoch_mse += mse
            epoch_kld += kld

        record['epoch'].append(epoch)
        record['kld'].append(epoch_kld/ args.epoch_size)
        record['mse'].append(epoch_mse/ args.epoch_size)
        record['beta'].append(kl_anneal.get_beta(epoch))
        record['tf_ratio'].append(args.tfr)
        record['loss'].append(epoch_loss/ args.epoch_size)

        if epoch >= args.tfr_start_decay_epoch:
            args.tfr = max(args.tfr - args.tfr_decay_step,args.tfr_lower_bound)

        progress.update(1)
        with open('./{}/train_record.txt'.format(args.log_dir), 'a') as train_record:
            train_record.write(('[epoch: %02d] loss: %.5f | mse loss: %.5f | kld loss: %.5f\n' % (epoch, epoch_loss  / args.epoch_size, epoch_mse / args.epoch_size, epoch_kld / args.epoch_size)))
        
        frame_predictor.eval()
        encoder.eval()
        decoder.eval()
        #prior.eval()
        posterior.eval()
        ave_psnr = 0
        #if epoch % 5 == 0:
        psnr_list = []
        for _ in range(len(validate_data) // args.batch_size):
            try:
                validate_seq, validate_cond = next(validate_iter)
            except StopIteration:
                validate_iter = get_batch(validate_loader)
                validate_seq, validate_cond = next(validate_iter)

            pred_seq = pred(validate_seq, validate_cond, modules, args, device)
            _, _, psnr = finn_eval_seq(validate_seq[args.n_past:], pred_seq[args.n_past-1:])
            psnr_list.append(psnr)
            
        ave_psnr = np.mean(np.concatenate(psnr))


        with open('./{}/train_record.txt'.format(args.log_dir), 'a') as train_record:
            train_record.write(('====================== validate psnr = {:.5f} ========================\n'.format(ave_psnr)))

        if ave_psnr > best_val_psnr:
            best_val_psnr = ave_psnr
            # save the model
            torch.save({
                'encoder': encoder,
                'decoder': decoder,
                'frame_predictor': frame_predictor,
                'posterior': posterior,
                #'prior': prior,
                'args': args,
                'last_epoch': epoch},
                '%s/model.pth' % args.log_dir)
        record['psnr'].append(ave_psnr)
        if epoch % 10 == 0:
            try:
                validate_seq, validate_cond = next(validate_iter)
            except StopIteration:
                validate_iter = get_batch(validate_loader)
                validate_seq, validate_cond = next(validate_iter)

            plot_pred(validate_seq, validate_cond, modules, epoch, args,device)
            fname = '%s/gen/pred_%d.gif' % (args.log_dir, epoch)
            img_seq = pred(validate_seq, validate_cond, modules, args, device)
            gif = []
            for img,gt in zip(img_seq,validate_seq):
                gif.append(np.concatenate([tensor_to_img(gt[0]),tensor_to_img(img[0])],axis=1))
            
            imageio.mimsave(fname,gif,duration=0.5)

        with open(os.path.join(args.log_dir,"record.pickle"),'wb') as f:
            pickle.dump(record,f)
if __name__ == '__main__':
    main()
        
