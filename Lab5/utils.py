import math
from operator import pos
import imageio
import numpy as np
import torch
from scipy import signal
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric

def kl_criterion(mu, logvar, args):
  # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
  KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
  KLD /= args.batch_size  
  return KLD
    
def eval_seq(gt, pred):
    T = len(gt)
    bs = gt[0].shape[0]
    ssim = np.zeros((bs, T))
    psnr = np.zeros((bs, T))
    mse = np.zeros((bs, T))
    for i in range(bs):
        for t in range(T):
            origin = gt[t][i]
            predict = pred[t][i]
            for c in range(origin.shape[0]):
                ssim[i, t] += ssim_metric(origin[c], predict[c]) 
                psnr[i, t] += psnr_metric(origin[c], predict[c])
            ssim[i, t] /= origin.shape[0]
            psnr[i, t] /= origin.shape[0]
            mse[i, t] = mse_metric(origin, predict)

    return mse, ssim, psnr

def mse_metric(x1, x2):
    err = np.sum((x1 - x2) ** 2)
    err /= float(x1.shape[0] * x1.shape[1] * x1.shape[2])
    return err

# ssim function used in Babaeizadeh et al. (2017), Fin et al. (2016), etc.
def finn_eval_seq(gt, pred):
    T = len(gt)
    bs = gt[0].shape[0]
    ssim = np.zeros((bs, T))
    psnr = np.zeros((bs, T))
    mse = np.zeros((bs, T))
    for i in range(bs):
        for t in range(T):
            origin = gt[t][i].detach().cpu().numpy()
            predict = pred[t][i].detach().cpu().numpy()
            for c in range(origin.shape[0]):
                res = finn_ssim(origin[c], predict[c]).mean()
                if math.isnan(res):
                    ssim[i, t] += -1
                else:
                    ssim[i, t] += res
                psnr[i, t] += finn_psnr(origin[c], predict[c])
            ssim[i, t] /= origin.shape[0]
            psnr[i, t] /= origin.shape[0]
            mse[i, t] = mse_metric(origin, predict)

    return mse, ssim, psnr

def finn_psnr(x, y, data_range=1.):
    mse = ((x - y)**2).mean()
    return 20 * math.log10(data_range) - 10 * math.log10(mse)

def fspecial_gauss(size, sigma):
    x, y = np.mgrid[-size // 2 + 1:size // 2 + 1, -size // 2 + 1:size // 2 + 1]
    g = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
    return g / g.sum()

def finn_ssim(img1, img2, data_range=1., cs_map=False):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    size = 11
    sigma = 1.5
    window = fspecial_gauss(size, sigma)

    K1 = 0.01
    K2 = 0.03

    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2
    mu1 = signal.fftconvolve(img1, window, mode='valid')
    mu2 = signal.fftconvolve(img2, window, mode='valid')
    mu1_sq = mu1*mu1
    mu2_sq = mu2*mu2
    mu1_mu2 = mu1*mu2
    sigma1_sq = signal.fftconvolve(img1*img1, window, mode='valid') - mu1_sq
    sigma2_sq = signal.fftconvolve(img2*img2, window, mode='valid') - mu2_sq
    sigma12 = signal.fftconvolve(img1*img2, window, mode='valid') - mu1_mu2

    if cs_map:
        return (((2 * mu1_mu2 + C1) * (2 * sigma12 + C2))/((mu1_sq + mu2_sq + C1) *
                    (sigma1_sq + sigma2_sq + C2)), 
                (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2))
    else:
        return ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                    (sigma1_sq + sigma2_sq + C2))

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def is_sequence(arg):
    return (not hasattr(arg, "strip") and
            not type(arg) is np.ndarray and
            not hasattr(arg, "dot") and
            (hasattr(arg, "__getitem__") or
            hasattr(arg, "__iter__")))

def plot_pred(validate_seq, validate_cond, modules, epoch, args, device):
    #shape of validate_seq: ((n_past+n_future) x batch_size x 3 x 64 x 64)
    modules['frame_predictor'].eval()
    modules['posterior'].eval()
    #modules['prior'].eval()
    modules['encoder'].eval()
    modules['decoder'].eval()
    nsample = 20 
    gen_seq = [[] for i in range(nsample)]
    gt_seq = [validate_seq[i] for i in range(len(validate_seq))]

    for s in range(nsample):
        modules['frame_predictor'].eval()
        modules['posterior'].eval()
        #modules['prior'].eval()
        modules['encoder'].eval()
        modules['decoder'].eval()
        
        gen_seq[s].append(validate_seq[0])
        x_in = validate_seq[0]
        for i in range(1, args.n_eval):
            h = modules['encoder'](x_in)
            if args.last_frame_skip or i < args.n_past:	
                h, skip = h
            else:
                h, _ = h
            h = h.detach()
            if i < args.n_past:
                h_target = modules['encoder'](validate_seq[i])
                h_target = h_target[0].detach()
                z_t, _, _ = modules['posterior'](h_target)
                modules['frame_predictor'](torch.cat([h, z_t,validate_cond[i]], 1))
                x_in = validate_seq[i]
                gen_seq[s].append(x_in)
            else:
                #z_t, _, _ = modules['prior'](h)
                z_t = torch.from_numpy(np.random.normal(0,1,size=(args.batch_size,64))).float().to(device)
                h = modules['frame_predictor'](torch.cat([h, z_t,validate_cond[i]], 1)).detach()
                x_in = modules['decoder']([h, skip]).detach()
                gen_seq[s].append(x_in)

    to_plot = []
    gifs = [ [] for t in range(args.n_eval) ]#args.n_past + args.n_future
    nrow = min(args.batch_size, 1)
    for i in range(nrow):
        # ground truth sequence
        row = [] 
        for t in range(args.n_eval):
            row.append(gt_seq[t][i])
        to_plot.append(row)#append a seqence of ground truth image(i_th batch)

        # best sequence
        min_mse = 1e7
        for s in range(nsample):
            mse = 0
            for t in range(args.n_eval):
                mse +=  torch.sum( (gt_seq[t][i].data.cpu() - gen_seq[s][t][i].data.cpu())**2 )
            if mse < min_mse:
                min_mse = mse
                min_idx = s

        s_list = [min_idx, 
                  np.random.randint(nsample), 
                  np.random.randint(nsample), 
                  np.random.randint(nsample), 
                  np.random.randint(nsample)]
        for ss in range(len(s_list)):
            s = s_list[ss]
            row = []
            for t in range(args.n_eval):
                row.append(gen_seq[s][t][i]) 
            to_plot.append(row)#append: the best predict result + 4 random samples
        for t in range(args.n_eval):
            row = []
            row.append(gt_seq[t][i])
            for ss in range(len(s_list)):
                s = s_list[ss]
                row.append(gen_seq[s][t][i])
            gifs[t].append(row)#row: (ground truth, best_predict, sample, sample, sample, sample)

    #fname = '%s/gen/sample_%d.png' % (args.log_dir, epoch) 

    pred_fname = '%s/gen/sample_%d_all_pred.gif' % (args.log_dir, epoch) 
    #gt_fname = '%s/gen/sample_%d_gt.gif' % (args.log_dir, epoch) 
    save_gif(pred_fname, gifs)

def tensor_to_img(input):
    img = input.detach().cpu().transpose(0,1).transpose(1,2).clamp(0,1).numpy()
    img = img / np.amax(img)
    img = 255*img
    img = img.astype(np.uint8)
    return img

def save_gif(pred_fname, inputs, duration=0.5):
    #inputs shape:(seq, batch, 6, 3, 64, 64)
    num = len(inputs[0][0])
    gifs = [ [] for t in range(num) ]
    for i in range(len(inputs)):
        for j in range(num):
            gifs[j].append(tensor_to_img(inputs[i][0][j]))

    con_gif = []
    for i in range(len(inputs)):
        img = gifs[0][i]
        for j in range(1,num):
            img = np.concatenate([img,gifs[j][i]],axis=1)
        con_gif.append(img)
    
    imageio.mimsave(pred_fname, con_gif, duration=duration)