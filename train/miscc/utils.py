import os
import errno
import numpy as np
import cupy as cp
from cupyx.scipy.signal import fftconvolve
from copy import deepcopy
from miscc.config import cfg
from scipy.io.wavfile import write
from torch.nn import init
import torch
import torch.nn as nn
import torchvision.utils as vutils
from wavefile import WaveWriter, Format
# import RT60
from multiprocessing import Pool
from torch.nn.functional import normalize
import scipy.signal
import matplotlib.pyplot as plt 
#############################
def KL_loss(mu, logvar):
    # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.mean(KLD_element).mul_(-0.5)
    return KLD


def compute_discriminator_loss(netD, real_RIRs, fake_RIRs,
                               real_labels, fake_labels,
                               conditions, gpus):
    criterion = nn.BCELoss()
    batch_size = real_RIRs.size(0)
    cond = conditions.detach()
    fake = fake_RIRs.detach()
    real_features = nn.parallel.data_parallel(netD, (real_RIRs), gpus)
    fake_features = nn.parallel.data_parallel(netD, (fake), gpus)
    # real pairs
    #print("util conditions ",cond.size())
    inputs = (real_features, cond)
    real_logits = nn.parallel.data_parallel(netD.get_cond_logits, inputs, gpus)
    errD_real = criterion(real_logits, real_labels)
    # wrong pairs
    inputs = (real_features[:(batch_size-1)], cond[1:])
    wrong_logits = \
        nn.parallel.data_parallel(netD.get_cond_logits, inputs, gpus)
    errD_wrong = criterion(wrong_logits, fake_labels[1:])
    # fake pairs
    inputs = (fake_features, cond)
    fake_logits = nn.parallel.data_parallel(netD.get_cond_logits, inputs, gpus)
    errD_fake = criterion(fake_logits, fake_labels)

    if netD.get_uncond_logits is not None:
        real_logits = \
            nn.parallel.data_parallel(netD.get_uncond_logits,
                                      (real_features), gpus)
        fake_logits = \
            nn.parallel.data_parallel(netD.get_uncond_logits,
                                      (fake_features), gpus)
        uncond_errD_real = criterion(real_logits, real_labels)
        uncond_errD_fake = criterion(fake_logits, fake_labels)
        #
        errD = ((errD_real + uncond_errD_real) / 2. +
                (errD_fake + errD_wrong + uncond_errD_fake) / 3.)
        errD_real = (errD_real + uncond_errD_real) / 2.
        errD_fake = (errD_fake + uncond_errD_fake) / 2.
    else:
        errD = errD_real + (errD_fake + errD_wrong) * 0.5
    return errD, errD_real.data, errD_wrong.data, errD_fake.data
    # return errD, errD_real.data[0], errD_wrong.data[0], errD_fake.data[0]



def compute_generator_loss(epoch,netD,real_RIRs, fake_RIRs, real_labels, conditions,filters, gpus):
    criterion = nn.BCELoss()
    loss = nn.L1Loss() #nn.MSELoss()
    loss1 = nn.MSELoss()
    RT_error = 0
    # print("num", real_RIRs.size(),real_RIRs.size()[0])
    # input("kk")
    # print("real_RIRs ", real_RIRs.shape)
    # print("fake_RIRs ", fake_RIRs.shape)

    cond = conditions.detach()
    fake_features = nn.parallel.data_parallel(netD, (fake_RIRs), gpus)
    # fake pairs
    inputs = (fake_features, cond)
    fake_logits = nn.parallel.data_parallel(netD.get_cond_logits, inputs, gpus)
    L1_error = loss(real_RIRs,fake_RIRs)
    # print("real shape ",real_RIRs.shape )
    # input("summa ")
    MSE_error1 = loss1(real_RIRs[:,:,0:3968],fake_RIRs[:,:,0:3968])
    MSE_error2 = loss1(real_RIRs[:,:,3968:4096],fake_RIRs[:,:,3968:4096])
    MSE_error3 = loss1(torch.sub(real_RIRs[:,0,0:3968],real_RIRs[:,1,0:3968]),torch.sub(fake_RIRs[:,0,0:3968],fake_RIRs[:,1,0:3968]))


    
    ######################Energy Decay Start############################
    filter_length = 16384  # a magic number, not need to tweak this much
    mult1 = 10

    real_ec = convert_IR2EC_batch(cp.asarray(real_RIRs), filters, filter_length)
    fake_ec = convert_IR2EC_batch(cp.asarray(fake_RIRs.to("cpu").detach()), filters, filter_length)



    divergence_loss = loss1(real_ec,fake_ec) * mult1
    ######################Energy Decay End############################


    MSE_ERROR11 = (MSE_error1+MSE_error3)*4096*10*10
    MSE_ERROR21 = MSE_error2*128*10*100*5
    MSE_ERROR = MSE_ERROR11+MSE_ERROR21
    criterion_loss = criterion(fake_logits, real_labels)
    # errD_fake = criterion(fake_logits, real_labels) + 5* 4096 * MSE_error1 #+ 40 * RT_error
    errD_fake = 2*criterion_loss + divergence_loss+(MSE_ERROR) #+ 5* 4096*MSE_error1
    if netD.get_uncond_logits is not None:
        fake_logits = \
            nn.parallel.data_parallel(netD.get_uncond_logits,
                                      (fake_features), gpus)
        uncond_errD_fake = criterion(fake_logits, real_labels)
        errD_fake += uncond_errD_fake
    return errD_fake, L1_error,divergence_loss, MSE_ERROR11,MSE_ERROR21 ,criterion_loss #,RT_error


#############################
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0.0)


#############################
def save_RIR_results(data_RIR, fake, epoch, RIR_dir):
    num = 64# cfg.VIS_COUNT
    print("fake size  ", fake.shape)

    fake = fake[0:num]
    print("fake size123  ", fake.shape)

    # data_RIR is changed to [0,1]
    if data_RIR is not None:
        data_RIR = data_RIR[0:num]
        for i in range(num):
            # #print("came 1")
            real_RIR_path = RIR_dir+"/real_sample"+str(i)+"_epoch_"+str(epoch)+".wav" 
            fake_RIR_path = RIR_dir+"/fake_sample"+str(i)+"_epoch_"+str(epoch)+".wav"
            fs =16000

            real_IR = np.array(data_RIR[i].to("cpu").detach())
            fake_IR = np.array(fake[i].to("cpu").detach())

            r = WaveWriter(real_RIR_path, channels=2, samplerate=fs)
            r.write(np.array(real_IR))
            f = WaveWriter(fake_RIR_path, channels=2, samplerate=fs)
            f.write(np.array(fake_IR))           


            # write(real_RIR_path,fs,real_IR)
            # write(fake_RIR_path,fs,fake_IR)


            # write(real_RIR_path,fs,real_IR)
            # write(fake_RIR_path,fs,fake_IR)

        # vutils.save_image(
        #     data_RIR, '%s/real_samples.png' % RIR_dir,
        #     normalize=True)
        # # fake.data is still [-1, 1]
        # vutils.save_image(
        #     fake.data, '%s/fake_samples_epoch_%03d.png' %
        #     (RIR_dir, epoch), normalize=True)
    else:
        for i in range(num):
            # #print("came 2")
            fake_RIR_path = RIR_dir+"/small_fake_sample"+str(i)+"_epoch_"+str(epoch)+".wav"
            fs =16000
            fake_IR = np.array(fake[i].to("cpu").detach())
            f = WaveWriter(fake_RIR_path, channels=1, samplerate=fs)
            f.write(np.array(fake_IR))
            
            # write(fake_RIR_path,fs,fake[i].astype(np.float32))

        # vutils.save_image(
        #     fake.data, '%s/lr_fake_samples_epoch_%03d.png' %
        #     (RIR_dir, epoch), normalize=True)


def save_model(netG, netD,mesh_net, epoch, model_dir):
    torch.save(
        netG.state_dict(),
        '%s/netG_epoch_%d.pth' % (model_dir, epoch))
    torch.save(
        mesh_net.state_dict(),
        '%s/mesh_net_epoch_%d.pth' % (model_dir, epoch))
    torch.save(
        netD.state_dict(),
        '%s/netD_epoch_last.pth' % (model_dir))
    #print('Save G/D models')


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def convert_IR2EC(rir, filters, filter_length):
    subband_ECs = np.zeros((len(rir), filters.shape[1]))
    for i in range(filters.shape[1]):
        subband_ir = scipy.signal.fftconvolve(rir, filters[:, i])
        subband_ir = subband_ir[(filter_length - 1):]
        squared = np.square(subband_ir[:len(rir)])
        subband_ECs[:, i] = np.cumsum(squared[::-1])[::-1]
    return subband_ECs

def convert_IR2EC_batch(rir, filters, filter_length):
    # filters = cp.asarray([[filters]])
    rir = rir[:,:,0:3968]
    subband_ECs = cp.zeros((rir.shape[0],rir.shape[1],rir.shape[2], filters.shape[3]))
    for i in range(filters.shape[3]):
        subband_ir = fftconvolve(rir, filters[:,:,:, i])
        subband_ir = subband_ir[:,:,(filter_length - 1):]
        squared = cp.square(subband_ir[:,:,:rir.shape[2]])
        subband_ECs[:, :,:,i] = cp.log(cp.cumsum(squared[:,:,::-1],axis=2)[:,:,::-1])
    subband_ECs = torch.tensor(subband_ECs,device='cuda')
    return subband_ECs



def generate_complementary_filterbank(
        fc=[125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0],
        fs=16000,
        filter_order=4,
        filter_length=16384,
        power=True):
    """Return a zero-phase power (or amplitude) complementary filterbank via Butterworth prototypes.
    Parameters:
        fc - filter center frequencies
        fs - sampling rate
        filter_order - order of the prototype Butterworth filters
        filter_length - length of the resulting zero-phase FIR filters
        power - boolean to set if the filter is power or amplitude complementary
    """

    # sort in increasing cutoff
    fc = np.sort(fc)

    assert fc[-1] <= fs/2

    numFilts = len(fc)
    nbins = filter_length
    signal_z1 = np.zeros(2 * nbins)
    signal_z1[0] = 1
    irBands = np.zeros((2 * nbins, numFilts))

    for i in range(numFilts - 1):
        wc = fc[i] / (fs/2.0)
        # if wc >= 1:
        #     wc = .999999

        B_low, A_low = scipy.signal.butter(filter_order, wc, btype='low')
        B_high, A_high = scipy.signal.butter(filter_order, wc, btype='high')


        # Store the low band
        irBands[:, i] = scipy.signal.lfilter(B_low, A_low, signal_z1)

        # Store the high
        signal_z1 = scipy.signal.lfilter(B_high, A_high, signal_z1)

        # Repeat for the last band of the filter bank
    irBands[:, -1] = signal_z1

    # Compute power complementary filters
    if power:
        ir2Bands = np.real(np.fft.ifft(np.square(np.abs(np.fft.fft(irBands, axis=0))), axis=0))
    else:
        ir2Bands = np.real(np.fft.ifft(np.abs(np.abs(np.fft.fft(irBands, axis=0))), axis=0))

    ir2Bands = np.concatenate((ir2Bands[nbins:(2 * nbins), :], ir2Bands[0:nbins, :]), axis=0)

    return ir2Bands
