# Copyright (c) 2019, Adobe Inc. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike
# 4.0 International Public License. To view a copy of this license, visit
# https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.



########  https://github.com/adobe/antialiased-cnns/blob/master/models_lpf/__init__.py



import torch
import torch.nn.parallel
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class Downsample(nn.Module):
    def __init__(self, pad_type='reflect', filt_size=3, stride=2, channels=None, pad_off=0):
        super(Downsample, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [int(1.*(filt_size-1)/2), int(np.ceil(1.*(filt_size-1)/2)), int(1.*(filt_size-1)/2), int(np.ceil(1.*(filt_size-1)/2))]
        self.pad_sizes = [pad_size+pad_off for pad_size in self.pad_sizes]
        self.stride = stride
        self.off = int((self.stride-1)/2.)
        self.channels = channels

        # print('Filter size [%i]'%filt_size)
        if(self.filt_size==1):
            a = np.array([1.,])
        elif(self.filt_size==2):
            a = np.array([1., 1.])
        elif(self.filt_size==3):
            a = np.array([1., 2., 1.])
        elif(self.filt_size==4):
            a = np.array([1., 3., 3., 1.])
        elif(self.filt_size==5):
            a = np.array([1., 4., 6., 4., 1.])
        elif(self.filt_size==6):
            a = np.array([1., 5., 10., 10., 5., 1.])
        elif(self.filt_size==7):
            a = np.array([1., 6., 15., 20., 15., 6., 1.])

        filt = torch.Tensor(a[:,None]*a[None,:])
        filt = filt/torch.sum(filt)
        self.register_buffer('filt', filt[None,None,:,:].repeat((self.channels,1,1,1)))

        self.pad = get_pad_layer(pad_type)(self.pad_sizes)

    def forward(self, inp):
        if(self.filt_size==1):
            if(self.pad_off==0):
                return inp[:,:,::self.stride,::self.stride]
            else:
                return self.pad(inp)[:,:,::self.stride,::self.stride]
        else:
            return F.conv2d(self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1])

def get_pad_layer(pad_type):
    if(pad_type in ['refl','reflect']):
        PadLayer = nn.ReflectionPad2d
    elif(pad_type in ['repl','replicate']):
        PadLayer = nn.ReplicationPad2d
    elif(pad_type=='zero'):
        PadLayer = nn.ZeroPad2d
    else:
        print('Pad type [%s] not recognized'%pad_type)
    return PadLayer


class Downsample1D(nn.Module):
    def __init__(self, pad_type='reflect', filt_size=3, stride=2, channels=None, pad_off=0):
        super(Downsample1D, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [int(1. * (filt_size - 1) / 2), int(np.ceil(1. * (filt_size - 1) / 2))]
        self.pad_sizes = [pad_size + pad_off for pad_size in self.pad_sizes]
        self.stride = stride
        self.off = int((self.stride - 1) / 2.)
        self.channels = channels

        # print('Filter size [%i]' % filt_size)
        if(self.filt_size == 1):
            a = np.array([1., ])
        elif(self.filt_size == 2):
            a = np.array([1., 1.])
        elif(self.filt_size == 3):
            a = np.array([1., 2., 1.])
        elif(self.filt_size == 4):
            a = np.array([1., 3., 3., 1.])
        elif(self.filt_size == 5):
            a = np.array([1., 4., 6., 4., 1.])
        elif(self.filt_size == 6):
            a = np.array([1., 5., 10., 10., 5., 1.])
        elif(self.filt_size == 7):
            a = np.array([1., 6., 15., 20., 15., 6., 1.])

        filt = torch.Tensor(a)
        filt = filt / torch.sum(filt)
        self.register_buffer('filt', filt[None, None, :].repeat((self.channels, 1, 1)))

        self.pad = get_pad_layer_1d(pad_type)(self.pad_sizes)

    def forward(self, inp):
        if(self.filt_size == 1):
            if(self.pad_off == 0):
                return inp[:, :, ::self.stride]
            else:
                return self.pad(inp)[:, :, ::self.stride]
        else:
            return F.conv1d(self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1])


def get_pad_layer_1d(pad_type):
    if(pad_type in ['refl', 'reflect']):
        PadLayer = nn.ReflectionPad1d
    elif(pad_type in ['repl', 'replicate']):
        PadLayer = nn.ReplicationPad1d
    elif(pad_type == 'zero'):
        PadLayer = nn.ZeroPad1d
    else:
        print('Pad type [%s] not recognized' % pad_type)
    return PadLayer



class InvertibleConv1x1(nn.Module):
    def __init__(self, num_channels, LU_decomposed):
        super().__init__()
        w_shape = [num_channels, num_channels]
        w_init = torch.qr(torch.randn(*w_shape))[0]

        if not LU_decomposed:
            self.weight = nn.Parameter(torch.Tensor(w_init))
        else:
            p, lower, upper = torch.lu_unpack(*torch.lu(w_init))
            s = torch.diag(upper)
            sign_s = torch.sign(s)
            log_s = torch.log(torch.abs(s))
            upper = torch.triu(upper, 1)
            l_mask = torch.tril(torch.ones(w_shape), -1)
            eye = torch.eye(*w_shape)

            self.register_buffer("p", p)
            self.register_buffer("sign_s", sign_s)
            self.lower = nn.Parameter(lower)
            self.log_s = nn.Parameter(log_s)
            self.upper = nn.Parameter(upper)
            self.l_mask = l_mask
            self.eye = eye

        self.w_shape = w_shape
        self.LU_decomposed = LU_decomposed

    def get_weight(self, input, reverse):
        b, c, h, w = input.shape

        if not self.LU_decomposed:
            dlogdet = torch.slogdet(self.weight)[1] * h * w
            if reverse:
                weight = torch.inverse(self.weight)
            else:
                weight = self.weight
        else:
            self.l_mask = self.l_mask.to(input.device)
            self.eye = self.eye.to(input.device)

            lower = self.lower * self.l_mask + self.eye

            u = self.upper * self.l_mask.transpose(0, 1).contiguous()
            u += torch.diag(self.sign_s * torch.exp(self.log_s))

            dlogdet = torch.sum(self.log_s) * h * w

            if reverse:
                u_inv = torch.inverse(u)
                l_inv = torch.inverse(lower)
                p_inv = torch.inverse(self.p)

                weight = torch.matmul(u_inv, torch.matmul(l_inv, p_inv))
            else:
                weight = torch.matmul(self.p, torch.matmul(lower, u))

        return weight.view(self.w_shape[0], self.w_shape[1], 1, 1), dlogdet

    def forward(self, input, logdet=None, reverse=False):
        """
        log-det = log|abs(|W|)| * pixels
        """
        weight, dlogdet = self.get_weight(input, reverse)

        if not reverse:
            z = F.conv2d(input, weight)
            if logdet is not None:
                logdet = logdet + dlogdet
            return z, logdet
        else:
            z = F.conv2d(input, weight)
            if logdet is not None:
                logdet = logdet - dlogdet
            return z, logdet
