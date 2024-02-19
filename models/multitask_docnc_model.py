#########################
# Date : 20230328
# Author : Ruilu Wang
#########################
import logging
from collections import OrderedDict
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
import os
from torch.nn.parallel import DataParallel, DistributedDataParallel
import models.networks as networks
import models.lr_scheduler as lr_scheduler
from .base_model import BaseModel
from models.loss import CharbonnierLoss
from models.loss_new import SSIMLoss,VGGLoss
import torch.nn.functional as F
import random
from metrics.calculate_PSNR_SSIM import psnr_np

import cv2
logger = logging.getLogger('base')

def mean_channels(F):
    assert(F.dim() == 4)
    spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (F.size(2) * F.size(3))


def stdv_channels(F):
    assert(F.dim() == 4)
    F_mean = mean_channels(F)
    F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (F.size(2) * F.size(3))
    return F_variance.pow(0.5)



class AttING(nn.Module):
    def __init__(self, in_channels, channels):
        super(AttING, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv2_1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2_2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.instance = nn.InstanceNorm2d(channels, affine=True)
        self.interative = nn.Sequential(
            nn.Conv2d(channels*2, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.1),
            nn.Sigmoid()
        )
        self.act = nn.LeakyReLU(0.1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.contrast = stdv_channels
        self.process = nn.Sequential(nn.Conv2d(channels*2, channels//2, kernel_size=3, padding=1, bias=True),
                                     nn.LeakyReLU(0.1),
                                     nn.Conv2d(channels//2, channels*2, kernel_size=3, padding=1, bias=True),
                                     nn.Sigmoid())
        self.conv1x1 = nn.Conv2d(2*channels, channels, 1, 1, 0)

    def forward(self, x):
        x1 = self.conv1(x)
        out_instance = self.instance(x1)
        out_identity = x1
        # feature_save(out_identity, '1')
        # feature_save(out_instance, '2')
        out1 = self.conv2_1(out_instance)
        out2 = self.conv2_2(out_identity)
        out = torch.cat((out1, out2), 1)
        xp1 = self.interative(out)*out2 + out1
        xp2 = (1-self.interative(out))*out1 + out2
        xp = torch.cat((xp1, xp2), 1)
        xp = self.process(self.contrast(xp)+self.avgpool(xp))*xp
        xp = self.conv1x1(xp)
        xout = xp

        return xout,out_instance




class DocNC(nn.Module):
    def __init__(self):
        super(DocNC, self).__init__()

        # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # if IsING == True:
        #     # self.conv1_1 = ING(3,32)
        #     self.conv1_1 = nn.Sequential(nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
        #                                  nn.InstanceNorm2d(32, affine=True))
        # else:
        self.conv0 = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)

        self.conv1_1 = AttING(3,32)
        # self.conv1_1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2)

        self.conv5_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.upv6 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv6_1 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.conv6_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.upv7 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv7_1 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.conv7_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.upv8 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv8_1 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.conv8_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.upv9 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv9_1 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.conv9_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)

        self.conv10_1 = nn.Conv2d(32, 3, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.conv0(x)

        conv1ori,instance = self.conv1_1(x)

        conv1 = self.lrelu(self.conv1_2(self.lrelu(conv1ori)))
        pool1 = self.pool1(conv1)

        conv2 = self.lrelu(self.conv2_1(pool1))
        conv2 = self.lrelu(self.conv2_2(conv2))
        pool2 = self.pool1(conv2)

        conv3 = self.lrelu(self.conv3_1(pool2))
        conv3 = self.lrelu(self.conv3_2(conv3))
        pool3 = self.pool1(conv3)

        conv4 = self.lrelu(self.conv4_1(pool3))
        conv4 = self.lrelu(self.conv4_2(conv4))
        pool4 = self.pool1(conv4)

        conv5 = self.lrelu(self.conv5_1(pool4))
        conv5 = self.lrelu(self.conv5_2(conv5))

        up6 = F.interpolate(self.upv6(conv5),size=(conv4.shape[2],conv4.shape[3]),mode='bilinear')
        up6 = torch.cat([up6, conv4], 1)
        conv6 = self.lrelu(self.conv6_1(up6))
        conv6 = self.lrelu(self.conv6_2(conv6))

        up7 = F.interpolate(self.upv7(conv6),size=(conv3.shape[2],conv3.shape[3]),mode='bilinear')
        up7 = torch.cat([up7, conv3], 1)
        conv7 = self.lrelu(self.conv7_1(up7))
        conv7 = self.lrelu(self.conv7_2(conv7))

        up8 = F.interpolate(self.upv8(conv7),size=(conv2.shape[2],conv2.shape[3]),mode='bilinear')
        up8 = torch.cat([up8, conv2], 1)
        conv8 = self.lrelu(self.conv8_1(up8))
        conv8 = self.lrelu(self.conv8_2(conv8))

        up9 = F.interpolate(self.upv9(conv8),size=(conv1.shape[2],conv1.shape[3]),mode='bilinear')
        up9 = torch.cat([up9, conv1], 1)
        conv9 = self.lrelu(self.conv9_1(up9))
        conv9 = self.lrelu(self.conv9_2(conv9))


        conv10 = self.conv10_1(conv9)
        out = conv10
        
        return out,instance,conv1ori

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.normal_(0.0, 0.02)
            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0.0, 0.02)

    def lrelu(self, x):
        outt = torch.max(0.2 * x, x)
        return outt
    
class multitask_docnc_model(BaseModel):
    def __init__(self, opt):
        super(multitask_docnc_model, self).__init__(opt)

        self.rank = -1  # non dist training
        train_opt = opt['train']

        # define network and load pretrained models
        print(self.device)
        self.netG = DocNC().to(self.device)
        if opt['dist']:
            self.netG = DistributedDataParallel(self.netG, device_ids=[torch.cuda.current_device()])
        else:
            self.netG = DataParallel(self.netG)
        # print network
        self.print_network()
        self.load()

#######################   Continue learning model parameter setting
        if train_opt['ewc']:
            self.Importance_Pre = torch.load(os.path.join(self.opt['path']['pretrain'], 'Importance.pth'))
            self.Star_vals_Pre = torch.load(os.path.join(self.opt['path']['pretrain'], 'Star.pth'))
            logger.info("Load Pretrain Importance and Stars!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        else:
            self.Importance = []
            self.Star_vals = []
            for w in self.netG.parameters():
                self.Importance.append(torch.zeros_like(w))
                self.Star_vals.append(torch.zeros_like(w))
            logger.info("Initial Importance and Stars with zeros!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

###############################   Distilation setting
        if train_opt['distill']:
            self.netG_Pre = networks.define_G(opt).to(self.device)
            self.netG_Pre = DataParallel(self.netG_Pre)
            self.load_Pre()
            self.netG_Pre.eval()
####################################################################
        if self.is_train:
            self.netG.train()

            #### loss
            loss_type = train_opt['pixel_criterion']
            if loss_type == 'l1':
                self.cri_pix = nn.L1Loss().to(self.device)
                self.cri_ssim = SSIMLoss().to(self.device)
                self.mse = nn.MSELoss().to(self.device)
                # self.cri_vgg = VGGLoss(id=4).to(self.device)
            elif loss_type == 'l2':
                self.cri_pix = nn.MSELoss().to(self.device)
                self.cri_ssim = SSIMLoss().to(self.device)
            elif loss_type == 'cb':
                self.cri_pix = CharbonnierLoss().to(self.device)
                self.cri_ssim = SSIMLoss().to(self.device)
                # self.cri_vgg = VGGLoss(id=4).to(self.device)
                # referring to DE-GAN
                # we add binary loss
            elif loss_type == 'binary':
                self.cri_pix = nn.MSELoss().to(self.device)
                self.cri_ssim = SSIMLoss().to(self.device)  
                self.mse = nn.MSELoss().to(self.device)
                self.cri_bce =  nn.BCEWithLogitsLoss().to(self.device)         
            else:
                raise NotImplementedError('Loss type [{:s}] is not recognized.'.format(loss_type))


            self.l_pix_w = train_opt['pixel_weight']
            self.l_ssim_w = train_opt['ssim_weight']
            self.l_vgg_w = train_opt['vgg_weight']

            #### optimizers
            wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0
            if train_opt['fix_some_part']:
                normal_params = []
                tsa_fusion_params = []
                for k, v in self.netG.named_parameters():
                    if v.requires_grad:
                        if 'tsa_fusion' in k:
                            tsa_fusion_params.append(v)
                        else:
                            normal_params.append(v)
                    else:
                        if self.rank <= 0:
                            logger.warning('Params [{:s}] will not optimize.'.format(k))
                optim_params = [
                    {  # add normal params first
                        'params': normal_params,
                        'lr': train_opt['lr_G']
                    },
                    {
                        'params': tsa_fusion_params,
                        'lr': train_opt['lr_G']
                    },
                ]
            else:
                optim_params = []
                for k, v in self.netG.named_parameters():
                    if v.requires_grad:
                        optim_params.append(v)
                    else:
                        if self.rank <= 0:
                            logger.warning('Params [{:s}] will not optimize.'.format(k))

            self.optimizer_G = torch.optim.Adam(optim_params, lr=train_opt['lr_G'],
                                                weight_decay=wd_G,
                                                betas=(train_opt['beta1'], train_opt['beta2']))
            self.optimizers.append(self.optimizer_G)

            #### schedulers
            if train_opt['lr_scheme'] == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.MultiStepLR_Restart(optimizer, train_opt['lr_steps'],
                                                         restarts=train_opt['restarts'],
                                                         weights=train_opt['restart_weights'],
                                                         gamma=train_opt['lr_gamma'],
                                                         clear_state=train_opt['clear_state']))
            elif train_opt['lr_scheme'] == 'CosineAnnealingLR_Restart':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.CosineAnnealingLR_Restart(
                            optimizer, train_opt['T_period'], eta_min=train_opt['eta_min'],
                            restarts=train_opt['restarts'], weights=train_opt['restart_weights']))
            else:
                raise NotImplementedError()

            self.log_dict = OrderedDict()

    def feed_data(self, data, need_GT=True):
        GT_IMG = data['GT']
        print(GT_IMG.shape)
        Back_IMG = data['Back']
        Blur_IMG = data['Blur']
        Noise_IMG = data['Noise']
        Watermark_IMG = data['Watermark']
        Shadow_IMG = data['Shadow']
        # LQright_IMG = data['LQright']
        self.Back = Back_IMG.to(self.device)
        self.Blur = Blur_IMG.to(self.device)
        self.Noise = Noise_IMG.to(self.device)
        self.Watermark = Watermark_IMG.to(self.device)
        self.Shadow = Shadow_IMG.to(self.device)
        # print('feed data:', self.Back.shape)
        # self.varright_L = LQright_IMG.to(self.device)
        if need_GT:
            self.real_H = GT_IMG.to(self.device)
    

    def set_params_lr_zero(self):
        # fix normal module
        self.optimizers[0].param_groups[0]['lr'] = 0

    def threshold_control(self, input):
        tmp0 = torch.ones(input.shape)*1
        tmp1 = torch.ones(input.shape)*0
        tmp0 = tmp0.to(self.device)
        tmp1 = tmp1.to(self.device)
        out = torch.where(input>tmp0,tmp0, input)
        out = torch.where(out<tmp1,tmp1, out)
        # print(out)   
        return out


    def optimize_parameters(self, step):
        if self.opt['train']['fix_some_part'] and step < self.opt['train']['fix_some_part']:
            self.set_params_lr_zero()

        self.netG.zero_grad() ################################################# new add
        self.optimizer_G.zero_grad()

        # LR_right = self.varright_L
        self.netG.cuda()

        out,instf,fusef = self.netG(self.Back)
        out = self.threshold_control(out)
        '''
        ############### Threshold Control  ####################
        tmp0 = torch.ones(out.shape)*1
        tmp1 = torch.ones(out.shape)*0
        tmp0 = tmp0.to(self.device)
        tmp1 = tmp1.to(self.device)
        out = torch.where(out>tmp0,tmp0, out)
        out = torch.where(out<tmp1,tmp1, out)
        print(out)
        #######################################################
        '''
        
        out_blur,_,_ = self.netG(self.Blur)

        out_noise,_,_ = self.netG(self.Noise)

        out_watermark,_,_ = self.netG(self.Watermark)

        out_shadow,_,_ = self.netG(self.Shadow)


        if self.opt['train']['distill']:
            var_fake = self.real_H
            # fakeout,gtinst, = self.netG(var_fake)
            with torch.no_grad():
                self.netG_Pre.eval()
                _,gtinstf,gtfusef = self.netG_Pre(var_fake.detach())


        gt = self.real_H
        
        # in the early process, out values are too little
        # it is not proper to binarilize at the begining 
        # threshhold
        '''
        if(torch.max(out)>0.97):
            t = torch.ones(out.shape)*0.95
            t = t.to(self.device)
            out = ((out>t)*1.0)
        print('gt max:', torch.max(gt), 'out max:', torch.max(out))
        '''
        print('gt max:', torch.max(gt), 'out max:', torch.max(out))
        print('gt max:', torch.min(gt), 'out max:', torch.min(out))

        # total loss 
        
        l_contrast = (self.mse(out, out_blur) +
                     self.mse(out, out_noise) +
                     self.mse(out, out_watermark) +
                     self.mse(out, out_shadow))/4
        
        l_total = self.mse(out, gt) + 100*self.cri_bce(out, gt) + l_contrast
        # print('mse:', self.mse(out, gt), 'bce:', self.cri_bce(out, gt),'con:', l_contrast)

                  # + 1.2*self.cri_pix(out, out_pre)
        if self.opt['train']['distill']:
            l_total += self.opt['train']['distill_coff']*(self.cri_pix(instf,gtinstf.detach())+self.cri_pix(fusef,gtfusef.detach()))

        if self.opt['train']['ewc']:
            for i, w in enumerate(self.netG.parameters()):
                l_total += self.opt['train']['ewc_coff']/2 * torch.sum(torch.mul(self.Importance_Pre[i], torch.abs(w - self.Star_vals_Pre[i])))\
                           + self.opt['train']['ewc_coff']/4 * torch.square(torch.sum(torch.mul(self.Importance_Pre[i], torch.abs(w - self.Star_vals_Pre[i]))))


        l_total.backward()
        self.optimizer_G.step()
        self.fake_H = out
        #print('out&gt min&max:',torch.max(self.fake_H), torch.max(self.real_H))
        psnr = psnr_np(self.fake_H.detach(), self.real_H.detach())
        #print('psnr:',psnr, psnr_np(self.fake_H.detach()*255, self.real_H.detach()*255))

        # set log
        self.log_dict['psnr'] = psnr.item()
        self.log_dict['l_total'] = l_total.item()

####### test function and it helpers
    def feed_val_data(self, data, need_GT=True):
        LQ_IMG = data['LQ']
        GT_IMG = data['GT']
        # LQright_IMG = data['LQright']
        self.var_L = LQ_IMG.to(self.device)
        self.de_path = data['LQ_path']
        # self.varright_L = LQright_IMG.to(self.device)
        if need_GT:
            self.real_H = GT_IMG.to(self.device)
    def split2(self,dataset,size,h,w):
        newdataset=[]
        nsize1=256
        nsize2=256
        for i in range (size):
            im=dataset[i]
            for ii in range(0,h,nsize1): #2048
                for iii in range(0,w,nsize2): #1536
                    newdataset.append(im[ii:ii+nsize1,iii:iii+nsize2,:])
    
        return np.array(newdataset) 
    def merge_image2(self,splitted_images, h,w):
        image=np.zeros(((h,w,3)))
        nsize1=256
        nsize2=256
        ind =0
        for ii in range(0,h,nsize1):
            for iii in range(0,w,nsize2):
                image[ii:ii+nsize1,iii:iii+nsize2,:]=splitted_images[ind]
                ind=ind+1
        return np.array(image) 

    def val(self):
        self.netG.eval()
        with torch.no_grad():
            # deg_image = self.var_L# /255.0
            
            deg_image = np.array(cv2.imread(self.de_path[0],cv2.IMREAD_UNCHANGED))/255
            print('read:',np.max(deg_image), np.min(deg_image))
            h,w,_ = deg_image.shape
            if(h<384):
                cv2.resize(deg_image, (w, 384))
                h = 384
            if(w<384):
                cv2.resize(deg_image, (384,h))
                w = 384
            while w%4!=0:
                w+=1
            while h%4!=0:
                h+=1
            deg_image = cv2.resize(deg_image,(w,h))
            #print(deg_image.shape)
            print('resize:',np.max(deg_image), np.min(deg_image))

            test_image = deg_image
            #print('deg_images.shape:', test_image.shape)
            h =  ((test_image.shape [0] // 256) +1)*256 
            w =  ((test_image.shape [1] // 256 ) +1)*256

            test_padding = np.zeros((h,w,3))+1
            test_padding[:test_image.shape[0],:test_image.shape[1]]=test_image

            test_image_p=self.split2(test_padding.reshape(1,h,w,3),1,h,w)
            #print('p:', test_image_p.shape)
            predicted_list=[]
            for l in range(test_image_p.shape[0]):
                #print("patch shape:",np.transpose(test_image_p[l], [2,0,1]).shape)
                tmp = np.transpose(test_image_p[l], [2,0,1])
                tmp = torch.from_numpy(tmp).reshape(1,3,256,256).float()
                #print(tmp.shape)
                self.netG(tmp)


                # for torch : 1.7.1
                predict = self.netG(tmp)[0].squeeze().permute((1,2,0)).cpu().numpy()
                # else :
                # predict = torch.permute(self.netG(tmp)[0].squeeze(), (1,2,0)).cpu().numpy()


                predicted_list.append(predict)

            predicted_image = np.array(predicted_list)#.reshape()
            predicted_image=self.merge_image2(predicted_image,h,w)

            '''
            # test parameter
            threshhold = 0.95
            predicted_image = (predicted_image[:,:]>threshhold)*1
            '''
            predicted_image=(predicted_image[:test_image.shape[0],:test_image.shape[1]]*255).round()
            #print('predicted shape:', predicted_image.shape)
            # print(predicted_image)
            
            # out,_,_ = self.netG(self.var_L)
            self.fake_H = predicted_image
        self.netG.train()
    def get_val_current_visuals(self, need_GT=True):
        out_dict = OrderedDict()
        out_dict['LQ'] = self.var_L.detach()[0].float().cpu()
        # out_dict['rlt'] = self.fake_H.detach()[0].float().cpu()
        out_dict['rlt'] = self.fake_H
        if need_GT:
            out_dict['GT'] = self.real_H.detach()[0].float().cpu()
        return out_dict
    def test(self):
        self.netG.eval()
        with torch.no_grad():
            out,_,_ = self.netG(self.Back)
            self.fake_H = out
        self.netG.train()

################# Initialize importance
    def Init_M(self):
        self.Importance = []
        self.Star_vals = []
        for w in self.netG.parameters():
            self.Importance.append(torch.zeros_like(w))
            self.Star_vals.append(torch.zeros_like(w))
        print("Init importance parameters again!!!!")

################ self compute importance
    def compute_M(self, step):
        phr1 = self.netG(self.var_L)

        #hr4 = self.real_H[:, :, 0::4, 0::4]
        #hr2 = self.real_H[:, :, 0::2, 0::2]
        hr1 = self.real_H

        l_t = self.mse(phr1, hr1)
        #+ self.cri_ssim(phr2, hr2) + self.cri_ssim(phr4, hr4)
        l_total = -l_t
        self.netG.zero_grad()
        self.optimizer_G.zero_grad()
        l_total.backward()
        with torch.no_grad():
            for i, w in enumerate(self.netG.parameters()):
                self.Importance[i].mul_(step / (step + 1))
                self.Importance[i].add_(torch.abs(w.grad.data)/(step+1))

        self.netG.zero_grad()
        self.optimizer_G.zero_grad()



    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_GT=True):
        out_dict = OrderedDict()
        out_dict['LQ'] = self.Back.detach()[0].float().cpu()
        out_dict['rlt'] = self.fake_H.detach()[0].float().cpu()
        if need_GT:
            out_dict['GT'] = self.real_H.detach()[0].float().cpu()
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)
        if self.rank <= 0:
            logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
            logger.info(s)

    def load(self):
        load_path_G = self.opt['path']['pretrain_model_G']
        if load_path_G is not None:
            logger.info('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG, self.opt['path']['strict_load'])

    def load_Pre(self):
        load_path_G = self.opt['path']['pretrain_model_G']
        if load_path_G is not None:
            logger.info('Loading model for WarmUp G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG_Pre, self.opt['path']['strict_load'])

    def save_M(self,name):
        with torch.no_grad():
            for i, w in enumerate(self.netG.parameters()):
                self.Star_vals[i].copy_(w)
        torch.save(self.Importance, os.path.join(self.opt['path']['models'], name+'Importance.pth'))
        torch.save(self.Star_vals, os.path.join(self.opt['path']['models'], name+'Star.pth'))

    def save(self, iter_label):
        self.save_network(self.netG, 'G', iter_label)

    def save_best(self,name):
        self.save_network(self.netG, 'best'+name, 0)
