import torch.utils.data as data
import torch
from torchvision.transforms import Compose, ToTensor
import os
import random
# from PIL import Image,ImageOps
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import torch.nn.functional as F
import cv2
import numpy as np


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def get_patch(input, target, patch_size, scale = 1, ix=-1, iy=-1):
    # if(len(input.shape)<3):
    #     input = np.expand_dims(input, 2).repeat(3, axis = 2)
    # ih, iw, channels = input.shape
    # if(ih<384):
    #     input = np.pad(input, ((0,384-ih)), 'edge')
    # elif(iw<384):
    #     input = np.pad(input, ((1,384-iw)), 'edge')
    ih, iw, channels = input[0].shape    

    # (th, tw) = (scale * ih, scale * iw)

    patch_mult = scale  # if len(scale) > 1 else 1
    tp = patch_mult * patch_size
    ip = tp // scale

    if ix == -1:
        ix = random.randrange(0, ih - ip + 1)
    if iy == -1:
        iy = random.randrange(0, iw - ip + 1)

    # (tx, ty) = (scale * ix, scale * iy)

    for idx,item in enumerate(input) :
        input[idx] = item[ix:ix + ip, iy:iy + ip, :]
    #input = input[ix:ix + ip, iy:iy + ip, :]  # [:, ty:ty + tp, tx:tx + tp]
    target = target[ix:ix + ip, iy:iy + ip, :]  # [:, iy:iy + ip, ix:ix + ip]


    return  input[0], input[1], input[2], input[3], input[4], target


def augment(inputs, target, hflip, rot):
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot180 = rot and random.random() < 0.5

    def _augment(inputs,target):
        if hflip:
            for idx,item in enumerate(inputs) :
                inputs[idx] = item[:, ::-1, :]
            target = target[:, ::-1, :]
        if vflip:
            for idx,item in enumerate(inputs) :
                inputs[idx] = item[::-1, :, :]
            target = target[::-1, :, :]
        if rot180:
            for idx,item in enumerate(inputs) :
                inputs[idx] = cv2.rotate(item, cv2.ROTATE_180)
            target = cv2.rotate(target, cv2.ROTATE_180)
        return inputs, target

    inputs, target = _augment(inputs, target)

    return inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], target



# def get_image_hdr(img):
#     img = cv2.imread(img,cv2.IMREAD_UNCHANGED)
#     img = np.round(img/(2**6)).astype(np.uint16)
#     img = img.astype(np.float32)/1023.0
#     return img

def get_image_ldr(img):
    img = cv2.imread(img,cv2.IMREAD_UNCHANGED).astype(np.float32)/255.0
    # if img.shape[1]*img.shape[2] >= 800*800:
    #     img = cv2.resize(img,(img.shape[1]//2,img.shape[0]//2))
    w,h = img.shape[0],img.shape[1]

    ########################################
    # 20221122
    # handile with images who only has one channle
    if(len(img.shape)<3):
        img = np.expand_dims(img, 2).repeat(3, axis = 2)
    if(h<384):
        cv2.resize(img, (384, w))
        h = 384
    if(w<384):
        cv2.resize(img, (h,384))
        w = 384
    ########################################

    while w%4!=0:
        w+=1
    while h%4!=0:
        h+=1
    img = cv2.resize(img,(h,w))
    return img


def load_image_train2(group):

    target = get_image_ldr(group[0])
    back = get_image_ldr(group[1])
    blur = get_image_ldr(group[2]) 
    noise = get_image_ldr(group[3])
    shadow = get_image_ldr(group[4])
    watermark = get_image_ldr(group[5])

    return target, back, blur, noise, shadow, watermark


def transform():
    return Compose([
        ToTensor(),
    ])

def BGR2RGB_toTensor(back, blur, noise, shadow, watermark, target):
    back = back[:, :, [2, 1, 0]]
    blur = blur[:, :, [2, 1, 0]]
    noise = noise[:, :, [2, 1, 0]]
    shadow = shadow[:, :, [2, 1, 0]]
    watermark = watermark[:, :, [2, 1, 0]]
    target = target[:, :, [2, 1, 0]]

    back = torch.from_numpy(np.ascontiguousarray(np.transpose(back, (2, 0, 1)))).float()
    blur = torch.from_numpy(np.ascontiguousarray(np.transpose(blur, (2, 0, 1)))).float()
    noise = torch.from_numpy(np.ascontiguousarray(np.transpose(noise, (2, 0, 1)))).float()
    shadow = torch.from_numpy(np.ascontiguousarray(np.transpose(shadow, (2, 0, 1)))).float()
    watermark = torch.from_numpy(np.ascontiguousarray(np.transpose(watermark, (2, 0, 1)))).float()
    target = torch.from_numpy(np.ascontiguousarray(np.transpose(target, (2, 0, 1)))).float()
    return back, blur, noise, shadow, watermark, target

class DatasetFromFolder(data.Dataset):
    """
    For test dataset, specify
    `group_file` parameter to target TXT file
    data_augmentation = None
    black_edge_crop = None
    flip = None
    rot = None
    """
    def __init__(self, upscale_factor, data_augmentation, group_file, patch_size, black_edges_crop, hflip, rot, transform=transform()):
        super(DatasetFromFolder, self).__init__()
        groups = [line.rstrip() for line in open(os.path.join(group_file))]
        # assert groups[0].startswith('/'), 'Paths from file_list must be absolute paths!'
        self.image_filenames = [group.split('|') for group in groups]
        #print(self.image_filenames)
        self.upscale_factor = upscale_factor
        self.transform = transform
        self.data_augmentation = data_augmentation
        self.patch_size = patch_size
        self.black_edges_crop = black_edges_crop
        self.hflip = hflip
        self.rot = rot

    def __getitem__(self, index):

        target, back, blur, noise, shadow, watermark = load_image_train2(self.image_filenames[index])

        if self.patch_size!=None:

            back, blur, noise, shadow, watermark, target = get_patch([back, blur, noise, shadow, watermark], target, self.patch_size, self.upscale_factor)
            


        if self.data_augmentation:
            back, blur, noise, shadow, watermark, target = augment([back, blur, noise, shadow, watermark], target, self.hflip, self.rot)

        if self.transform:
            back, blur, noise, shadow, watermark, target = BGR2RGB_toTensor(back, blur, noise, shadow, watermark, target)

        return {'Back': back, 'Blur': blur, 'Noise': noise, 'Shadow': shadow, 'Watermark': watermark, 'GT': target, 'GT_path': self.image_filenames[index][0], 'Back_path': self.image_filenames[index][1]}

    def __len__(self):
        return len(self.image_filenames)


