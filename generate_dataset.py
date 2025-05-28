import os
import re
import random
from random import randint
import shutil

import cv2
import numpy as np
from PIL import Image
from PIL import ImageFont, ImageDraw, ImageOps
from utils.motionblur import Kernel


def noisy(image, sigma = 0.05):
    # add Gaussian Noise on images
    # sigma is the magnitude of noise
    row,col,ch= image.shape
    mean = 0
    # sigma = 0.05
    gauss = np.random.normal(mean,sigma,(row,col,ch))
    gauss = gauss.reshape(row,col,ch)
    noisy = image/255 + gauss
    #所有值必須介於 0~1 之間，超過1 = 1，小於0 = 0
    noisy = np.clip(noisy, 0, 1)
    noisy = np.uint8(noisy*255)
    return noisy

def alpha_mix(fore, back):
    # add back ground for GT
    # /home/eescut/Desktop/DR_Datasets/DIBCO2009/Original/H01.bmp
    # get direction
    split_dir = re.split('[/|.]', fore)
    print(split_dir)
    GT_name = split_dir[-2]
    back_id = re.sub("\D", "", back)
    root = '/'.join(split_dir[0:-3]) + '/'
    target = root + 'withBack/'
    if(not os.path.exists(target)):
        os.mkdir(target)
    
    # read and resize
    fore = cv2.imread(fore)
    back = cv2.imread(back)
    foreground = fore
    (h,w,c) = fore.shape
    # resize the background to the same shape as foreground
    background = cv2.resize(back, (w,h))
    alpha = fore/255
    #255:white 0:black

    # Convert uint8 to float
    alpha = alpha.astype(float)
    background = background.astype(float)
    out = cv2.multiply(alpha, background)

    # Display image
    # cv2.imwrite("outImg.jpg", outImage/255)
    cv2.imwrite(''.join([target, GT_name, '_', back_id, '.jpg']), out)

def watermark(image_path):
    
    names = ['CLASSIFIED', 'TOP SECRET', 'DO NOT COPY', 'VERIFIED','CONFIDENTIAL', 'RIGHTS RESERVED', 'PYTHON', 'WATERMARKING', 'COMPANY', 'DRAFT']
    names2 = ['版权所有', '禁止翻印', '机密']
    # fonts1 = ['arial','times']
    fonts1 = ['./Fonts/ARIALNBI.TTF','./Fonts/ARIALNBI.TTF', './Fonts/TIMESBD.TTF', './Fonts/TIMESBI.TTF']
    fonts2 = ['./Fonts/MSYHBD.TTC', './Fonts/DENGB.TTF']

    im=(Image.open(image_path))#*255
    width, height = im.size
    
    mode = randint(0,1)
    if(mode == 1):
        # without box
        f = fnt = ImageFont.truetype( fonts1[randint(0,3)], randint(150, 350))
        txt=Image.new('L', (width, height))
        # create a  draw object to draw on txt.img
        d = ImageDraw.Draw(txt)
        ran=randint(0, 9)
        # draw text
        # ImageDraw.text(text_xy_anchor_coordination, text, font=font, fill=(225, 225, 225, 225))
        # d.text( (randint(20,260), randint(50,320)), names[ran],  font=f, fill=randint(245,253))
        d.text( (randint(0,int(width/3)), randint(0,int(height/3))), names[ran],  font=f, fill=randint(245,253))
    else:
        # with box
        txt=Image.new('L', (width, height))
        draw = ImageDraw.Draw(txt)
        fontSize=randint(125,350)
        font = ImageFont.truetype( fonts2[randint(0,1)], fontSize)
        x, y = (randint(0,int(width/3)), randint(0,int(height/3)))
        # x, y = (30, 200)
        # x, y = 10, 10
        col=(randint(0, 255),randint(0, 255),randint(0, 255))
        opacit=randint(245, 253)
        text = names2[randint(0,2)]
        w, h = font.getsize(text)
        draw.text((x, y), text,  font=font,  fill=opacit)
        cor = (x,y, x+w,y+h)
        width = fontSize//10
        for i in range(width):
            draw.rectangle(cor, fill= None, outline=opacit)   
            cor = (cor[0]-1,cor[1]+1, cor[2]+1,cor[3]+1) 
    w=txt.rotate(randint(0, 90),  expand=1)
    # .paste(img2, box, mask)
    # ImageOps.colorize() paint color   
    if(randint(0,1)>0.5):
        im.paste( ImageOps.colorize(w, 'white', 'black'), (0,0),  w)
    else:
        im.paste( ImageOps.colorize(w, (randint(0, 255),randint(0, 255),randint(0, 255)), (randint(0, 255),randint(0, 255),randint(0, 255))), (0,0),  w)
    return im

def shadow(src, shawdow_src):
    options = [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]
    overlap_rate = randint(10,50)/100

    origin = cv2.imread(src)
    shadow = cv2.imread(shawdow_src)
    (h,w,c) = origin.shape
    shadow = cv2.imread(shawdow_src)
    if(randint(0,3) != 0):
        shadow = cv2.resize(cv2.rotate(shadow, options[randint(0,2)]), (w, h))
    else:
        shadow = cv2.resize(shadow, (w,h))
    #M = cv2.getRotationMatrix2D((0, 0), 180, 1.0)
    #rotated = cv2.warpAffine(shadow, M, (w, h))

    # overlap = cv2.addWeighted(origin, 0.7, shadow, 0.3, 0)
    overlap = cv2.addWeighted(origin, 1-overlap_rate, shadow, overlap_rate, 0)
    return overlap

def blur(image_path):
    img = Image.open(image_path)
    k = Kernel((randint(10,100), randint(10,100)), intensity = randint(0,10)/10)

    blurred = k.applyTo(img, keep_image_dim=True)
    return blurred

def average_hw (src):
    # print the average height and width of all the images in src
    imgs = [src + i for i in os.listdir(src)]
    sumh = 0
    sumw = 0
    nums = len(imgs)
    for i in imgs:
        h,w,c = cv2.imread(i).shape
        sumh += h
        sumw += w
    print("average w:{:.2f} average h:{:.2f}".format(sumw/nums, sumh/nums))
    return 0

def get_properties(src):
    img = cv2.imread(src, cv2.IMREAD_UNCHANGED)
    # print(max(img[:,:,3]))
    print(src, img.shape)
    return 0

def get_img_path_randomly(src):
    path_list = os.listdir(src)
    nums = len(path_list)
    return src + path_list[random.randint(0, nums-1)]
def rename_file_according2order(src):
    file_list = [src + i for i in os.listdir(src)]
    for idx, i in enumerate(file_list):
        #postfix = i.split('.')[-1]
        os.rename(i, ''.join([src, 'background', str(idx), '.jpg']))
def generate_background():
    background_src_path = '/home/eescut/Desktop/DR_Datasets/background_texture/'
    dataset_list = ['/home/eescut/Desktop/DR_Datasets/DIBCO2010/GT/']
    rate = 23
    for dataset_path in dataset_list:
        img_paths = [dataset_path+i for i in os.listdir(dataset_path)]
        for img_path in img_paths:
            for i in range(rate):
                back = get_img_path_randomly(background_src_path)
                alpha_mix(img_path, back)
def generate_noise():
    dataset_list = ['/home/eescut/Desktop/DR_Datasets/DIBCO2013/'
                    , '/home/eescut/Desktop/DR_Datasets/DIBCO2014/'
                    , '/home/eescut/Desktop/DR_Datasets/DIBCO2016/'
                    , '/home/eescut/Desktop/DR_Datasets/NoisyOffice/'
                    , '/home/eescut/Desktop/DR_Datasets/PHIBD/'
                    , '/home/eescut/Desktop/DR_Datasets/SMADI/']

    for idx,dataset_path in enumerate(dataset_list):
        withBack_path = dataset_path + 'withBack/'
        img_names = os.listdir(withBack_path)
        img_paths = [withBack_path+i for i in img_names]
        target = dataset_path + "noise/"
        if(not os.path.exists(target)):
            os.mkdir(target)
        for idy, img_path in enumerate(img_paths):
            print(img_path)
            img = cv2.imread(img_path)
            noisy_005 = noisy(img, 0.05)
            noisy_01 = noisy(img, 0.1)
            noisy_02 = noisy(img, 0.2)
            cv2.imwrite(''.join([target, img_names[idy][:-4], '_n0.05.jpg']), noisy_005)
            cv2.imwrite(''.join([target, img_names[idy][:-4], '_n0.1.jpg']), noisy_01)
            cv2.imwrite(''.join([target, img_names[idy][:-4], '_n0.2.jpg']), noisy_02)
def generate_watermark():
    dataset_list = ['/home/eescut/Desktop/DR_Datasets/DIBCO2009/'
                    , '/home/eescut/Desktop/DR_Datasets/DIBCO2009/'
                    , '/home/eescut/Desktop/DR_Datasets/DIBCO2013/'
                    , '/home/eescut/Desktop/DR_Datasets/DIBCO2014/'
                    , '/home/eescut/Desktop/DR_Datasets/DIBCO2016/'
                    , '/home/eescut/Desktop/DR_Datasets/NoisyOffice/'
                    , '/home/eescut/Desktop/DR_Datasets/PHIBD/'
                    , '/home/eescut/Desktop/DR_Datasets/SMADI/'
                    , '/home/eescut/Desktop/DR_Datasets/BickelyDiary/']

    for idx,dataset_path in enumerate(dataset_list):
        withBack_path = dataset_path + 'withBack/'
        img_names = os.listdir(withBack_path)
        img_paths = [withBack_path+i for i in img_names]
        target = dataset_path + "watermark/"
        if(not os.path.exists(target)):
            os.mkdir(target)
        for idy, img_path in enumerate(img_paths):
            print(img_path)
            for i in range(3):
                watermarked = watermark(img_path)
                watermarked.save(''.join([target, img_names[idy][:-4], "_w", str(i), ".jpg"]))   
def generate_shadow():
    dataset_list = ['/home/eescut/Desktop/DR_Datasets/DIBCO2010/']
    '''
    dataset_list = ['/home/eescut/Desktop/DR_Datasets/DIBCO2013/'
                    , '/home/eescut/Desktop/DR_Datasets/DIBCO2014/'
                    , '/home/eescut/Desktop/DR_Datasets/DIBCO2016/'
                    , '/home/eescut/Desktop/DR_Datasets/NoisyOffice/'
                    , '/home/eescut/Desktop/DR_Datasets/PHIBD/'
                    , '/home/eescut/Desktop/DR_Datasets/SMADI/'
                    , '/home/eescut/Desktop/DR_Datasets/BickelyDiary/']
    '''
    shadow_list = ['./shadow_masks/shadow_mask0.jpg'
                   , './shadow_masks/shadow_mask1.jpg'
                   , './shadow_masks/shadow_mask2.jpg'
                   , './shadow_masks/shadow_mask3.jpg'
                   , './shadow_masks/shadow_mask4.jpg']
    for idx,dataset_path in enumerate(dataset_list):
        withBack_path = dataset_path + 'withBack/'
        img_names = os.listdir(withBack_path)
        img_paths = [withBack_path+i for i in img_names]
        target = dataset_path + "shadow/"
        if(not os.path.exists(target)):
            os.mkdir(target)
        for idy, img_path in enumerate(img_paths):
            print(img_path)
            for i in range(3):
                withshadow = shadow(img_path, shadow_list[randint(0,4)])
                cv2.imwrite(''.join([target, img_names[idy][:-4], "_s", str(i), ".jpg"]), withshadow)
def generate_blur():
    dataset_list = ['/home/eescut/Desktop/DR_Datasets/DIBCO2010/']
    '''
    dataset_list = ['/home/eescut/Desktop/DR_Datasets/DIBCO2013/'
                    , '/home/eescut/Desktop/DR_Datasets/DIBCO2014/'
                    , '/home/eescut/Desktop/DR_Datasets/DIBCO2016/'
                    , '/home/eescut/Desktop/DR_Datasets/NoisyOffice/'
                    , '/home/eescut/Desktop/DR_Datasets/PHIBD/'
                    , '/home/eescut/Desktop/DR_Datasets/SMADI/'
                    , '/home/eescut/Desktop/DR_Datasets/BickelyDiary/']
    '''
    for idx,dataset_path in enumerate(dataset_list):
        withBack_path = dataset_path + 'withBack/'
        img_names = os.listdir(withBack_path)
        img_paths = [withBack_path+i for i in img_names]
        target = dataset_path + "blur/"
        if(not os.path.exists(target)):
            os.mkdir(target)
        for idy, img_path in enumerate(img_paths):
            print(img_path)
            for i in range(3):
                blurred = blur(img_path)
                blurred.save(''.join([target, img_names[idy][:-4], "_b", str(i), ".jpg"]))
def generate_hybrid():
    dataset_list = ['/home/eescut/Desktop/DR_Datasets/DIBCO2009/'
                    , '/home/eescut/Desktop/DR_Datasets/DIBCO2010/'
                    , '/home/eescut/Desktop/DR_Datasets/DIBCO2013/'
                    , '/home/eescut/Desktop/DR_Datasets/DIBCO2014/'
                    , '/home/eescut/Desktop/DR_Datasets/DIBCO2016/'
                    , '/home/eescut/Desktop/DR_Datasets/NoisyOffice/'
                    , '/home/eescut/Desktop/DR_Datasets/PHIBD/'
                    , '/home/eescut/Desktop/DR_Datasets/SMADI/'
                    , '/home/eescut/Desktop/DR_Datasets/BickelyDiary/']
    dataset_name = [re.split('[/]',i)[-2] for i in dataset_list]
    print(dataset_name)
    x_name = ['blur/', 'noise/', 'shadow/', 'watermark/', 'withBack/', 'GT/']
    y_name = ['Degraded/Blur/', 'Degraded/Noise/', 'Degraded/Shadow/', 'Degraded/Watermark/'
             , 'Degraded/WithBack/', 'GT/']
    # target_root = './Hybrid/'
    target_root = './Hybrid_1.0/'
    for idd,dataset in enumerate(dataset_list):
        subpath_list = [dataset + i for i in x_name]
        targetpath_list = [target_root + i for i in y_name]
        for idx, subpath in enumerate(subpath_list):
            filename_list = os.listdir(subpath)
            xfilepath_list = [subpath + i for i in filename_list]
            yfilepath_list = [''.join([targetpath_list[idx], dataset_name[idd],'_',i]) for i in filename_list]
            for idf, xfilepath in enumerate(xfilepath_list):
                print(xfilepath, yfilepath_list[idf])
                shutil.copyfile(xfilepath, yfilepath_list[idf])
def generate_corresponding_txt():
    GTrootpath = '/home/eescut/Desktop/DR_Datasets/Hybrid/GT/'
    degraded_path_list = ['/home/eescut/Desktop/DR_Datasets/Hybrid/Degraded/Blur/'
                    , '/home/eescut/Desktop/DR_Datasets/Hybrid/Degraded/Noise/'
                    , '/home/eescut/Desktop/DR_Datasets/Hybrid/Degraded/Shadow/'
                    , '/home/eescut/Desktop/DR_Datasets/Hybrid/Degraded/Watermark/'
                    , '/home/eescut/Desktop/DR_Datasets/Hybrid/Degraded/WithBack/']
    GT_files = os.listdir(GTrootpath)
    GT_paths = [GTrootpath + i for i in GT_files]
    with open('/home/eescut/Desktop/DR_Datasets/Fine_tune/fine_tune.txt', 'w') as f:
        for GT_file in GT_files:
            GT_name = GT_file.split('.')[0]
            for degraded_path in degraded_path_list:
                target = GTrootpath + GT_file
                input_img = ''

                files = os.listdir(degraded_path)
                corresponding_files = [i for i in files if(GT_name in i)]
                for corresponding_flie in corresponding_files:
                    content = ''.join([degraded_path, corresponding_flie, '|', GTrootpath, GT_file, '\n'])
                    print(content)
                    f.write(content)

def generate_mutitask_corresponding_txt():
    '''
    Target line form : path_GT|path_back|path_nosie|path_watermark|path_blur|path_shadow
    '''
    GTrootpath = '/home/eescut/Desktop/DR_Datasets/Hybrid_1.0/GT/'
    degraded_path_list = ['/home/eescut/Desktop/DR_Datasets/Hybrid_1.0/Degraded/Blur/'
                    , '/home/eescut/Desktop/DR_Datasets/Hybrid_1.0/Degraded/Noise/'
                    , '/home/eescut/Desktop/DR_Datasets/Hybrid_1.0/Degraded/Shadow/'
                    , '/home/eescut/Desktop/DR_Datasets/Hybrid_1.0/Degraded/Watermark/'
                    , '/home/eescut/Desktop/DR_Datasets/Hybrid_1.0/Degraded/WithBack/']
    Back_files = os.listdir(degraded_path_list[4])
    Back_paths = [degraded_path_list[4] + i for i in Back_files]
    with open('/home/eescut/Desktop/DR_Datasets/multi_task/hy1.txt', 'w') as f:
        for Back_file in Back_files:
            Back_name = Back_file.split('.')[0]
            # extract GT name from Withback
            GT_name = ''
            if(Back_name[-2] == '_'):
                GT_name = Back_file.split('.')[0][:-2]
            else:
                GT_name = Back_file.split('.')[0][:-3]
            for item in os.listdir(GTrootpath):
                if(GT_name in item):
                    GT_name = item
            GT_file = GT_name
            print(GT_file, Back_name)
            # create pairs
            blur_postfix = ['_b0.jpg', '_b1.jpg', '_b2.jpg']
            noise_postfix = ['_n0.1.jpg', '_n0.2.jpg', '_n0.05.jpg']
            shadow_postfix = ['_s0.jpg', '_s1.jpg', '_s2.jpg']
            watermark_postfix = ['_w0.jpg', '_w1.jpg', '_w2.jpg']
            for i in range(3):
                blur_file = Back_name + blur_postfix[i]
                noise_file = Back_name + noise_postfix[i]
                shadow_file = Back_name + shadow_postfix[i]
                watermark_file = Back_name + watermark_postfix[i]
                print(blur_file, noise_file, shadow_file, watermark_file)
                content = ''.join([GTrootpath, GT_file, '|', 
                                degraded_path_list[4], Back_file, '|', 
                                degraded_path_list[0], blur_file, '|',
                                degraded_path_list[1], noise_file, '|',
                                degraded_path_list[2], shadow_file, '|',
                                degraded_path_list[3], watermark_file, '\n'])
                f.write(content)
            '''
            # create corresponding line
            for degraded_path in degraded_path_list[0:4]:
                target = GTrootpath + GT_file
                input_img = ''

                files = os.listdir(degraded_path)
                corresponding_files = [i for i in files if(GT_name in i)]
                for corresponding_flie in corresponding_files:
                    content = ''.join([degraded_path, corresponding_flie, '|', GTrootpath, GT_file, '\n'])
                    print(content)
                    f.write(content)
            '''

def generate_subfiles():
    assign_rate = [0,24,32,33]
    summ = 33
    input_path = '/home/eescut/Desktop/DR_Datasets/Hybrid/hybrid.txt'
    output_paths = ['/home/eescut/Desktop/DR_Datasets/Hybrid/hybrid_train.txt'
                   , '/home/eescut/Desktop/DR_Datasets/Hybrid/hybrid_test.txt'
                   , '/home/eescut/Desktop/DR_Datasets/Hybrid/hybrid_validation.txt']
    with open(input_path) as f:
        content = f.readlines()
        random.shuffle(content)
        lenghth = len(content)
        for idx, output_path in enumerate(output_paths):
            output = open(output_path, 'w')
            start = int(assign_rate[idx]/summ*lenghth)
            end = int(assign_rate[idx+1]/summ*lenghth)
            print(start, end)
            output.write(''.join(content[start:end]))
def split_train_val_test():
    # Our task is to generate image path pairs like below :
    # eg. a sample line from groups_test_mixSICEV2.txt
    # /home/jieh/Dataset/Continous/Exposure/test/input/Exp5/a5000-kme_0204.jpg|/home/jieh/Dataset/Continous/Exposure/test/target/a5000-kme_0204.jpg
    # 3:1, 8:1 -> 24:8:1
    root_list = ['/home/eescut/Desktop/DR_Datasets/DIBCO2009/'
                    , '/home/eescut/Desktop/DR_Datasets/DIBCO2014/'
                    , '/home/eescut/Desktop/DR_Datasets/DIBCO2016/']
    gt_root_list = [i+'GT/' for i in root_list]
    ori_root_list = [i+'Original/' for i in root_list]
    gt_files = []
    for i in gt_root_list:
        sub_files = [i+j for j in sorted(os.listdir(i), key = lambda x:x.split('.')[0])]
        gt_files += sub_files
    ori_files = []
    for i in ori_root_list:
        sub_files = [i+j for j in sorted(os.listdir(i), key = lambda x:x.split('.')[0])]
        ori_files += sub_files
    for i in range(50):
        print(gt_files[i], ori_files[i])
    
    with open('/home/eescut/Desktop/DR_Datasets/Fine_tune/fine_tune.txt', 'w') as f:
        for idx,GT_file in enumerate(gt_files):

            content = ''.join([ori_files[idx], '|', GT_file, '\n'])
            print(content)
            f.write(content)

def generate_finetune_txt(root, target_root):
    dataset_roots = [root+i for i in ['DIBCO2009/','DIBCO2010/','DIBCO2013/','DIBCO2014/','DIBCO2016/']]
    ori_files = []
    gt_files = []
    for dataset in dataset_roots:
        gts = [dataset+'GT/'+i for i in sorted(os.listdir(dataset+'GT/'))]
        origins = [dataset+'Original/'+i for i in sorted(os.listdir(dataset+'Original/'))]
        print(gts)
        print(origins)
        gt_files += gts
        ori_files += origins
    print(ori_files)
    print(gt_files)
    with open(target_root, 'w') as f:
        for idx,gt_file in enumerate(gt_files):
            content = ''.join([ori_files[idx], '|', gt_files[idx], '\n'])
            print(content)
            f.write(content)
    return 0

if __name__ == "__main__":
    # generate_finetune_txt('/home/eescut/Desktop/DR_Datasets/','/home/eescut/Desktop/DR_Datasets/Fine_tune/fine_tune_09.txt')
    generate_mutitask_corresponding_txt()
    '''
    get_properties('/home/eescut/Desktop/DR_Datasets/Hybrid/Degraded/Noise/DIBCO2016_7_gt_51_n0.2.jpg')
    get_properties('/home/eescut/Desktop/DR_Datasets/Hybrid/GT/DIBCO2016_7_gt.bmp')

    for output_path in output_path:
        with open(output_path) as out:
            out.write()

    ori_path = ["./PHIBD/GT/Persian06GT.png", "./DIBCO2016/Original/6.bmp"]
    back_path = ["./backgrounds/grunge-stained-old-paper-texture.jpg"
    , "./backgrounds/image-from-rawpixel-id-377674-jpeg.jpg"
    , "./backgrounds/old-paper-background-texture-6.jpg"
    , "./backgrounds/old-paper-background-texture-34.jpg"]
    #alpha_mix(ori_path[1], back_path[3])
    add_shadow("./with_shadow0.jpg", "./outImg2.jpg")
    '''
    
    
    