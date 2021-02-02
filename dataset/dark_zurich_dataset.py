import os
import os.path as osp
import numpy as np
import random
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import collections
import torch
import torchvision
from torch.utils import data
from PIL import Image
from dataset.autoaugment import ImageNetPolicy
from dataset.all_file_paths import all_file_paths
import time

class DarkZurichDataSet(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, resize_size=(1024, 512), crop_size=(512, 1024), mean=(128, 128, 128), scale=True, mirror=True, ignore_label=255, set='day', autoaug=False):
        if set == 'day':
            self.imgs_file_txt = all_file_paths['zurich_day_imgs_txt']
        if set == 'night':
            self.imgs_file_txt = all_file_paths['zurich_night_imgs_txt']
        if set == 'twilight':
            self.imgs_file_txt = all_file_paths['zurich_twilight_imgs_txt']
        elif set == 'test':
            self.imgs_file_txt = all_file_paths['zurich_test_imgs_txt']
        elif set == 'val':
            self.imgs_file_txt = all_file_paths['zurich_val_imgs_txt']
        self.crop_size = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirror
        self.resize_size = resize_size
        self.autoaug = autoaug
        self.h = crop_size[0]
        self.w = crop_size[1]
        self.files = self.get_lines(self.imgs_file_txt)
        if not max_iters==None:
            self.files = self.files * int(np.ceil(float(max_iters) / len(self.files)))
        self.set = set
        # for split in ["train", "trainval", "val"]:
         
        #https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py
        '''
        project Cityscapes to Oxford Robot
        7 road -> 8;  8 sidewalk -> 7; building 11 -> 6; wall 12 -> 255;
        fence 13 -> 255; pole 17-> 255: light 19 -> 5; sign 20->4;
        vegetation -> 255; terrain -> 255; sky 23 -> 0; person 24 -> 1 ;
        rider 25 -> 1 ; car 26 -> 3; truck 27 ->3; bus 28 ->3; train 31->255;
        motorcycle 32->2 ; bike 33 -> 2;

        '''

    def get_lines(self, file_path):
        with open(file_path, 'r') as f:
            file_lines = [l.strip() for l in f.readlines()]
        file_lines.sort(key=lambda x: x.split('/')[-1])
        return file_lines

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        img_path = self.files[index]

        image = Image.open(img_path).convert('RGB')
        # resize
        image = image.resize(self.resize_size, Image.BICUBIC)
        if self.autoaug:
            policy = ImageNetPolicy()
            image = policy(image)

        image = np.asarray(image, np.float32)

        size = image.shape
        image = image[:, :, ::-1]  # change to BGR
        image -= self.mean
        image = image.transpose((2, 0, 1))
        x1 = random.randint(0, image.shape[1] - self.h)
        y1 = random.randint(0, image.shape[2] - self.w)
        image = image[:, x1:x1+self.h, y1:y1+self.w]

        if self.is_mirror and random.random() < 0.5:
            image = np.flip(image, axis = 2)
        return image.copy(), np.array(size), img_path


if __name__ == '__main__':
    dst = DarkZurichDataSet('./data/Cityscapes/data', './dataset/cityscapes_list/train.txt', mean=(0,0,0), set = 'train')
    trainloader = data.DataLoader(dst, batch_size=4)
    for i, data in enumerate(trainloader):
        imgs, _, _, _ = data
        if i == 0:
            img = torchvision.utils.make_grid(imgs).numpy()
            img = np.transpose(img, (1, 2, 0))
            img = img[:, :, ::-1]
            img = Image.fromarray(np.uint8(img) )
            img.save('Cityscape_Demo.jpg')
        break
