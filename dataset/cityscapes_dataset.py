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

class cityscapesDataSet(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, resize_size=(1024, 512), crop_size=(512, 1024), mean=(128, 128, 128), scale=True, mirror=True, ignore_label=255, set='day', autoaug=False):
        if 'day' in set:
            self.imgs_file_txt = all_file_paths['city_day_imgs_txt']
        elif 'night' in set:
            self.imgs_file_txt = all_file_paths['city_night_imgs_txt']
        elif 'twilight' in set:
            self.imgs_file_txt = all_file_paths['city_twilight_imgs_txt']
        else:
            raise NotImplementedError()
        self.gts_file_txt = all_file_paths['city_lbls_txt']
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
        self.gts = self.get_lines(self.gts_file_txt)
        if not max_iters==None:
            self.files = self.files * int(np.ceil(float(max_iters) / len(self.files)))
            self.gts = self.gts * int(np.ceil(float(max_iters) / len(self.gts)))
        assert len(self.files) == len(self.gts)
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
        self.id_to_trainid = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
                              19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
                              26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}

    def get_lines(self, file_path):
        with open(file_path, 'r') as f:
            file_lines = [l.strip() for l in f.readlines()]
        file_lines.sort(key=lambda x: x.split('/')[-1])
        return file_lines

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        img_path = self.files[index]
        lbl_path = self.gts[index]

        image = Image.open(img_path).convert('RGB')
        label = Image.open(lbl_path)
        name = os.path.basename(img_path)
        # resize
        image, label = image.resize(self.resize_size, Image.BICUBIC), label.resize(self.resize_size, Image.NEAREST)
        if self.autoaug:
            policy = ImageNetPolicy()
            image = policy(image)

        image, label = np.asarray(image, np.float32), np.asarray(label, np.uint8)

        # re-assign labels to match the format of Cityscapes
        label_copy = 255 * np.ones(label.shape, dtype=np.uint8)
        for k, v in list(self.id_to_trainid.items()):
            label_copy[label == k] = v

        size = image.shape
        image = image[:, :, ::-1]  # change to BGR
        image -= self.mean
        image = image.transpose((2, 0, 1))
        x1 = random.randint(0, image.shape[1] - self.h)
        y1 = random.randint(0, image.shape[2] - self.w)
        image = image[:, x1:x1+self.h, y1:y1+self.w]
        label_copy = label_copy[x1:x1+self.h, y1:y1+self.w]

        if self.is_mirror and random.random() < 0.5:
            image = np.flip(image, axis = 2)
            label_copy = np.flip(label_copy, axis = 1)
        return image.copy(), label_copy.copy(), np.array(size), img_path


if __name__ == '__main__':
    dst = cityscapesDataSet('./data/Cityscapes/data', './dataset/cityscapes_list/train.txt', mean=(0,0,0), set = 'train')
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
