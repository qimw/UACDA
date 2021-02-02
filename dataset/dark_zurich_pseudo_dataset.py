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
from PIL import Image,ImageFile
from dataset.autoaugment import ImageNetPolicy
from dataset.all_file_paths import all_file_paths

ImageFile.LOAD_TRUNCATED_IMAGES = True

class DarkZurichPseudoDataSet(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, resize_size=(1024, 512), crop_size=(512, 1024), mean=(128, 128, 128), scale=False, mirror=True, ignore_label=255, set='day', autoaug=False, synthia=False, threshold = 1.0):
        self.root = root
        if 'day' in set:
            self.imgs_files_txt = all_file_paths['zurich_day_imgs_txt']
            self.lbls_files_txt = all_file_paths['zurich_day_plbls_txt']
        elif 'twilight' in set:
            self.imgs_files_txt = all_file_paths['zurich_twilight_imgs_txt']
            self.lbls_files_txt = all_file_paths['zurich_twilight_plbls_txt']
        elif 'night' in set:
            self.imgs_files_txt = all_file_paths['zurich_night_imgs_txt']
            self.lbls_files_txt = all_file_paths['zurich_night_plbls_txt']
        else:
            raise NotImplementedError()
        self.files = self.get_lines(self.imgs_files_txt)
        self.gts = [os.path.join('./data', set, _) for _ in self.get_lines(self.lbls_files_txt)]

        self.crop_size = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirror
        self.resize_size = resize_size
        self.autoaug = autoaug
        self.h = crop_size[0]
        self.w = crop_size[1]
        # self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        if not max_iters==None:
            self.files = self.files * int(np.ceil(float(max_iters) / len(self.files)))
            self.gts = self.gts * int(np.ceil(float(max_iters) / len(self.gts)))
        assert len(self.files) == len(self.gts)
        self.set = set
         
        #https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py
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
        label = Image.open(lbl_path).convert('L')
        name = os.path.basename(img_path)
        # resize
        if self.scale:
            random_scale = 0.8 + random.random()*0.4 # 0.8 - 1.2
            image = image.resize( ( round(self.resize_size[0] * random_scale), round(self.resize_size[1] * random_scale)) , Image.BICUBIC)
            label = label.resize( ( round(self.resize_size[0] * random_scale), round(self.resize_size[1] * random_scale)) , Image.NEAREST)
        else:
            image = image.resize( ( self.resize_size[0], self.resize_size[1] ) , Image.BICUBIC)
            label = label.resize( ( self.resize_size[0], self.resize_size[1] ) , Image.NEAREST)
        if self.autoaug:
            policy = ImageNetPolicy()
            image = policy(image)
        image = np.asarray(image, np.float32)
        label = np.asarray(label, np.uint8)

        # re-assign labels to match the format of Cityscapes
        #label_copy = 255 * np.ones(label.shape, dtype=np.float32)
        #for k, v in list(self.id_to_trainid.items()):
        #    label_copy[label == k] = v
        label_copy = label

        size = image.shape
        image = image[:, :, ::-1]  # change to BGR
        image -= self.mean
        image = image.transpose((2, 0, 1))
        #print(image.shape, label.shape)
        for i in range(10): #find hard samples
            x1 = random.randint(0, image.shape[1] - self.h)
            y1 = random.randint(0, image.shape[2] - self.w)
            tmp_label_copy = label_copy[x1:x1+self.h, y1:y1+self.w]
            tmp_image = image[:, x1:x1+self.h, y1:y1+self.w]
            u =  np.unique(tmp_label_copy)
            if len(u) > 10:
                break
            #else:
                #print('Cityscape-Pseudo: Too young too naive for %d times!'%i)
        image = tmp_image
        label_copy = tmp_label_copy

        if self.is_mirror and random.random() < 0.5:
            image = np.flip(image, axis = 2)
            label_copy = np.flip(label_copy, axis = 1)
        return image.copy(), label_copy.copy(), np.array(size), name

