# coding:utf-8
# Modified by Yuxiang Sun, Aug. 2, 2019
# Email: sun.yuxiang@outlook.com

import os
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image 

class MF_dataset(Dataset):

    def __init__(self, data_dir, split, have_label, input_h=320, input_w=960 ,transform=[]):
        super(MF_dataset, self).__init__()

        assert split in ['train', 'val', 'test', 'test_day', 'test_night', 'val_test'], 'split must be "train"|"val"|"test"|"test_day"|"test_night"|"val_test"'  # test_day, test_night

        with open(os.path.join(data_dir, split+'.txt'), 'r') as f:
            self.names = [name.strip() for name in f.readlines()]

        self.data_dir  = data_dir
        self.split     = split
        self.input_h   = input_h
        self.input_w   = input_w
        self.transform = transform
        self.is_train  = have_label
        self.n_data    = len(self.names)


    def read_image(self, name, folder):
        file_path = os.path.join(self.data_dir,  '%s/%s_%s.png' % (folder, folder, name))
        image = np.asarray(Image.open(file_path)) # (w,h,c)
        #image.flags.writeable = True
        return image

    def read_gray(self, name, folder):
        file_path = os.path.join(self.data_dir,  '%s/%s_%s.png' % (folder, folder, name))
        gray = np.asarray(Image.open(file_path).convert('L'))
        # image.flags.writeable = True
        return gray

    def get_train_item(self, index):

        name = self.names[index]

        RGB = self.read_image(name, 'fl_rgb')
        GRAY = self.read_gray(name, 'fl_rgb')
        thermal = self.read_image(name, 'fl_ir_aligned')
        # fuse = self.read_image(name, 'fuse')
        label = self.read_gray(name, 'fl_rgb_labels')
        # for func in self.transform:
        #     RGB, label = func(RGB, label)

        RGB = np.asarray(Image.fromarray(RGB).resize((self.input_w, self.input_h)), dtype=np.float32).transpose(
            (2, 0, 1)) / 255
        # fuse = np.asarray(Image.fromarray(fuse).resize((self.input_w, self.input_h)), dtype=np.float32).transpose(
        #     (2, 0, 1)) / 255
        # print(name,end='')
        # print(thermal.shape)
        thermal = np.asarray(Image.fromarray(thermal).resize((self.input_w, self.input_h)), dtype=np.float32).transpose(
            (2, 0, 1)) / 255
        # GRAY = np.asarray(Image.fromarray(GRAY).resize((self.input_w, self.input_h)), dtype=np.float32).transpose(
        #     (2, 0, 1)) / 255
        fuse = (thermal) / 2
        label = np.asarray(Image.fromarray(label).resize((self.input_w, self.input_h)), dtype=np.int64)
        # res = np.concatenate((image[0:3, :, :], thermal[0:1, :, :]), axis=0)
        # res = np.concatenate((res, image[3:4, :, :]), axis=0)
        res = np.concatenate((RGB, thermal[0:1, :, :]))
        res = np.concatenate((res, fuse[0:1, :, :]))

        return torch.tensor(res), torch.tensor(label), name

    def get_test_item(self, index):

        name = self.names[index]

        RGB = self.read_image(name, 'fl_rgb')
        GRAY = self.read_gray(name, 'fl_rgb')
        thermal = self.read_image(name, 'fl_ir_aligned')
        # fuse = self.read_image(name, 'fuse')
        label = self.read_gray(name, 'fl_rgb_labels')
        RGB = np.asarray(Image.fromarray(RGB).resize((self.input_w, self.input_h)), dtype=np.float32).transpose(
            (2, 0, 1)) / 255
        # fuse = np.asarray(Image.fromarray(fuse).resize((self.input_w, self.input_h)), dtype=np.float32).transpose(
        #     (2, 0, 1)) / 255
        thermal = np.asarray(Image.fromarray(thermal).resize((self.input_w, self.input_h)), dtype=np.float32).transpose(
            (2, 0, 1)) / 255
        # GRAY = np.asarray(Image.fromarray(GRAY).resize((self.input_w, self.input_h)), dtype=np.float32).transpose(
        #     (2, 0, 1)) / 255
        fuse = (thermal) / 2
        # res = np.concatenate((image[0:3, :, :], thermal[0:1, :, :]), axis=0)
        # res = np.concatenate((res, image[3:4, :, :]), axis=0)
        res = np.concatenate((RGB, thermal[0:1, :, :]))
        res = np.concatenate((res, fuse[0:1, :, :]))
        return torch.tensor(res), name


    def __getitem__(self, index):

        if self.is_train is True:
            return self.get_train_item(index)
        else: 
            return self.get_test_item (index)

    def __len__(self):
        return self.n_data

if __name__ == '__main__':
    data_dir = '../../data/MF/'
    MF_dataset()
