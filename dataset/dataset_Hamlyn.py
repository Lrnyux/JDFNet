import torch
import argparse
import logging
import os
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import cv2
from PIL import Image
from torchvision import transforms, utils
import os
from skimage import transform
from utils.co_transform import *

class Hamlyn_dataset_depth(Dataset):
    def __init__(self,args,stage='Train'):
        self.args = args
        self.stage = stage
        if self.stage == 'Train':
            self.img_root = args.train_img_root
        if self.stage == 'Val':
            self.img_root = args.val_img_root
        if self.stage == 'Test':
            self.img_root = args.test_img_root

        self.img_l_filelist = []
        self.img_r_filelist = []


        sub_file = os.path.join(self.img_root,'image_0')
        sub_listdir = sorted(os.listdir(sub_file))
        len_sub_listdir = len(sub_listdir)
        for jdx in range(len_sub_listdir):
            img_l_root = os.path.join(self.img_root,'image_0',sub_listdir[jdx])
            img_r_root = os.path.join(self.img_root,'image_1',sub_listdir[jdx])
            self.img_l_filelist.append(img_l_root)
            self.img_r_filelist.append(img_r_root)

        assert len(self.img_r_filelist) == len(self.img_l_filelist)

        self.input_transform = transforms.Compose([
            transforms.ColorJitter(brightness=(0.8, 1.2), contrast=(0.8, 1.2), saturation=(0.8, 1.2), hue=(-0.1, 0.1))
        ])

        self.co_transform = []
        if args.co_transform[0]:
            self.co_transform.append(RandomTranslate(10.))
        if args.co_transform[1]:
            self.co_transform.append(RandomRotate(10, 5))
        if args.co_transform[2]:
            self.co_transform.append(RandomVerticalFlip())
        if args.co_transform[3]:
            self.co_transform.append(RandomHorizontalFlip())
        if args.co_transform[4]:
            self.co_transform.append(RandomSwap())
        self.co_transform = Compose(self.co_transform)

        self.Resize_layer = transforms.Resize([self.args.image_size[1], self.args.image_size[0]])

    def __len__(self):
        return len(self.img_l_filelist)

    def __getitem__(self, index):
        img_l_root = self.img_l_filelist[index]
        img_r_root = self.img_r_filelist[index]

        img_l_ori = np.array(Image.open(img_l_root)) / 255.0
        img_r_ori = np.array(Image.open(img_r_root)) / 255.0

        h_ori, w_ori, c = img_l_ori.shape

        img_l_ori_pad = np.zeros((h_ori + 64, w_ori, c))
        img_r_ori_pad = np.zeros((h_ori + 64, w_ori, c))
        img_l_ori_pad[32:-32,:,:] = img_l_ori
        img_r_ori_pad[32:-32,:,:] = img_r_ori

        img_l_ori = np.copy(img_l_ori_pad)
        img_r_ori = np.copy(img_r_ori_pad)


        img_l_info = transform.resize(img_l_ori, [self.args.image_size[1], self.args.image_size[0], c])
        img_r_info = transform.resize(img_r_ori, [self.args.image_size[1], self.args.image_size[0], c])

        if self.stage == 'Train':
            img_inputs = [img_l_info, img_r_info]
            img_inputs, _ = self.co_transform(img_inputs, None)
            img_l_info = img_inputs[0]
            img_r_info = img_inputs[1]

        img_l_info = np.transpose(img_l_info, [2, 0, 1])
        img_r_info = np.transpose(img_r_info, [2, 0, 1])
        img_l_ori = np.transpose(img_l_ori, [2, 0, 1])
        img_r_ori = np.transpose(img_r_ori, [2, 0, 1])
        img_l_info = torch.from_numpy(img_l_info).to(torch.float32)
        img_r_info = torch.from_numpy(img_r_info).to(torch.float32)
        img_l_info = self.Resize_layer(img_l_info)
        img_r_info = self.Resize_layer(img_r_info)

        img_l_ori = torch.from_numpy(img_l_ori).to(torch.float32)
        img_r_ori = torch.from_numpy(img_r_ori).to(torch.float32)

        if self.stage == 'Train':
            img_l_info = self.input_transform(img_l_info)
            img_r_info = self.input_transform(img_r_info)

        sample = {}
        sample['imgl'] = img_l_info
        sample['imgr'] = img_r_info
        sample['imgl_ori'] = img_l_ori
        sample['imgr_ori'] = img_r_ori
        sample['imgl_root'] = img_l_root
        sample['imgr_root'] = img_r_root

        return sample


class Hamlyn_dataset_flow(Dataset):
    def __init__(self,args,stage='Train'):
        self.args = args
        self.stage = stage
        self.step = args.step
        if self.stage == 'Train':
            self.img_root = args.train_img_root
        if self.stage == 'Val':
            self.img_root = args.val_img_root
        if self.stage == 'Test':
            self.img_root = args.test_img_root

        self.img_sta_filelist = []
        self.img_end_filelist = []


        sub_file = os.path.join(self.img_root,'image_0')
        sub_listdir = sorted(os.listdir(sub_file))
        len_sub_listdir = len(sub_listdir)
        for jdx in range(self.step+1,len_sub_listdir-self.step-1):
            img_sta_root = os.path.join(sub_file,sub_listdir[jdx])
            img_end_root = os.path.join(sub_file,sub_listdir[jdx+self.step])
            self.img_sta_filelist.append(img_sta_root)
            self.img_end_filelist.append(img_end_root)

        assert len(self.img_sta_filelist) == len(self.img_end_filelist)

        self.input_transform = transforms.Compose([
            transforms.ColorJitter(brightness=(0.8, 1.2), contrast=(0.8, 1.2), saturation=(0.8, 1.2), hue=(-0.1, 0.1))
        ])

        self.co_transform = []
        if args.co_transform[0]:
            self.co_transform.append(RandomTranslate(10.))
        if args.co_transform[1]:
            self.co_transform.append(RandomRotate(10, 5))
        if args.co_transform[2]:
            self.co_transform.append(RandomVerticalFlip())
        if args.co_transform[3]:
            self.co_transform.append(RandomHorizontalFlip())
        if args.co_transform[4]:
            self.co_transform.append(RandomSwap())
        self.co_transform = Compose(self.co_transform)

        self.Resize_layer = transforms.Resize([self.args.image_size[1], self.args.image_size[0]])

    def __len__(self):
        return len(self.img_sta_filelist)

    def __getitem__(self, index):
        img_sta_root = self.img_sta_filelist[index]
        img_end_root = self.img_end_filelist[index]

        img_sta_ori = np.array(Image.open(img_sta_root)) / 255.0
        img_end_ori = np.array(Image.open(img_end_root)) / 255.0
        h_ori, w_ori, c = img_sta_ori.shape

        img_sta_ori_pad = np.zeros((h_ori + 64, w_ori, c))
        img_end_ori_pad = np.zeros((h_ori + 64, w_ori, c))
        img_sta_ori_pad[32:-32, :, :] = img_sta_ori
        img_end_ori_pad[32:-32, :, :] = img_end_ori

        img_sta_ori = np.copy(img_sta_ori_pad)
        img_end_ori = np.copy(img_end_ori_pad)

        # img_sta_info = np.array(Image.open(img_sta_root).resize((self.args.image_size[0], self.args.image_size[1]))) / 255.0
        # img_end_info = np.array(Image.open(img_end_root).resize((self.args.image_size[0], self.args.image_size[1]))) / 255.0

        img_sta_info = transform.resize(img_sta_ori, [self.args.image_size[1], self.args.image_size[0],c])
        img_end_info = transform.resize(img_end_ori, [self.args.image_size[1], self.args.image_size[0],c])



        if self.stage == 'Train':
            img_inputs = [img_sta_info, img_end_info]
            img_inputs, _ = self.co_transform(img_inputs, None)
            img_sta_info = img_inputs[0]
            img_end_info = img_inputs[1]

        img_sta_info = np.transpose(img_sta_info, [2, 0, 1])
        img_end_info = np.transpose(img_end_info, [2, 0, 1])
        img_sta_ori = np.transpose(img_sta_ori, [2, 0, 1])
        img_end_ori = np.transpose(img_end_ori, [2, 0, 1])
        img_sta_info = torch.from_numpy(img_sta_info).to(torch.float32)
        img_end_info = torch.from_numpy(img_end_info).to(torch.float32)
        img_sta_info = self.Resize_layer(img_sta_info)
        img_end_info = self.Resize_layer(img_end_info)

        img_sta_ori = torch.from_numpy(img_sta_ori).to(torch.float32)
        img_end_ori = torch.from_numpy(img_end_ori).to(torch.float32)

        if self.stage == 'Train':
            img_sta_info = self.input_transform(img_sta_info)
            img_sta_info = self.input_transform(img_sta_info)

        sample = {}
        sample['imgl'] = img_sta_info
        sample['imgr'] = img_end_info
        sample['imgl_ori'] = img_sta_ori
        sample['imgr_ori'] = img_end_ori
        sample['imgl_root'] = img_sta_root
        sample['imgr_root'] = img_end_root

        return sample

class Hamlyn_dataset(Dataset):
    # This dataset return [L_t, R_t, L_t+T, R_t+T]
    def __init__(self, args, stage='Train'):
        self.args = args
        self.stage = stage
        self.step = args.step
        if self.stage == 'Train':
            self.img_root = args.train_img_root
        if self.stage == 'Val':
            self.img_root = args.val_img_root
        if self.stage == 'Test':
            self.img_root = args.test_img_root

        self.img_l_sta_filelist = []
        self.img_l_end_filelist = []
        self.img_r_sta_filelist = []
        self.img_r_end_filelist = []

        self.img_l_sta_spec_filelist = []
        self.img_l_end_spec_filelist = []
        self.img_r_sta_spec_filelist = []
        self.img_r_end_spec_filelist = []

        sub_file_left = os.path.join(self.img_root, 'image_0')
        sub_file_right = os.path.join(self.img_root, 'image_1')
        sub_file_left_spec = os.path.join(self.img_root,'image_0_despecular')
        sub_file_right_spec = os.path.join(self.img_root, 'image_1_despecular')

        sub_listdir = sorted(os.listdir(sub_file_left))
        len_sub_listdir = len(sub_listdir)
        for jdx in range(self.step + 1, len_sub_listdir - self.step - 1):
            img_l_sta_root = os.path.join(sub_file_left, sub_listdir[jdx])
            img_l_end_root = os.path.join(sub_file_left, sub_listdir[jdx + self.step])
            img_r_sta_root = os.path.join(sub_file_right, sub_listdir[jdx])
            img_r_end_root = os.path.join(sub_file_right, sub_listdir[jdx + self.step])

            img_l_sta_spec_root = os.path.join(sub_file_left_spec, sub_listdir[jdx])
            img_l_end_spec_root = os.path.join(sub_file_left_spec, sub_listdir[jdx + self.step])
            img_r_sta_spec_root = os.path.join(sub_file_right_spec, sub_listdir[jdx])
            img_r_end_spec_root = os.path.join(sub_file_right_spec, sub_listdir[jdx + self.step])

            self.img_l_sta_filelist.append(img_l_sta_root)
            self.img_l_end_filelist.append(img_l_end_root)
            self.img_r_sta_filelist.append(img_r_sta_root)
            self.img_r_end_filelist.append(img_r_end_root)
            self.img_l_sta_spec_filelist.append(img_l_sta_spec_root)
            self.img_l_end_spec_filelist.append(img_l_end_spec_root)
            self.img_r_sta_spec_filelist.append(img_r_sta_spec_root)
            self.img_r_end_spec_filelist.append(img_r_end_spec_root)

        assert len(self.img_l_sta_filelist) == len(self.img_l_end_filelist)
        assert len(self.img_r_sta_filelist) == len(self.img_r_end_filelist)
        assert len(self.img_l_sta_filelist) == len(self.img_r_sta_filelist)
        assert len(self.img_l_sta_spec_filelist) == len(self.img_r_end_spec_filelist)

        # self.input_transform = transforms.Compose([
        #     transforms.ColorJitter(brightness=(0.8, 1.2), contrast=(0.8, 1.2), saturation=(0.8, 1.2), hue=(-0.1, 0.1))
        # ])

        self.Resize_layer = transforms.Resize([self.args.image_size[1], self.args.image_size[0]])

    def __len__(self):
        return len(self.img_l_sta_filelist)

    def __getitem__(self, index):
        img_l_sta_root = self.img_l_sta_filelist[index]
        img_l_end_root = self.img_l_end_filelist[index]
        img_r_sta_root = self.img_r_sta_filelist[index]
        img_r_end_root = self.img_r_end_filelist[index]

        img_l_sta_spec_root = self.img_l_sta_spec_filelist[index]
        img_l_end_spec_root = self.img_l_end_spec_filelist[index]
        img_r_sta_spec_root = self.img_r_sta_spec_filelist[index]
        img_r_end_spec_root = self.img_r_end_spec_filelist[index]

        img_l_sta_ori = np.array(Image.open(img_l_sta_root)) / 255.0
        img_l_end_ori = np.array(Image.open(img_l_end_root)) / 255.0
        img_r_sta_ori = np.array(Image.open(img_r_sta_root)) / 255.0
        img_r_end_ori = np.array(Image.open(img_r_end_root)) / 255.0

        img_l_sta_spec_ori = np.array(Image.open(img_l_sta_spec_root)) / 255.0
        img_l_end_spec_ori = np.array(Image.open(img_l_end_spec_root)) / 255.0
        img_r_sta_spec_ori = np.array(Image.open(img_r_sta_spec_root)) / 255.0
        img_r_end_spec_ori = np.array(Image.open(img_r_end_spec_root)) / 255.0


        h_ori, w_ori, c = img_l_sta_ori.shape

        img_l_sta_ori_pad = np.zeros((h_ori + 64, w_ori, c))
        img_l_end_ori_pad = np.zeros((h_ori + 64, w_ori, c))
        img_r_sta_ori_pad = np.zeros((h_ori + 64, w_ori, c))
        img_r_end_ori_pad = np.zeros((h_ori + 64, w_ori, c))

        img_l_sta_ori_pad[32:-32, :, :] = img_l_sta_ori
        img_l_end_ori_pad[32:-32, :, :] = img_l_end_ori
        img_r_sta_ori_pad[32:-32, :, :] = img_r_sta_ori
        img_r_end_ori_pad[32:-32, :, :] = img_r_end_ori

        img_l_sta_ori = np.copy(img_l_sta_ori_pad)
        img_l_end_ori = np.copy(img_l_end_ori_pad)
        img_r_sta_ori = np.copy(img_r_sta_ori_pad)
        img_r_end_ori = np.copy(img_r_end_ori_pad)

        img_l_sta_ori_spec_pad = np.zeros((h_ori + 64, w_ori, c))
        img_l_end_ori_spec_pad = np.zeros((h_ori + 64, w_ori, c))
        img_r_sta_ori_spec_pad = np.zeros((h_ori + 64, w_ori, c))
        img_r_end_ori_spec_pad = np.zeros((h_ori + 64, w_ori, c))

        img_l_sta_ori_spec_pad[32:-32, :, :] = img_l_sta_spec_ori
        img_l_end_ori_spec_pad[32:-32, :, :] = img_l_end_spec_ori
        img_r_sta_ori_spec_pad[32:-32, :, :] = img_r_sta_spec_ori
        img_r_end_ori_spec_pad[32:-32, :, :] = img_r_end_spec_ori

        img_l_sta_spec_ori = np.copy(img_l_sta_ori_spec_pad)
        img_l_end_spec_ori = np.copy(img_l_end_ori_spec_pad)
        img_r_sta_spec_ori = np.copy(img_r_sta_ori_spec_pad)
        img_r_end_spec_ori = np.copy(img_r_end_ori_spec_pad)








        img_l_sta_info = transform.resize(img_l_sta_ori, [self.args.image_size[1], self.args.image_size[0], c])
        img_l_end_info = transform.resize(img_l_end_ori, [self.args.image_size[1], self.args.image_size[0], c])
        img_r_sta_info = transform.resize(img_r_sta_ori, [self.args.image_size[1], self.args.image_size[0], c])
        img_r_end_info = transform.resize(img_r_end_ori, [self.args.image_size[1], self.args.image_size[0], c])

        img_l_sta_spec_info = transform.resize(img_l_sta_spec_ori,
                                               [self.args.image_size[1], self.args.image_size[0], c])
        img_l_end_spec_info = transform.resize(img_l_end_spec_ori,
                                               [self.args.image_size[1], self.args.image_size[0], c])
        img_r_sta_spec_info = transform.resize(img_r_sta_spec_ori,
                                               [self.args.image_size[1], self.args.image_size[0], c])
        img_r_end_spec_info = transform.resize(img_r_end_spec_ori,
                                               [self.args.image_size[1], self.args.image_size[0], c])

        img_info_list = [img_l_sta_info, img_l_end_info, img_r_sta_info, img_r_end_info,
                         img_l_sta_spec_info, img_l_end_spec_info, img_r_sta_spec_info, img_r_end_spec_info]
        img_info_list = [np.transpose(img_sample, [2, 0, 1]) for img_sample in img_info_list]
        img_info_list = [torch.from_numpy(img_sample).to(torch.float32) for img_sample in img_info_list]
        img_info_list = [self.Resize_layer(img_sample) for img_sample in img_info_list]

        # if self.stage == 'Train':
        #     img_info_list = [self.input_transform(img_sample) for img_sample in img_info_list]

        img_l_sta_info = img_info_list[0]
        img_l_end_info = img_info_list[1]
        img_r_sta_info = img_info_list[2]
        img_r_end_info = img_info_list[3]
        img_l_sta_spec_info = img_info_list[4]
        img_l_end_spec_info = img_info_list[5]
        img_r_sta_spec_info = img_info_list[6]
        img_r_end_spec_info = img_info_list[7]

        sample = {}
        sample['imgl_sta'] = img_l_sta_info
        sample['imgl_sta_spec'] = img_l_sta_spec_info
        sample['imgl_end'] = img_l_end_info
        sample['imgl_end_spec'] = img_l_end_spec_info
        sample['imgr_sta'] = img_r_sta_info
        sample['imgr_sta_spec'] = img_r_sta_spec_info
        sample['imgr_end'] = img_r_end_info
        sample['imgr_end_spec'] = img_r_end_spec_info
        sample['imgl_sta_root'] = img_l_sta_root
        sample['imgl_end_root'] = img_l_end_root
        sample['imgr_sta_root'] = img_r_sta_root
        sample['imgr_end_root'] = img_r_end_root

        return sample