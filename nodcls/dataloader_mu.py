# coding=utf-8
from __future__ import print_function
from PIL import Image
import os
import os.path
import math
import numpy as np
import sys
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle
import torch
import torch.utils.data as data
import torch.nn.functional as F

# SPP(空间金字塔池化)
class SPP(torch.nn.Module):

    def __init__(self, num_levels, pool_type='max_pool'):
        super(SPP, self).__init__()

        self.num_levels = num_levels
        self.pool_type = pool_type

    def forward(self, x):
        num, c, h, w = x.size() # num:样本数量 c:通道数 h:高 w:宽
        for i in range(self.num_levels):
            level = i+1
            kernel_size = (math.ceil(h / level), math.ceil(w / level))
            stride = (math.ceil(h / level), math.ceil(w / level))
            padding = (math.floor((kernel_size[0]*level-h+1)/2), math.floor((kernel_size[1]*level-w+1)/2))

            # 选择池化方式
            if self.pool_type == 'max_pool':
                tensor = F.max_pool2d(x, kernel_size=kernel_size, stride=stride, padding=padding).view(num, -1)
            else:
                tensor = F.avg_pool2d(x, kernel_size=kernel_size, stride=stride, padding=padding).view(num, -1)

            # 展开
            if (i == 0):
                x_flatten = tensor.view(num, -1)
            else:
                x_flatten = torch.cat((x_flatten, tensor.view(num, -1)), 1)
        return x_flatten
"""
带语义信息的多尺度输入
1. 按照d计算裁剪尺寸
2. 使用mask进行屏蔽
3. 图像金字塔->30
"""
class lunanod(data.Dataset):
    def __init__(self, npypath, fnamelst, labellst, featlst, train=True,
                 transform=None, target_transform=None,mask=None,download=False):
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        _SPP = SPP(4,pool_type='max_pool')
        # now load the picked numpy arrays
        if self.train:
            self.train_data = []
            self.train_labels = []
            self.train_feat = featlst
            for label, fentry in zip(labellst, fnamelst):
                D = label[3] if label[3]%2==0 else label[3]+1 #计算裁剪尺寸
                if mask.shape[0]!=24 or mask.shape[1]!=24 or mask.shape[1]!=24:
                    print("Mask shape error!want (24,24,24),but give(%d,%d,%d)" % (mask.shape[0],mask.shape[1],mask.shape[2]))
                else:
                    #上采样到(96,96,96)
                    mask96 = self.upsample(mask)
                    # 掩码处理
                    fentry_clean = fentry.mul(mask96)
                    out30 = _SPP.forward(fentry_clean)
                if type(fentry) != 'str':
                    self.train_data.append(out30)
                    self.train_labels.append(label)
                    # print('1')
                else:
                    file = os.path.join(npypath, out30)
                    self.train_data.append(np.load(file))
                    self.train_labels.append(label)

            self.train_data = np.concatenate(self.train_data)
            self.train_data = self.train_data.reshape((len(fnamelst), 32, 32, 32))
            # self.train_labels = np.asarray(self.train_labels)
            # self.train_data = self.train_data.transpose((0, 2, 3, 4, 1))  # convert to HWZC
            self.train_len = len(fnamelst)
        else:
            self.test_data = []
            self.test_labels = []
            self.test_feat = featlst
            for label, fentry in zip(labellst, fnamelst):
                if fentry.shape[0] != 32 or fentry.shape[1] != 32 or fentry.shape[2] != 32:
                    print(fentry.shape, type(fentry), type(fentry)!='str')
                if type(fentry) != 'str':
                    self.test_data.append(fentry)
                    self.test_labels.append(label)
                    # print('1')
                else:
                    file = os.path.join(npypath, fentry)
                    self.test_data.append(np.load(file))
                    self.test_labels.append(label)
            self.test_data = np.concatenate(self.test_data)
            # print(self.test_data.shape)
            self.test_data = self.test_data.reshape((len(fnamelst), 32, 32, 32))
            # self.test_labels = np.asarray(self.test_labels)
            # self.test_data = self.test_data.transpose((0, 2, 3, 4, 1))  # convert to HWZC
            self.test_len = len(fnamelst)
            print(self.test_data.shape, len(self.test_labels), len(self.test_feat))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target, feat = self.train_data[index], self.train_labels[index], self.train_feat[index]
        else:
            img, target, feat = self.test_data[index], self.test_labels[index], self.test_feat[index]
        # img = torch.from_numpy(img) 
        # img = img.cuda(async = True)

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        # print('1', img.shape, type(img))
        # img = Image.fromarray(img)
        # print('2', img.size)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        # print(img.shape, target.shape, feat.shape)
        # print(target)

        return img, target, feat

    def __len__(self):
        if self.train:
            return self.train_len
        else:
            return self.test_len

    def upsample(self,mask24):
        m = torch.nn.Upsample(scale_factor=4, mode='nearest')
        return m(mask24)


