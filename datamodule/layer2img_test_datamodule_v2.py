# -*- coding:utf-8 -*-
import json
import os

import albumentations as A
import cv2 as cv
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from natsort import natsorted


class testJsonDatamodule(pl.LightningDataModule):
    def __init__(self, test_img_root, cond_json_path=None, image_size=(640, 400), batch_size=1,
                 num_workers=8, **kwargs):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.test_img_root = test_img_root
        with open(cond_json_path, "r") as f:
            self.cond_list = json.load(f)
        self.test_transform = A.Compose([
            A.Resize(*image_size)
        ])

    def setup(self, stage=None):
        self.test_set = unlabeled_Dataset(img_root=self.test_img_root, cond_list=self.cond_list,
                                          transform=self.test_transform)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)


class unlabeled_Dataset(Dataset):
    def __init__(self, img_root, cond_list, transform):
        assert os.path.exists(img_root)

        self.transform = transform
        self.img_root = img_root
        self.cond_list = cond_list

    def __getitem__(self, index):

        path =  self.cond_list[index]['path']
        class_cond = self.cond_list[index]['class']
        shift_cond = self.cond_list[index]['shift']

        img = cv.imread(os.path.join(self.img_root, path), cv.IMREAD_GRAYSCALE)
        if self.transform is not None:
            transformed = self.transform(image=img)
            img = transformed['image']
        img = transforms.ToTensor()(img)

        class_cond = torch.tensor(class_cond).long()
        shift_cond = torch.tensor(shift_cond).float()

        return {'layer_cond': img, 'path':path, 'class_cond':class_cond, 'shift_cond':shift_cond}

    def __len__(self):
        return len(self.cond_list)
