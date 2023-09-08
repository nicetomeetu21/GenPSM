# -*- coding:utf-8 -*-
import json
import os

import albumentations as A
import cv2 as cv
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import torch

class trainDatamodule(pl.LightningDataModule):
    def __init__(self, train_data_root, train_label_root, train_json_path, train_cond_json_path, image_size=(640,400), batch_size=1, num_workers=8,
                 **kwargs):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_data_root = train_data_root
        self.train_label_root = train_label_root
        with open(train_json_path, "r") as f:
            self.train_pathes = json.load(f)
        print('train_pathes ',len(self.train_pathes))
        with open(train_cond_json_path, "r") as f:
            self.cond_dict = json.load(f)
        self.train_transform = A.Compose([
            A.Resize(*image_size),
            A.Normalize((0.5,), (0.5,))
        ])

    def setup(self, stage=None):
        self.train_set = cond_Dataset(img_root=self.train_data_root, label_root=self.train_label_root,
                                         pathes=self.train_pathes, transform=self.train_transform, cond_dict=self.cond_dict)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)


class cond_Dataset(Dataset):
    def __init__(self, img_root, label_root, pathes, transform, cond_dict):
        assert os.path.exists(img_root)
        assert os.path.exists(label_root)

        self.transform = transform
        self.img_root = img_root
        self.label_root = label_root
        self.pathes = pathes
        self.cond_dict = cond_dict

    def __getitem__(self, index):
        # print(os.path.join(self.img_root, self.pathes[index]), os.path.join(self.label_root, self.pathes[index]))
        img = cv.imread(os.path.join(self.img_root, self.pathes[index]), cv.IMREAD_GRAYSCALE)
        layer_cond = cv.imread(os.path.join(self.label_root, self.pathes[index]), cv.IMREAD_GRAYSCALE)
        if self.transform is not None:
            transformed = self.transform(image=img, masks=[layer_cond])
            img = transformed['image']
            layer_cond = transformed['masks'][0]
        img = transforms.ToTensor()(img)
        layer_cond = transforms.ToTensor()(layer_cond)
        class_cond = self.cond_dict[self.pathes[index][:-4]]['class']
        shift_cond = self.cond_dict[self.pathes[index][:-4]]['mdice']
        class_cond = torch.tensor(class_cond).long()
        shift_cond = torch.tensor(shift_cond).float()
        return {'image': img, 'layer_cond': layer_cond, 'path': self.pathes[index], 'class_cond':class_cond, 'shift_cond':shift_cond}

    def __len__(self):
        return len(self.pathes)
