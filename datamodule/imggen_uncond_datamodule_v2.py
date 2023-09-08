# -*- coding:utf-8 -*-
import json
import os

import albumentations as A
import cv2 as cv
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import transforms


class trainDatamodule(pl.LightningDataModule):
    def __init__(self, train_data_root, train_json_path, image_size, crop_size, batch_size=1, num_workers=8,
                 **kwargs):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_data_root = train_data_root
        with open(train_json_path, "r") as f:
            self.train_pathes = json.load(f)
        print(len(self.train_pathes))
        self.train_transform = A.Compose([
            A.Resize(*image_size),
            A.RandomCrop(*crop_size),
            A.Normalize((0.5,), (0.5,))
        ])

    def setup(self, stage=None):
        self.train_set = uncond_Dataset(img_root=self.train_data_root, pathes=self.train_pathes,
                                        transform=self.train_transform)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)


class uncond_Dataset(Dataset):
    def __init__(self, img_root, pathes, transform):
        assert os.path.exists(img_root)

        self.img_root = img_root
        self.pathes = pathes
        self.transform = transform

    def __getitem__(self, index):
        img = cv.imread(os.path.join(self.img_root, self.pathes[index]), cv.IMREAD_GRAYSCALE)

        if self.transform is not None:
            transformed = self.transform(image=img)
            img = transformed['image']

        img = transforms.ToTensor()(img)

        return {'image': img, 'path': self.pathes[index]}

    def __len__(self):
        return len(self.pathes)
