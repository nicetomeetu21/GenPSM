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

class spDatamodule(pl.LightningDataModule):
    def __init__(self, train_img1_root, train_label1_root, train_json1_path, train_img2_root, train_label2_root, train_json2_path, test_img_root, test_label_root, test_json_path,
                 image_size, crop_size, batch_size=1, num_workers=8, **kwargs):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_img1_root = train_img1_root
        self.train_label1_root = train_label1_root
        self.train_img2_root = train_img2_root
        self.train_label2_root = train_label2_root
        self.test_img_root = test_img_root
        self.test_label_root = test_label_root
        with open(train_json1_path, "r") as f:
            self.train_pathes1 = json.load(f)
        with open(train_json2_path, "r") as f:
            self.train_pathes2 = json.load(f)
        with open(test_json_path, "r") as f:
            self.test_pathes = json.load(f)
        print(len(self.train_pathes1),len(self.train_pathes2), len(self.test_pathes))
        self.train_transform = A.Compose([
            A.Resize(*image_size),
            A.RandomCrop(*crop_size),
            A.HorizontalFlip(),
            A.Normalize((0.5,), (0.5,))
        ])
        self.test_transform = A.Compose([
            A.Resize(*image_size),
            A.Normalize((0.5,), (0.5,))
        ])

    def setup(self, stage=None):
        train_set1 = labeled_Dataset(img_root=self.train_img1_root, label_root=self.train_label1_root,
                                     pathes=self.train_pathes1,
                                     transform=self.train_transform)
        train_set2 = labeled_Dataset(img_root=self.train_img2_root, label_root=self.train_label2_root,
                                     pathes=self.train_pathes2,
                                     transform=self.train_transform)
        self.train_set = ConcatDataset([train_set1, train_set2], ['real', 'fake'])
        self.test_set = labeled_Dataset(img_root=self.test_img_root, label_root=self.test_label_root,
                                        pathes=self.test_pathes,
                                        transform=self.test_transform)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.test_set, batch_size=1, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=1, shuffle=False, num_workers=self.num_workers)


class testJsonDatamodule(pl.LightningDataModule):
    def __init__(self, test_img_root, test_json_path=None, test_label_root=None, image_size=(640, 400), batch_size=1,
                 num_workers=8, **kwargs):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.test_img_root = test_img_root
        self.test_label_root = test_label_root
        self.test_transform = A.Compose([
            A.Resize(*image_size),
            A.Normalize((0.5,), (0.5,))
        ])

        if test_json_path is None:
            self.test_pathes = natsorted(os.listdir(test_img_root))
        else:
            with open(test_json_path, "r") as f:
                self.test_pathes = json.load(f)
        print(self.test_pathes)
    def setup(self, stage=None):
        if self.test_label_root is None:
            self.test_set = unlabeled_Dataset(img_root=self.test_img_root, pathes=self.test_pathes,
                                              transform=self.test_transform)
        else:
            self.test_set = labeled_Dataset(img_root=self.test_img_root, label_root=self.test_label_root,
                                            pathes=self.test_pathes, transform=self.test_transform)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)


class labeled_Dataset(Dataset):
    def __init__(self, img_root, label_root, pathes, transform):
        print(img_root, label_root)
        assert os.path.exists(img_root)
        assert os.path.exists(label_root)

        self.transform = transform
        self.img_root = img_root
        self.label_root = label_root
        self.pathes = pathes

    def __getitem__(self, index):
        img = cv.imread(os.path.join(self.img_root, self.pathes[index]), cv.IMREAD_GRAYSCALE)
        label = cv.imread(os.path.join(self.label_root, self.pathes[index]), cv.IMREAD_GRAYSCALE)
        # print(os.path.join(self.img_root, self.pathes[index]), os.path.join(self.label_root, self.pathes[index]))
        if self.transform is not None:
            transformed = self.transform(image=img, masks=[label])
            img = transformed['image']
            label = transformed['masks'][0]
        img = transforms.ToTensor()(img)
        label = torch.from_numpy(label).long()
        label = label.unsqueeze(0)

        return {'image': img, 'label': label, 'path': self.pathes[index]}

    def __len__(self):
        return len(self.pathes)


class unlabeled_Dataset(Dataset):
    def __init__(self, img_root, pathes, transform):
        assert os.path.exists(img_root)

        self.transform = transform
        self.img_root = img_root
        self.pathes = pathes

    def __getitem__(self, index):
        # print(os.path.join(self.img_root, self.pathes[index]))
        img = cv.imread(os.path.join(self.img_root, self.pathes[index]), cv.IMREAD_GRAYSCALE)
        if self.transform is not None:
            transformed = self.transform(image=img)
            img = transformed['image']
        img = transforms.ToTensor()(img)

        return {'image': img, 'path': self.pathes[index]}

    def __len__(self):
        return len(self.pathes)


class ConcatDataset(Dataset):
    def __init__(self, datasets, dataset_names, align=False):
        self.datasets = datasets
        self.transforms = transforms
        self.align = align
        self.dataset_names = dataset_names

    def __getitem__(self, index):
        if self.align:
            data = [d[index] for d in self.datasets]
        else:
            data = {}
            data[self.dataset_names[0]] = self.datasets[0][index]
            for i, d in enumerate(self.datasets):
                if i == 0: continue
                id = torch.randint(0, len(d), (1,))
                data[self.dataset_names[i]] = d[id]
        return data

    def __len__(self):
        return len(self.datasets[0])
