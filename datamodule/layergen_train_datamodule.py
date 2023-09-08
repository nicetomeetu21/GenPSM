# -*- coding:utf-8 -*-
import json
import os

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset


class trainDatamodule(pl.LightningDataModule):
    def __init__(self, train_data_root, train_json_path, train_cond_json_path, batch_size=1, num_workers=8,
                 **kwargs):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_data_root = train_data_root
        with open(train_json_path, "r") as f:
            self.train_pathes = json.load(f)

        with open(train_cond_json_path, "r") as f:
            self.cond_dict = json.load(f)
        print(len(self.train_pathes))


    def setup(self, stage=None):
        self.train_set = cond_Dataset(img_root=self.train_data_root, pathes=self.train_pathes, cond_dict=self.cond_dict)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)


class cond_Dataset(Dataset):
    def __init__(self, img_root, pathes, cond_dict):
        assert os.path.exists(img_root)

        self.img_root = img_root
        self.pathes = pathes
        self.cond_dict = cond_dict

    def __getitem__(self, index):
        img = np.load(os.path.join(self.img_root, self.pathes[index][:-4]+'.npy'))
        img = torch.from_numpy(img).float()
        img = img.unsqueeze(0)
        img /= 640
        img -= 0.5
        img /= 0.5

        class_cond = self.cond_dict[self.pathes[index][:-4]]['class']
        shift_cond = self.cond_dict[self.pathes[index][:-4]]['mdice']
        class_cond = torch.tensor(class_cond).long()
        shift_cond = torch.tensor(shift_cond).float()

        return {'image': img, 'path': self.pathes[index], 'class_cond':class_cond, 'shift_cond':shift_cond}

    def __len__(self):
        return len(self.pathes)
