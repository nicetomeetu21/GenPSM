# -*- coding:utf-8 -*-
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import json


class test_layergen_cond_Datamodule(pl.LightningDataModule):
    def __init__(self, cond_json_path, selected_degree=[0,1,2,3,4,5,6,7,8,9], batch_size=1, num_workers=8, **kwargs):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        with open(cond_json_path, "r") as f:
            self.cond_list = json.load(f)
        print(len(self.cond_list))
        print(self.cond_list[:5])
        self.select_list = []
        for item in self.cond_list:
            path = item['path']
            _,degree,_ = path.split('/')
            degree = float(degree)
            degree *= 10
            degree = int(degree)
            # print(degree)
            if degree in selected_degree:
                self.select_list.append(item)

        print('len(self.select_list)', len(self.select_list))

    def setup(self, stage=None):
        self.test_set = test_cond_Dataset(cond_list=self.select_list)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)


class test_cond_Dataset(Dataset):
    def __init__(self, cond_list):
        self.cond_list = cond_list

    def __getitem__(self, index):
        path =  self.cond_list[index]['path']
        class_cond = self.cond_list[index]['class']
        shift_cond = self.cond_list[index]['shift']

        class_cond = torch.tensor(class_cond).long()
        shift_cond = torch.tensor(shift_cond).float()

        return {'path':path, 'class_cond':class_cond, 'shift_cond':shift_cond}

    def __len__(self):
        return len(self.cond_list)