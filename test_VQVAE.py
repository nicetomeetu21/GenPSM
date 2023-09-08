# -*- coding:utf-8 -*-
import os
from argparse import ArgumentParser

import torch
import pytorch_lightning as pl
from torchvision.utils import save_image

from datamodule.layerseg_single_datamodule import testJsonDatamodule
from train_VQVAE import VQModel


def get_parser():
    parser = ArgumentParser()
    parser.add_argument("--result_save_dir", type=str,default=r'')
    parser.add_argument('--ckpt_path', type=str,default=r'')
    parser.add_argument('--test_img_root', type=str,default=r'')
    parser.add_argument('--test_json_path', type=str, default=r'jsons/octa6mm/test_pathes.json')
    parser.add_argument("--image_size", default=(640, 512))
    # lightning args
    parser.add_argument("--command", default="test")
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--accelerator', default='gpu')
    parser.add_argument('--devices', default=[0])
    parser.add_argument('--reproduce', type=int, default=False)
    parser.add_argument('--logger', type=int, default=False)
    return parser


def main(opts):
    datamodule = testJsonDatamodule(test_img_root=opts.test_img_root, test_json_path=opts.test_json_path,
                                    image_size=opts.image_size, batch_size=opts.batch_size,
                                    num_workers=opts.num_workers)
    model = VQModel_test(opts)
    trainer = pl.Trainer.from_argparse_args(opts)
    trainer.test(model=model, datamodule=datamodule)

class VQModel_test(VQModel):
    def __init__(self, opts):
        super().__init__(opts)
        self.opts = opts
        self.init_from_ckpt(opts.ckpt_path)

    def test_step(self, batch, batch_idx):
        x = batch['image']
        pathes = batch['path']
        if os.path.exists(os.path.join(self.opts.result_save_dir, pathes[-1])): return
        xrec, _, _ = self(x, return_pred_indices=True)
        x = x*0.5+0.5
        xrec = xrec*0.5+0.5
        for i, path in enumerate(pathes):
            if '/' in pathes[i]:
                names = pathes[i].split('/')
                save_dir = '/'.join([self.opts.result_save_dir] + names[:-1])
                os.makedirs(save_dir, exist_ok=True)
            save_image(torch.cat([x, xrec], dim=0), os.path.join(self.opts.result_save_dir, pathes[i]))

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location=self.device)["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
            print(f"Unexpected Keys: {unexpected}")



if __name__ == '__main__':
    parser = get_parser()
    opts = parser.parse_args()
    main(opts)

