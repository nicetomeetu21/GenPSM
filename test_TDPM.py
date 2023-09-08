# -*- coding:utf-8 -*-
import os
import cv2 as cv
from argparse import ArgumentParser
import pytorch_lightning as pl

from ldm.models.diffusion.ddim import DDIMSampler

from base.init_experiment import initExperiment
from train_TDPM import DDPM
from utils.layercode2mask import layer_code2img
from datamodule.layergen_test_datamodule import test_layergen_cond_Datamodule

# python test_TDPM.py --result_save_dir /home/Data/huangkun/layerseg/test/label --ckpt_path /home/Data/huangkun/layerseg/6mm/diffusion_results/ours/ddpm_vit_layercode_classShiftcond_total_LS_v4_lora_contrast_0.01_new_4_2023-07-26T14-46-33/lightning_logs/version_0/checkpoints/model-epoch=49_merged.ckpt  --test_json_path jsons/generated_94500.json
def get_parser():
    parser = ArgumentParser()
    parser.add_argument("--result_save_dir", type=str, default=r'') # path/to/generated/label
    parser.add_argument('--ckpt_path', type=str, default=r'') # path/to/TDPM/checkpoint
    parser.add_argument('--test_json_path', type=str, default='jsons/generated_94500.json')
    parser.add_argument('--use_ddim', type=bool, default=False)
    parser.add_argument('--ddim_steps', type=int, default=500)
    # lightning args
    parser.add_argument("--command", default="test")
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=32)
    parser.add_argument('--accelerator', default='gpu')
    parser.add_argument('--devices', default=[0])
    parser.add_argument('--reproduce', default=False)
    parser.add_argument('--logger', default=False)
    return parser


def main(opts):
    datamodule = test_layergen_cond_Datamodule(cond_json_path=opts.test_json_path, batch_size=opts.batch_size,
                                               num_workers=opts.num_workers)
    # exit()
    model = DDPM_test(opts)
    trainer = pl.Trainer.from_argparse_args(opts)
    trainer.test(model=model, datamodule=datamodule)


class DDPM_test(DDPM):
    def __init__(self, opts):
        super().__init__(opts)
        self.init_from_ckpt(opts.ckpt_path)
        os.makedirs(opts.result_save_dir, exist_ok=True)

    def test_step(self, batch, batch_idx):
        pathes = batch['path']
        class_cond = batch['class_cond']
        shift_cond = batch['shift_cond']

        cond = [class_cond, shift_cond]
        if os.path.exists(os.path.join(self.opts.result_save_dir, pathes[-1])): return
        if self.opts.use_ddim:
            samples = self.ddim_sample(cond=cond, batch_size=len(pathes))
        else:
            samples = self.sample(cond=cond, batch_size=len(pathes), return_intermediates=False, clip_denoised=True)
        samples = samples * 0.5 + 0.5
        samples = samples.to('cpu').numpy() * 640
        for i in range(len(pathes)):
            img = layer_code2img(samples[i, 0, :, :])
            if '/' in pathes[i]:
                names = pathes[i].split('/')
                save_dir = '/'.join([self.opts.result_save_dir] + names[:-1])
                os.makedirs(save_dir, exist_ok=True)
                print(save_dir)
            cv.imwrite(os.path.join(self.opts.result_save_dir, pathes[i]), img)

    def ddim_sample(self, cond, batch_size, return_intermediates=False):
        ddim_sampler = DDIMSampler(self)
        shape = [self.channels] + list(self.image_size)
        samples, intermediates = ddim_sampler.sample(self.opts.ddim_steps, batch_size,
                                                     shape, cond, verbose=False)
        if return_intermediates:
            return samples, intermediates
        else:
            return samples


if __name__ == '__main__':
    parser = get_parser()
    opts = parser.parse_args()
    initExperiment(opts)
    main(opts)
