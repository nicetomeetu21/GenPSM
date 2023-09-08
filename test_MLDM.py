# -*- coding:utf-8 -*-
import os
from argparse import ArgumentParser
import pytorch_lightning as pl
from torchvision.utils import save_image

from ldm.models.diffusion.ddim import DDIMSampler
from base.init_experiment import initExperiment
from train_MLDM import LDM
from datamodule.layer2img_test_datamodule_v2 import testJsonDatamodule

def get_parser():
    parser = ArgumentParser()
    parser.add_argument("--result_save_dir", type=str,default='') # path/to/generated/image
    parser.add_argument('--ckpt_path', type=str,default='') # path/to/MLDM/checkpoint
    parser.add_argument('--test_img_root', type=str, default='') # path/to/label
    parser.add_argument('--test_json_path', type=str, default='') # path/to/generated/json
    parser.add_argument('--first_stage_ckpt', type=str, default='') # path/to/VQVAE/checkpoint
    parser.add_argument('--use_ddim', type=bool, default=True)
    parser.add_argument('--ddim_steps', type=int, default=200)
    parser.add_argument("--image_size", default=(640, 512))
    parser.add_argument("--latent_size", default=(80, 64))
    parser.add_argument("--latent_channel", default=4)
    # lightning args
    parser.add_argument("--command", default="test")
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--accelerator', default='gpu')
    parser.add_argument('--devices', default=[0])
    parser.add_argument('--reproduce', type=int, default=False)
    parser.add_argument('--logger', type=int, default=False)
    return parser


def main(opts):
    model = LDM_test(opts)
    trainer = pl.Trainer.from_argparse_args(opts)
    datamodule = testJsonDatamodule(test_img_root=opts.test_img_root, cond_json_path=opts.test_json_path,
                                    image_size=opts.image_size, batch_size=opts.batch_size,
                                    num_workers=opts.num_workers)
    trainer.test(model=model, datamodule=datamodule)


class LDM_test(LDM):
    def __init__(self, opts):
        super().__init__(opts)
        self.init_from_ckpt(opts.ckpt_path)
        os.makedirs(opts.result_save_dir, exist_ok=True)

        self.model_ema.copy_to(self.model)
        print("Switched to EMA weights")

    def test_step(self, batch, batch_idx):
        pathes = batch['path']
        if os.path.exists(os.path.join(self.opts.result_save_dir, pathes[-1])): return

        layer_cond = batch['layer_cond']
        class_cond = batch['class_cond']
        shift_cond = batch['shift_cond']
        c = [layer_cond, class_cond, shift_cond]

        if self.opts.use_ddim:
            samples = self.ddim_sample(cond=c, batch_size=len(pathes))
        else:
            samples = self.sample(c=c, batch_size=len(pathes), return_intermediates=False, clip_denoised=True)
        samples = self.decode_first_stage(samples).cpu()
        samples = samples * 0.5 + 0.5
        for i in range(len(pathes)):
            if '/' in pathes[i]:
                names = pathes[i].split('/')
                save_dir = '/'.join([self.opts.result_save_dir] + names[:-1])
                os.makedirs(save_dir, exist_ok=True)
                # print(save_dir)
            save_image(samples[i:i + 1], os.path.join(self.opts.result_save_dir, pathes[i]))

    def ddim_sample(self, cond, batch_size, return_intermediates=False):
        ddim_sampler = DDIMSampler(self)
        shape = [self.channels]  + list(self.latent_size)
        out = ddim_sampler.sample(self.opts.ddim_steps, batch_size,
                                                     shape, cond, verbose=False, return_intermediates=return_intermediates)
        return out

if __name__ == '__main__':
    parser = get_parser()
    opts = parser.parse_args()
    initExperiment(opts)
    main(opts)
