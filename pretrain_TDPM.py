# -*- coding:utf-8 -*-
import os
from argparse import ArgumentParser
from tqdm import tqdm
import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar

from base.init_experiment import initExperiment_v2
from utils.util import instantiate_from_config

from ldm.modules.diffusionmodules.util import extract_into_tensor, noise_like
from ldm.modules.ema import LitEma
from ldm.util import default
from utils.util_for_opencv_diffusion import DDPM_base
from utils.layercode2mask import layer_code2img
from networks.modified_monai_vit_w_cond_LS_v4 import ViTAutoEnc
import cv2 as cv

def get_parser():
    parser = ArgumentParser()
    parser.add_argument("--exp_name", default='') # save name
    parser.add_argument('--result_root', type=str, default='') # path/to/results
    parser.add_argument('--data_config',default='configs/layergen/octa6mm.yaml') # path/to/config
    parser.add_argument("--command", default="fit")
    parser.add_argument('--devices', default=[0])
    parser.add_argument("--max_epochs", type=int, default=800)
    parser.add_argument("--limit_train_batches", type=int, default=10000)# 10000
    parser.add_argument("--base_learning_rate", type=float, default=4.5e-6)
    parser.add_argument('--accumulate_grad_batches', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=32)
    parser.add_argument('--scale_lr', type=bool, default=True)
    parser.add_argument('--profiler', default='simple')
    parser.add_argument('--accelerator', default='gpu')
    parser.add_argument('--reproduce', type=int, default=False)
    return parser


def main(opts, data_cfg):
    datamodule = instantiate_from_config(data_cfg)
    model = DDPM(opts)
    if opts.command == "fit":
        ckpt_callback = ModelCheckpoint(save_last=True, filename="model-{epoch}")
        trainer = pl.Trainer.from_argparse_args(opts, callbacks=[ckpt_callback, TQDMProgressBar(refresh_rate=50)])
        trainer.fit(model=model, datamodule=datamodule)


class DDPM(DDPM_base):
    def __init__(self, opts):
        super().__init__()
        self.opts = opts
        self.save_hyperparameters()

        self.model = ViTAutoEnc(in_channels=1, patch_size=(8, 1), img_size=(8, 400), spatial_dims=2)

        self.image_size = (8, 400)
        self.channels = 1

        self.parameterization = "eps"  # all assuming fixed variance schedules
        self.loss_type = "l2"
        self.use_ema = True
        self.use_positional_encodings = False
        self.v_posterior = 0.  # weight for choosing posterior variance as sigma = (1-v) * beta_tilde + v * beta
        self.original_elbo_weight = 0.
        self.l_simple_weight = 1.
        self.register_schedule()

        # self.local_len = 32

        if self.use_ema:
            self.model_ema = LitEma(self.model)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)

    def p_losses(self, x_start, t, cond):

        local_len = torch.randint(32, 128 + 1, (1,), device=x_start.device).item()
        start_pos = torch.randint(0, 400 - local_len + 1, (1,), device=x_start.device)
        start_pos2 = torch.randint(0, 400 - local_len + 1, (1,), device=x_start.device)
        x_start_local = x_start[:,:,:, start_pos:start_pos + local_len]

        x_ma = torch.amax(x_start_local[:,:,1:-1,:], dim=(1,2,3))
        x_mi = torch.amin(x_start_local[:,:,1:-1,:], dim=(1,2,3))
        shift_mi = (-1 -x_mi)
        shift_ma = 1-x_ma
        rand_height_shift = torch.rand(1, device=self.device) * (shift_ma-shift_mi) +shift_mi
        rand_height_shift = rand_height_shift[:,None,None, None]
        x_start_local[:,:,1:-1,:] += rand_height_shift*0.5

        c = torch.randint(0, 2, (1,), device=self.device)
        if c > 0:
            x_start_local = torch.flip(x_start_local, dims=[3])

        noise_local = torch.randn_like(x_start_local)
        x_noisy_local = self.q_sample(x_start=x_start_local, t=t, noise=noise_local)

        model_out_local = self.apply_model_local(x_noisy_local, t, cond, start_pos2)

        if self.parameterization == "eps":
            # target = noise
            target_local = noise_local
        elif self.parameterization == "x0":
            # target = x_start
            target_local = x_start_local
        else:
            raise NotImplementedError(f"Paramterization {self.parameterization} not yet supported")

        # loss = self.get_loss(model_out, target, mean=False).mean(dim=[1, 2, 3])
        loss_local = self.get_loss(model_out_local, target_local, mean=False).mean(dim=[1, 2, 3])

        log_prefix = 'train' if self.training else 'val'
        # loss_simple = loss.mean() * self.l_simple_weight
        # loss_vlb = (self.lvlb_weights[t] * loss).mean()

        loss = loss_local.mean()
        loss_dict = {}
        loss_dict.update({f'{log_prefix}/loss_simple': loss.mean()})
        # loss_dict.update({f'{log_prefix}/loss_vlb': loss_vlb})
        loss_dict.update({f'{log_prefix}/loss': loss})
        return loss, loss_dict

    def forward(self, x, cond):
        # b, c, h, w, device, img_size, = *x.shape, x.device, self.image_size
        # assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        return self.p_losses(x, t, cond)

    def apply_model(self, x, t, cond):
        class_cond, shift_cond = cond
        return self.model(x=x, timesteps=t, class_cond=class_cond, shift_cond=shift_cond)

    def apply_model_local(self, x, t, cond, pos_start):
        class_cond, shift_cond = cond
        return self.model.local_forward(x=x, timesteps=t, class_cond=class_cond, shift_cond=shift_cond, pos_start=pos_start)

    def training_step(self, batch, batch_idx):
        x = batch['image']
        class_cond = batch['class_cond']
        shift_cond = batch['shift_cond']
        # print(x.shape, class_cond.shape, shift_cond.shape)
        loss, loss_dict = self(x, [class_cond, shift_cond])

        self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        if batch_idx == 0:
            self.batch_sample = batch
        return loss

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self.model)

    def on_train_epoch_end(self):
        # print(-1)
        with self.ema_scope("Plotting"):
            img_save_dir = os.path.join(self.opts.default_root_dir, 'val_results', str(self.current_epoch))
            os.makedirs(img_save_dir, exist_ok=True)
            x = self.batch_sample['image'][:8]
            class_cond = self.batch_sample['class_cond'][:8]
            shift_cond = self.batch_sample['shift_cond'][:8]
            samples = self.sample(cond=[class_cond, shift_cond], batch_size=x.shape[0],
                                               return_intermediates=False, clip_denoised=True)
            # decoding
            x = x * 0.5 + 0.5
            samples = samples * 0.5 + 0.5
            x = x.to('cpu').numpy() * 640
            samples = samples.to('cpu').numpy() * 640
            for i in range(x.shape[0]):
                visual = []
                visual.append(layer_code2img(x[i, 0, :, :]))
                visual.append(layer_code2img(samples[i, 0, :, :]))
                visual = cv.hconcat(visual)
                save_name = str(i) + '_' + str(class_cond[i].item()) + '_' + str(shift_cond[i].item()) + '.png'
                # print(save_name)
                cv.imwrite(os.path.join(img_save_dir, save_name), visual)

            # val local
            img_save_dir = os.path.join(self.opts.default_root_dir, 'val_results_local', str(self.current_epoch))
            os.makedirs(img_save_dir, exist_ok=True)
            x = self.batch_sample['image']
            class_cond = self.batch_sample['class_cond']
            shift_cond = self.batch_sample['shift_cond']
            local_len = torch.randint(32, 128 + 1, (1,), device=x.device).item()
            start_pos2 = torch.randint(0, 400 - local_len + 1, (1,), device=x.device)
            # print(1)
            samples = self.sample_local(cond=[class_cond, shift_cond], local_len=local_len, pos_start=start_pos2, batch_size=x.shape[0],
                                               return_intermediates=False, clip_denoised=True)
            # print(2)
            x = x * 0.5 + 0.5
            samples = samples * 0.5 + 0.5
            x = x.to('cpu').numpy() * 640
            samples = samples.to('cpu').numpy() * 640
            for i in range(x.shape[0]):
                visual = []
                visual.append(layer_code2img(x[i, 0, :, :]))
                ret = np.zeros_like(x[i, 0, :, :])
                ret[:, start_pos2:start_pos2+local_len] = samples[i, 0, :, :]
                visual.append(layer_code2img(ret))
                visual = cv.hconcat(visual)
                save_name = str(i) + '_' + str(class_cond[i].item()) + '_' + str(shift_cond[i].item()) + '.png'
                # print(save_name)
                cv.imwrite(os.path.join(img_save_dir, save_name), visual)

    def predict_start_from_noise(self, x_t, t, noise):
        return (
                extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, cond, clip_denoised):
        model_out = self.apply_model(x, t, cond)
        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
        elif self.parameterization == "x0":
            x_recon = model_out
        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, cond, clip_denoised, repeat_noise=False):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, cond=cond, clip_denoised=clip_denoised)
        # no noise when t == 0
        noise = noise_like(x.shape, device, repeat_noise)
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, cond, return_intermediates=False, log_every_t=100, clip_denoised=True):
        device = self.betas.device
        b = shape[0]
        img = torch.randn(shape, device=device)
        if return_intermediates:
            intermediates = [img]
        with tqdm(reversed(range(0, self.num_timesteps)), desc='Sampling t', total=self.num_timesteps,
                  mininterval=5) as pbar:
            for i in pbar:
                img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long), cond,
                                    clip_denoised=clip_denoised)
                if return_intermediates and (i % log_every_t == 0 or i == self.num_timesteps - 1):
                    intermediates.append(img)
        if return_intermediates:
            return img, intermediates
        return img

    @torch.no_grad()
    def sample(self, cond, batch_size=1, return_intermediates=False, clip_denoised=True):
        return self.p_sample_loop([batch_size, self.channels] + list(self.image_size), cond,
                                  return_intermediates=return_intermediates, clip_denoised=clip_denoised)
    def p_mean_variance_local(self, x, t, cond, pos_start, clip_denoised):
        model_out = self.apply_model_local(x, t, cond, pos_start)
        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
        elif self.parameterization == "x0":
            x_recon = model_out
        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample_local(self, x, t, cond, pos_start, clip_denoised, repeat_noise=False):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance_local(x=x, t=t, cond=cond, pos_start=pos_start, clip_denoised=clip_denoised)
        # no noise when t == 0
        noise = noise_like(x.shape, device, repeat_noise)
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
    @torch.no_grad()
    def p_sample_loop_local(self, shape, cond, pos_start, return_intermediates=False, log_every_t=100, clip_denoised=True):
        device = self.betas.device
        b = shape[0]
        img = torch.randn(shape, device=device)
        if return_intermediates:
            intermediates = [img]
        with tqdm(reversed(range(0, self.num_timesteps)), desc='Sampling t', total=self.num_timesteps,
                  mininterval=5) as pbar:
            for i in pbar:
                img = self.p_sample_local(img, torch.full((b,), i, device=device, dtype=torch.long), cond, pos_start,
                                    clip_denoised=clip_denoised)
                if return_intermediates and (i % log_every_t == 0 or i == self.num_timesteps - 1):
                    intermediates.append(img)
        if return_intermediates:
            return img, intermediates
        return img

    @torch.no_grad()
    def sample_local(self, cond, local_len, pos_start, batch_size=1, return_intermediates=False, clip_denoised=True):
        return self.p_sample_loop_local([batch_size, self.channels, self.image_size[0], local_len], cond, pos_start,
                                  return_intermediates=return_intermediates, clip_denoised=clip_denoised)

    def configure_optimizers(self):
        lr = self.opts.learning_rate
        params = list(self.model.parameters())
        opt = torch.optim.AdamW(params, lr=lr)
        return opt


if __name__ == '__main__':
    parser = get_parser()
    opts = parser.parse_args()
    data_cfg = OmegaConf.load(opts.data_config)
    initExperiment_v2(opts, data_cfg)
    main(opts, data_cfg)
