# -*- coding:utf-8 -*-
import os
from argparse import ArgumentParser
from tqdm import tqdm
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar

from base.init_experiment import initExperiment_v2
from utils.util import instantiate_from_config

from ldm.modules.diffusionmodules.util import extract_into_tensor, noise_like
from ldm.modules.ema import LitEma
from ldm.util import default
from utils.util_for_opencv_diffusion import DDPM_base
from utils.layercode2mask import layer_code2img
from networks.modified_monai_vit_w_cond_LS_v4_lora_contrast import ViTAutoEnc, MLPBlock
import cv2 as cv
import loralib as lora
from utils.util import set_requires_grad
import torch.nn.functional as F

# python finetune_TDPM.py --exp_name TDPM_1_6mm --result_root /home/Data/huangkun/layerseg/test  --data_config configs/layergen/octa6mm.yaml --first_stage_ckpt /home/Data/huangkun/layerseg/6mm/diffusion_results/totals/layergen/ddpm_vit_layercode_classShiftcond_total_LS_v4_pretrained_2_2023-07-12T21-33-53/lightning_logs/version_0/checkpoints/last.ckpt
def get_parser():
    parser = ArgumentParser()
    parser.add_argument("--exp_name", default='') # save name
    parser.add_argument('--result_root', type=str, default='') # path/to/results
    parser.add_argument('--data_config', default='configs/layergen/octa6mm.yaml')  # path/to/config
    parser.add_argument('--first_stage_ckpt', default='')  # path/to/pretrained/checkpoint
    parser.add_argument("--command", default="fit")
    parser.add_argument('--devices', default=[0])
    parser.add_argument("--max_epochs", type=int, default=400)
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
        ckpt_callback = ModelCheckpoint(save_last=False, save_top_k=-1, every_n_epochs=50, filename="model-{epoch}")
        trainer = pl.Trainer.from_argparse_args(opts, callbacks=[ckpt_callback, TQDMProgressBar(refresh_rate=50)])
        ckpt_path = opts.first_stage_ckpt

        model.init_from_ckpt(ckpt_path)
        lora.mark_only_lora_as_trainable(model)
        set_requires_grad(model.model.class_embed, True)
        set_requires_grad(model.model.patch_embedding, True)
        set_requires_grad(model.model.shift_embed, True)
        set_requires_grad(model.model.time_embed, True)
        set_requires_grad(model.model.conv3d_transpose, True)
        set_requires_grad(model.model.conv3d_transpose_1, True)
        trainer.fit(model=model, datamodule=datamodule)

def device_as(t1, t2):
   """
   Moves t1 to the device of t2
   """
   return t1.to(t2.device)

class ContrastiveLoss(nn.Module):
   """
   Vanilla Contrastive loss, also called InfoNceLoss as in SimCLR paper
   """
   def __init__(self, batch_size, temperature=0.5):
       super().__init__()
       self.batch_size = batch_size
       self.temperature = temperature
       self.mask = (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool)).float()
       self.projection = MLPBlock(hidden_size=51200, mlp_dim=512, dropout_rate=0.0)

   def calc_similarity_batch(self, a, b):
       representations = torch.cat([a, b], dim=0)
       return F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)

   def forward(self, proj_1, proj_2):
       """
       proj_1 and proj_2 are batched embeddings [batch, embedding_dim]
       where corresponding indices are pairs
       z_i, z_j in the SimCLR paper
       """
       # print(proj_1.shape)
       proj_1 = proj_1.reshape(proj_1.shape[0],-1)
       proj_2 = proj_2.reshape(proj_2.shape[0],-1)
       # print(proj_1.shape)
       proj_1 = self.projection(proj_1)
       proj_2 = self.projection(proj_2)
       batch_size = proj_1.shape[0]
       z_i = F.normalize(proj_1, p=2, dim=1)
       z_j = F.normalize(proj_2, p=2, dim=1)

       similarity_matrix = self.calc_similarity_batch(z_i, z_j)

       sim_ij = torch.diag(similarity_matrix, batch_size)
       sim_ji = torch.diag(similarity_matrix, -batch_size)

       positives = torch.cat([sim_ij, sim_ji], dim=0)

       nominator = torch.exp(positives / self.temperature)

       denominator = device_as(self.mask, similarity_matrix) * torch.exp(similarity_matrix / self.temperature)

       all_losses = -torch.log(nominator / torch.sum(denominator, dim=1))
       loss = torch.sum(all_losses) / (2 * self.batch_size)
       return loss

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
        self.use_ema = False
        self.use_positional_encodings = False
        self.v_posterior = 0.  # weight for choosing posterior variance as sigma = (1-v) * beta_tilde + v * beta
        self.original_elbo_weight = 0.
        self.l_simple_weight = 1.
        self.register_schedule()

        self.batch_size = opts.batch_size
        self.simclr_loss = ContrastiveLoss(self.batch_size)

        # self.local_len = 32

        # if self.use_ema:
        #     self.model_ema = LitEma(self.model)
        #     print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)

    def p_losses(self, x_start, t, cond):
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_out, feat_out,_ = self.apply_model_constrast(x_noisy, t, cond)
        model_out = model_out[:self.batch_size]

        loss_contrast = self.simclr_loss(feat_out[:self.batch_size].detach(), feat_out[self.batch_size:])

        if self.parameterization == "eps":
            target = noise
        elif self.parameterization == "x0":
            target = x_start
        else:
            raise NotImplementedError(f"Paramterization {self.parameterization} not yet supported")

        loss_simple = self.get_loss(model_out, target, mean=False).mean(dim=[1, 2, 3])

        log_prefix = 'train' if self.training else 'val'

        loss = loss_simple.mean() + loss_contrast.mean()*0.01
        loss_dict = {}
        loss_dict.update({f'{log_prefix}/loss_simple': loss_simple.mean()})
        loss_dict.update({f'{log_prefix}/loss_contrast': loss_contrast.mean()})
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

    def apply_model_constrast(self, x, t, cond):
        class_cond, shift_cond = cond
        rand_class_cond = torch.randint(1, 7, (x.shape[0],), device=self.device).long()
        rand_shift_cond = shift_cond-shift_cond*torch.rand((x.shape[0],), device=self.device)
        x_in = torch.cat([x,x], dim=0)
        t_in = torch.cat([t, t], dim=0)
        class_cond_in = torch.cat([class_cond, rand_class_cond], dim=0)
        shift_cond_in = torch.cat([shift_cond, rand_shift_cond], dim=0)
        return self.model(x=x_in, timesteps=t_in, class_cond=class_cond_in, shift_cond=shift_cond_in, out_feat=True)

    def training_step(self, batch, batch_idx):
        x = batch['image']
        class_cond = batch['class_cond']
        shift_cond = batch['shift_cond']
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

    def configure_optimizers(self):
        lr = self.opts.learning_rate
        params = list(self.model.parameters())+list(self.simclr_loss.projection.parameters())
        opt = torch.optim.AdamW(params, lr=lr)
        return opt


if __name__ == '__main__':
    parser = get_parser()
    opts = parser.parse_args()
    data_cfg = OmegaConf.load(opts.data_config)
    initExperiment_v2(opts, data_cfg)
    main(opts, data_cfg)
