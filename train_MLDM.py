# -*- coding:utf-8 -*-
import os
from argparse import ArgumentParser
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.utilities.distributed import rank_zero_only
from torch.optim.lr_scheduler import LambdaLR
from omegaconf import OmegaConf
from tqdm import tqdm
from torchvision.utils import save_image

from base.init_experiment import initExperiment_v2
from utils.util_for_opencv_diffusion import DDPM_base, disabled_train
from ldm.lr_scheduler import LambdaLinearScheduler
from ldm.modules.diffusionmodules.util import extract_into_tensor, noise_like
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution
from utils.util import instantiate_from_config
from ldm.modules.ema import LitEma
from ldm.util import default
from train_VQVAE import VQModel
from networks.openaimodel_layerClassShiftCond_ldm_v2 import UNetModel

def get_parser():
    parser = ArgumentParser()
    parser.add_argument("--exp_name", default='') # save name
    parser.add_argument('--result_root', type=str, default=r'') # path/to/results
    parser.add_argument('--data_config',default=r'') # path/to/config
    parser.add_argument('--first_stage_ckpt', type=str, default='') # path/to/VQVAE/checkpoint
    parser.add_argument("--image_size", default=(640, 512))
    parser.add_argument("--latent_size", default=(80, 64))
    parser.add_argument("--latent_channel", default=4)
    parser.add_argument("--command", default="fit")
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--limit_train_batches", type=int, default=2000)  # 2000
    parser.add_argument("--base_learning_rate", type=float, default=4.5e-6)
    parser.add_argument('--accumulate_grad_batches', type=int, default=4)
    parser.add_argument('--scale_lr', type=bool, default=True)
    parser.add_argument('--profiler', default='simple')
    parser.add_argument('--accelerator', default='gpu')
    parser.add_argument('--devices', default=[0])
    parser.add_argument('--reproduce', type=int, default=False)
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--num_workers', type=int, default=32)
    return parser


def main(opts, data_cfg):
    for k, v in opts.__dict__.items():print(f"{k}: {v}")
    datamodule = instantiate_from_config(data_cfg)
    model = LDM(opts)
    if opts.command == "fit":
        ckpt_callback = ModelCheckpoint(save_last=True, filename="model-{epoch}")
        trainer = pl.Trainer.from_argparse_args(opts, callbacks=[ckpt_callback, TQDMProgressBar(refresh_rate=5)])
        trainer.fit(model=model, datamodule=datamodule)


class VQModelInterface(VQModel):
    def __init__(self, opts, *args, **kwargs):
        super().__init__(opts)

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        return h

    def decode(self, h, force_not_quantize=False):
        # also go through quantization layer
        if not force_not_quantize:
            quant, emb_loss, info = self.quantize(h)
        else:
            quant = h
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec


class LDM(DDPM_base):
    def __init__(self, opts):
        super().__init__()
        self.opts = opts
        self.save_hyperparameters()

        ddconfig = {'double_z': False, 'z_channels': 8, 'resolution': 512, 'in_channels': 1, 'out_ch': 1, 'ch': 32,
                    'ch_mult': [1, 2, 4, 4], 'num_res_blocks': 2, 'attn_resolutions': [], 'dropout': 0.0}
        unet_config = {'image_size': opts.latent_size, 'in_channels': opts.latent_channel+ddconfig['z_channels'],
                       'out_channels': opts.latent_channel, 'model_channels': 192,
                       'attention_resolutions': [2, 4, 8], 'num_res_blocks': 2, 'channel_mult': [1, 2, 4, 4],
                       'num_heads': 8, 'use_scale_shift_norm': True, 'resblock_updown': True, 'num_classes': 7, 'ddconfig':ddconfig}
        self.instantiate_first_stage(opts)
        self.model = UNetModel(**unet_config)

        self.latent_size = opts.latent_size
        self.channels = opts.latent_channel

        self.parameterization = "eps"  # all assuming fixed variance schedules
        self.loss_type = "l1"
        self.use_ema = True
        self.use_positional_encodings = False
        self.v_posterior = 0.  # weight for choosing posterior variance as sigma = (1-v) * beta_tilde + v * beta
        self.original_elbo_weight = 0.
        self.l_simple_weight = 1.
        self.scale_by_std = False
        self.log_every_t = 100

        scale_factor = 1.0
        if not self.scale_by_std:
            self.scale_factor = scale_factor
        else:
            self.register_buffer('scale_factor', torch.tensor(scale_factor))

        self.register_schedule()
        if self.use_ema:
            self.model_ema = LitEma(self.model)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

    def instantiate_first_stage(self, opts):
        print(opts.first_stage_ckpt)
        model = VQModelInterface.load_from_checkpoint(opts.first_stage_ckpt)
        states = torch.load(opts.first_stage_ckpt, map_location=self.device)
        model.load_state_dict(states['state_dict'])
        self.first_stage_model = model.eval()
        self.first_stage_model.train = disabled_train
        for param in self.first_stage_model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def encode_first_stage(self, x):
        return self.first_stage_model.encode(x)

    def get_first_stage_encoding(self, encoder_posterior):
        if isinstance(encoder_posterior, DiagonalGaussianDistribution):
            z = encoder_posterior.sample()
        elif isinstance(encoder_posterior, torch.Tensor):
            z = encoder_posterior
        else:
            raise NotImplementedError(f"encoder_posterior of type '{type(encoder_posterior)}' not yet implemented")
        return self.scale_factor * z

    @torch.no_grad()
    def decode_first_stage(self, z):
        z = 1. / self.scale_factor * z
        return self.first_stage_model.decode(z)

    @rank_zero_only
    @torch.no_grad()
    def on_train_batch_start(self, batch, batch_idx):
        # only for very first batch
        if self.scale_by_std and self.current_epoch == 0 and self.global_step == 0 and batch_idx == 0:
            assert self.scale_factor == 1., 'rather not use custom rescaling and std-rescaling simultaneously'
            # set rescale weight to 1./std of encodings
            print("### USING STD-RESCALING ###")
            x = batch['image']
            x = x.to(self.device)
            encoder_posterior = self.encode_first_stage(x)
            z = self.get_first_stage_encoding(encoder_posterior).detach()
            del self.scale_factor
            self.register_buffer('scale_factor', 1. / z.flatten().std())
            print(f"setting self.scale_factor to {self.scale_factor}")
            print("### USING STD-RESCALING ###")

    @torch.no_grad()
    def get_input(self, batch):
        x = batch['image'].to(self.device)
        layer_cond = batch['layer_cond']
        class_cond = batch['class_cond']
        shift_cond = batch['shift_cond']
        c = [layer_cond, class_cond, shift_cond]
        encoder_posterior = self.encode_first_stage(x)
        z = self.scale_factor * self.get_first_stage_encoding(encoder_posterior).detach()
        return z, c, x

    def apply_model(self, x, t, cond):
        layer_cond, class_cond, shift_cond = cond
        return self.model(x=x, timesteps=t, layer_cond=layer_cond, class_cond=class_cond, shift_cond=shift_cond)

    def forward(self, x, c):
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        return self.p_losses(x, c, t)

    def training_step(self, batch, batch_idx):
        z, c, _ = self.get_input(batch)
        loss, loss_dict = self(z, c)

        if batch_idx == 0:
            self.sample_batch = batch
        self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        lr = self.optimizers().param_groups[0]['lr']
        self.log('lr_abs', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        return loss

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self.model)

    def training_epoch_end(self, outputs):
        with self.ema_scope("Plotting"):
            img_save_dir = os.path.join(self.opts.default_root_dir, 'train_progress', str(self.current_epoch))
            os.makedirs(img_save_dir, exist_ok=True)

            z, c, x = self.get_input(self.sample_batch)

            x_samples, denoise_x_row = self.sample(c=c,batch_size=x.shape[0], return_intermediates=True, clip_denoised=True)
            img_samples = self.decode_first_stage(x_samples).to('cpu')
            layer_cond, class_cond, shift_cond = c[0].to('cpu'), c[1].to('cpu'), c[2].to('cpu')
            x_rec = self.decode_first_stage(z).to('cpu')
            x = x.to('cpu')
            for i in range(x.shape[0]):
                save_name = str(i) + '_' + str(class_cond[i].item()) + '_' + str(shift_cond[i].item()) + '.png'
                denoise_img_row = []
                for z_noisy in denoise_x_row:
                    denoise_img_row.append(self.decode_first_stage(z_noisy)[i:i+1].to('cpu'))
                denoise_img_row = torch.cat(denoise_img_row, dim=0) * 0.5 + 0.5
                save_image(denoise_img_row, os.path.join(img_save_dir, 'gen_denoise_'+save_name))

                save_image([x[i] * 0.5 + 0.5, layer_cond[i], img_samples[i] * 0.5 + 0.5],
                           os.path.join(img_save_dir, save_name))
                save_image(x_rec[i] * 0.5 + 0.5, os.path.join(img_save_dir, 'x_rec_'+save_name))
                diffusion_row = []
                for t in range(self.num_timesteps):
                    if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                        t = torch.tensor([t]).repeat(z[i:i+1].shape[0]).to(self.device).long()
                        noise = torch.randn_like(z[i:i+1])
                        z_noisy = self.q_sample(x_start=z[i:i+1], t=t, noise=noise)
                        diffusion_row.append(self.decode_first_stage(z_noisy).to('cpu'))
                diffusion_row = torch.cat(diffusion_row, dim=0)*0.5+0.5
                save_image(diffusion_row, os.path.join(img_save_dir, 'gen_forward_'+save_name))

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)

    def p_losses(self, x_start, cond, t, noise=None):
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_out = self.apply_model(x_noisy, t, cond)

        loss_dict = {}
        if self.parameterization == "eps":
            target = noise
        elif self.parameterization == "x0":
            target = x_start
        else:
            raise NotImplementedError(f"Paramterization {self.parameterization} not yet supported")

        loss = self.get_loss(model_out, target, mean=False).mean(dim=[1, 2, 3])

        log_prefix = 'train' if self.training else 'val'
        loss_dict.update({f'{log_prefix}/loss_simple': loss.mean()})
        loss_simple = loss.mean() * self.l_simple_weight

        loss_vlb = self.get_loss(model_out, target, mean=False).mean(dim=(1, 2, 3))
        loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()
        loss_dict.update({f'{log_prefix}/loss_vlb': loss_vlb})

        loss = loss_simple + self.original_elbo_weight * loss_vlb

        loss_dict.update({f'{log_prefix}/loss': loss})

        return loss, loss_dict

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

    def p_mean_variance(self, x, c, t, clip_denoised):
        model_out = self.apply_model(x, t, c)
        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
        elif self.parameterization == "x0":
            x_recon = model_out
        else:
            raise NotImplementedError()
        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance, x_recon

    @torch.no_grad()
    def p_sample(self, x, c, t, clip_denoised, temperature=1., noise_dropout=0., repeat_noise=False):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance, x0 = self.p_mean_variance(x=x, c=c, t=t, clip_denoised=clip_denoised)
        noise = noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise, x0

    @torch.no_grad()
    def p_sample_loop(self, c, shape, return_intermediates=False, log_every_t=100, clip_denoised=True):
        device = self.betas.device
        b = shape[0]
        x = torch.randn(shape, device=device)
        # intermediates = [x]
        if return_intermediates:
            intermediates_x0 = [x]
        with tqdm(reversed(range(0, self.num_timesteps)), desc='Sampling t', total=self.num_timesteps,mininterval=2) as pbar:
            for i in pbar:
                x, x0 = self.p_sample(x, c, torch.full((b,), i, device=device, dtype=torch.long),
                                      clip_denoised=clip_denoised)
                if return_intermediates and (i % log_every_t == 0 or i == self.num_timesteps - 1):
                    intermediates_x0.append(x0)
        if return_intermediates:
            return x, intermediates_x0
        return x

    @torch.no_grad()
    def sample(self, c=None, batch_size=1, return_intermediates=False, clip_denoised=True):
        return self.p_sample_loop(c, [batch_size, self.channels] + list(self.latent_size),
                                  return_intermediates=return_intermediates, clip_denoised=clip_denoised)

    def configure_optimizers(self):
        lr = self.opts.learning_rate
        params = list(self.model.parameters())
        opt = torch.optim.AdamW(params, lr=lr)

        scheduler_config = {'warm_up_steps': [10000], 'cycle_lengths': [10000000000000], 'f_start': [1e-06],
                            'f_max': [1.0],
                            'f_min': [1.0]}
        scheduler = LambdaLinearScheduler(**scheduler_config)
        print("Setting up LambdaLR scheduler...")
        scheduler = [
            {
                'scheduler': LambdaLR(opt, lr_lambda=scheduler.schedule),
                'interval': 'step',
                'frequency': 1
            }]
        return [opt], scheduler


if __name__ == '__main__':
    parser = get_parser()
    opts = parser.parse_args()
    data_cfg = OmegaConf.load(opts.data_config)
    initExperiment_v2(opts, data_cfg)
    main(opts, data_cfg)
