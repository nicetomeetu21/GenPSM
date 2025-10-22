import os
import pytorch_lightning as pl
import shutil
import datetime
import sys
from omegaconf import OmegaConf
def initExperiment(opts, cfg=None):
    if opts.reproduce:
        pl.seed_everything(42, workers=True)
        opts.deterministic = opts.reproduce
        opts.benchmark = not opts.reproduce

    if opts.command == 'fit':
        now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        opts.exp_name = opts.exp_name+'_'+now
        opts.default_root_dir = os.path.join(opts.result_root, opts.exp_name)
        if not os.path.exists(opts.default_root_dir):
            os.makedirs(opts.default_root_dir)
            code_dir = os.path.abspath(os.path.dirname(os.getcwd()))
            shutil.copytree(code_dir, os.path.join(opts.default_root_dir, 'code'))
            print('save in', opts.default_root_dir)
        else:
            sys.exit("result_dir exists: "+opts.default_root_dir)

        # solve learning rate
        bs, base_lr =  cfg.data.params.batch_size, opts.base_learning_rate
        if opts.accelerator == 'cpu':
            ngpu = 1
        else:
            ngpu = len(opts.devices)
        if hasattr(opts, 'accumulate_grad_batches'):
            accumulate_grad_batches = opts.accumulate_grad_batches
        else:
            accumulate_grad_batches = 1
        print(f"accumulate_grad_batches = {accumulate_grad_batches}")
        if opts.scale_lr:
            opts.learning_rate = accumulate_grad_batches * ngpu * bs * base_lr
            print(
                "Setting learning rate to {:.2e} = {} (accumulate_grad_batches) * {} (num_gpus) * {} (batchsize) * {:.2e} (base_lr)".format(
                    opts.learning_rate, accumulate_grad_batches, ngpu, bs, base_lr))
        else:
            opts.learning_rate = base_lr
            print("++++ NOT USING LR SCALING ++++")
            print(f"Setting learning rate to {opts.learning_rate:.2e}")

def initExperiment_v2(opts, data_cfg):
    if opts.reproduce:
        pl.seed_everything(42, workers=True)
        opts.deterministic = opts.reproduce
        opts.benchmark = not opts.reproduce
    if opts.command == 'fit':
        OmegaConf.update(data_cfg, "params.batch_size", opts.batch_size)
        OmegaConf.update(data_cfg, "params.num_workers", opts.num_workers)

        now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        opts.exp_name = opts.exp_name+'_'+now
        opts.default_root_dir = os.path.join(opts.result_root, opts.exp_name)
        if not os.path.exists(opts.default_root_dir):
            os.makedirs(opts.default_root_dir)
            code_dir = os.path.abspath(os.path.dirname(os.getcwd()))
            shutil.copytree(code_dir, os.path.join(opts.default_root_dir, 'code'))
            print('save in', opts.default_root_dir)
        else:
            sys.exit("result_dir exists: "+opts.default_root_dir)

        # solve learning rate
        bs, base_lr = opts.batch_size, opts.base_learning_rate
        if opts.accelerator == 'cpu':
            ngpu = 1
        else:
            ngpu = len(opts.devices)
        if hasattr(opts, 'accumulate_grad_batches'):
            accumulate_grad_batches = opts.accumulate_grad_batches
        else:
            accumulate_grad_batches = 1
        print(f"accumulate_grad_batches = {accumulate_grad_batches}")
        if opts.scale_lr:
            opts.learning_rate = accumulate_grad_batches * ngpu * bs * base_lr
            print(
                "Setting learning rate to {:.2e} = {} (accumulate_grad_batches) * {} (num_gpus) * {} (batchsize) * {:.2e} (base_lr)".format(
                    opts.learning_rate, accumulate_grad_batches, ngpu, bs, base_lr))
        else:
            opts.learning_rate = base_lr
            print("++++ NOT USING LR SCALING ++++")
            print(f"Setting learning rate to {opts.learning_rate:.2e}")