# -*- coding:utf-8 -*-
from argparse import ArgumentParser
import torch
from utils.util_for_opencv_diffusion import DDPM_base
from networks.modified_monai_vit_w_cond_LS_v4_lora_contrast import ViTAutoEnc
import loralib as lora
from networks import modified_monai_vit_w_cond_LS_v4

def get_parser():
    parser = ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, default='') # path/to/finetuned/ckpt
    parser.add_argument('--save_path', type=str, default='') # path/to/merged/ckpt
    return parser


def main(opts):
    model_lora = DDPM()
    ckpt_path = opts.ckpt_path
    save_path = opts.save_path

    sd = torch.load(ckpt_path, map_location='cpu')
    missing, unexpected = model_lora.load_state_dict(sd["state_dict"], strict=False)
    print(f"Restored from {ckpt_path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
    if len(missing) > 0 or len(unexpected) > 0:
        print(f"Missing Keys: {missing}")
        print(f"Unexpected Keys: {unexpected}")

    for name, module in model_lora.model.named_modules():
        if isinstance(module, lora.MergedLinear):
            print('modules:', name, module.training, module.merge_weights, module.merged, module.r, module.enable_lora)

    # test_lora_merge_unmerge(model_lora.model)
    # exit()
    # t = torch.randint(0, 1000, (1,), device=model_lora.device).long()
    # x = torch.zeros((1,1,8,400), device=model_lora.device)
    # c = torch.randint(0, 2, (1,), device=model_lora.device).long()
    # ret1 = model_lora.model(x, t, c)
    # model_lora.model.eval()
    # print(model_lora.model)
    # for name, module in model_lora.model.named_children():
    #     print('children module:', name)
    # model_lora.model(torch.zeros())

            # for n,param in module.named_parameters():
            #     print(n,param)

    model_lora.model.eval()
    for name, module in model_lora.model.named_modules():
        if isinstance(module, lora.MergedLinear):
            print('modules:', name, module.training, module.merge_weights, module.merged, module.r, module.enable_lora)
    # exit()
    # ret2 = model_lora.model(x, t, c)
    # print(torch.abs(ret1-ret2).sum())
    new_net = modified_monai_vit_w_cond_LS_v4.ViTAutoEnc(in_channels=1, patch_size=(8, 1), img_size=(8, 400), spatial_dims=2)
    missing, unexpected = new_net.load_state_dict(model_lora.model.state_dict(), strict=False)
    print(f"Restored from model with {len(missing)} missing and {len(unexpected)} unexpected keys")
    if len(missing) > 0 or len(unexpected) > 0:
        print(f"Missing Keys: {missing}")
        print(f"Unexpected Keys: {unexpected}")
    model_lora.model = new_net
    sd["state_dict"] = model_lora.state_dict()

    torch.save(sd, save_path)


def test_lora_merge_unmerge(model):
    t = torch.randint(0, 1000, (1,), device='cpu').long()
    x = torch.zeros((1,1,8,400), device='cpu')
    c = torch.randint(0, 2, (1,), device='cpu').long()
    s = torch.rand((1, ), device='cpu')

    initial_weight = model.blocks[0].attn.qkv.weight.clone()
    model.train()
    assert torch.equal(model.blocks[0].attn.qkv.weight, initial_weight)

    # perform an update to the LoRA weights
    lora.mark_only_lora_as_trainable(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    model(x,t,c,s).sum().backward()
    optimizer.step()
    optimizer.zero_grad()
    # the weight remains unchanged (only lora A and B change)
    assert torch.equal(model.blocks[0].attn.qkv.weight, initial_weight)

    # 'merge' and then 'unmerge' should neutralize themselves
    weight_before = model.blocks[0].attn.qkv.weight.clone()
    model.eval()
    assert not torch.equal(model.blocks[0].attn.qkv.weight, weight_before)
    model.train()
    # note: numerically, `W + (A * B) - (A * B) == W` does not hold exactly
    assert torch.allclose(model.blocks[0].attn.qkv.weight, weight_before)

    # calling eval/train multiple times in a row should not merge/unmerge multiple times
    model.eval()
    assert model.blocks[0].attn.qkv.merged
    weight_after = model.blocks[0].attn.qkv.weight.clone()
    model.eval()
    model.eval()
    assert torch.equal(model.blocks[0].attn.qkv.weight, weight_after)
    model.train()
    assert not model.blocks[0].attn.qkv.merged
    weight_after = model.blocks[0].attn.qkv.weight.clone()
    model.train()
    model.train()
    assert torch.equal(model.blocks[0].attn.qkv.weight, weight_after)
    print('finished testing')



class DDPM(DDPM_base):
    def __init__(self):
        super().__init__()

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


if __name__ == '__main__':
    parser = get_parser()
    opts = parser.parse_args()
    main(opts)
