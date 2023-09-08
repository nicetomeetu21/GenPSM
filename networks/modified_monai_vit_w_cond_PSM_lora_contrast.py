
from typing import Sequence, Union

import torch
import torch.nn as nn

from monai.networks.blocks.patchembedding import PatchEmbeddingBlock
from monai.networks.layers import Conv
from monai.utils import ensure_tuple_rep
from ldm.modules.diffusionmodules.util import timestep_embedding, linear

__all__ = ["ViTAutoEnc"]

from monai.networks.blocks.mlp import MLPBlock

from monai.utils import optional_import

import loralib as lora


Rearrange, _ = optional_import("einops.layers.torch", name="Rearrange")


class SABlock(nn.Module):
    """
    A self-attention block, based on: "Dosovitskiy et al.,
    An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>"
    """

    def __init__(self, hidden_size: int, num_heads: int, dropout_rate: float = 0.0, qkv_bias: bool = False) -> None:
        """
        Args:
            hidden_size: dimension of hidden layer.
            num_heads: number of attention heads.
            dropout_rate: faction of the input units to drop.
            qkv_bias: bias term for the qkv linear layer.

        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden size should be divisible by num_heads.")

        self.num_heads = num_heads
        # self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.input_rearrange = Rearrange("b h (qkv l d) -> qkv b l h d", qkv=3, l=num_heads)
        self.out_rearrange = Rearrange("b h l d -> b l (h d)")
        self.drop_output = nn.Dropout(dropout_rate)
        self.drop_weights = nn.Dropout(dropout_rate)
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim**-0.5

        # self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=qkv_bias)
        self.qkv = lora.MergedLinear(hidden_size, 3*hidden_size, r=32, enable_lora=[True, False, True], bias=qkv_bias)
        self.out_proj = lora.Linear(hidden_size, hidden_size, r=16)

    def forward(self, x):
        output = self.input_rearrange(self.qkv(x))
        q, k, v = output[0], output[1], output[2]
        att_mat = (torch.einsum("blxd,blyd->blxy", q, k) * self.scale).softmax(dim=-1)
        att_mat = self.drop_weights(att_mat)
        x = torch.einsum("bhxy,bhyd->bhxd", att_mat, v)
        x = self.out_rearrange(x)
        x = self.out_proj(x)
        x = self.drop_output(x)
        return x

class TransformerBlock(nn.Module):
    """
    A transformer block, based on: "Dosovitskiy et al.,
    An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>"
    """

    def __init__(
        self, hidden_size: int, mlp_dim: int, num_heads: int, dropout_rate: float = 0.0, qkv_bias: bool = False
    ) -> None:
        """
        Args:
            hidden_size: dimension of hidden layer.
            mlp_dim: dimension of feedforward layer.
            num_heads: number of attention heads.
            dropout_rate: faction of the input units to drop.
            qkv_bias: apply bias term for the qkv linear layer

        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.mlp = MLPBlock(hidden_size, mlp_dim, dropout_rate)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn = SABlock(hidden_size, num_heads, dropout_rate, qkv_bias)
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ViTAutoEnc(nn.Module):
    """
    Vision Transformer (ViT), based on: "Dosovitskiy et al.,
    An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>"

    Modified to also give same dimension outputs as the input size of the image
    """

    def __init__(
        self,
        in_channels: int,
        img_size: Union[Sequence[int], int],
        patch_size: Union[Sequence[int], int],
        out_channels: int = 1,
        deconv_chns: int = 16,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_layers: int = 12,
        num_heads: int = 12,
        pos_embed: str = "conv",
        dropout_rate: float = 0.0,
        spatial_dims: int = 3,
        num_classes: int = 7,
    ) -> None:
        """
        Args:
            in_channels: dimension of input channels or the number of channels for input
            img_size: dimension of input image.
            patch_size: dimension of patch size.
            hidden_size: dimension of hidden layer.
            out_channels: number of output channels.
            deconv_chns: number of channels for the deconvolution layers.
            mlp_dim: dimension of feedforward layer.
            num_layers: number of transformer blocks.
            num_heads: number of attention heads.
            pos_embed: position embedding layer type.
            dropout_rate: faction of the input units to drop.
            spatial_dims: number of spatial dimensions.

        Examples::

            # for single channel input with image size of (96,96,96), conv position embedding and segmentation backbone
            # It will provide an output of same size as that of the input
            >>> net = ViTAutoEnc(in_channels=1, patch_size=(16,16,16), img_size=(96,96,96), pos_embed='conv')

            # for 3-channel with image size of (128,128,128), output will be same size as of input
            >>> net = ViTAutoEnc(in_channels=3, patch_size=(16,16,16), img_size=(128,128,128), pos_embed='conv')

        """

        super().__init__()

        self.patch_size = ensure_tuple_rep(patch_size, spatial_dims)
        self.spatial_dims = spatial_dims

        self.patch_embedding = PatchEmbeddingBlock(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=patch_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            pos_embed=pos_embed,
            dropout_rate=dropout_rate,
            spatial_dims=self.spatial_dims,
        )
        self.blocks = nn.ModuleList(
            [TransformerBlock(hidden_size, mlp_dim, num_heads, dropout_rate) for i in range(num_layers)]
        )
        self.norm = nn.LayerNorm(hidden_size)

        new_patch_size = [4] * self.spatial_dims
        conv_trans = Conv[Conv.CONVTRANS, self.spatial_dims]
        # self.conv3d_transpose* is to be compatible with existing 3d model weights.
        self.conv3d_transpose = conv_trans(hidden_size, deconv_chns, kernel_size=patch_size, stride=patch_size)
        self.conv3d_transpose_1 = conv_trans(in_channels=deconv_chns, out_channels=out_channels, kernel_size=1, stride=1)

        self.model_channels = hidden_size

        self.embed_dim = self.model_channels//4
        self.time_embed = nn.Sequential(
            linear(self.embed_dim,  self.model_channels),
            nn.SiLU(),
            linear( self.model_channels, self.model_channels),
        )
        self.class_embed = nn.Embedding(num_classes, self.model_channels)
        self.shift_embed = nn.Sequential(
            linear(1, self.embed_dim),
            nn.SiLU(),
            linear(self.embed_dim, self.model_channels),
        )
        # self.shift_emb = nn.Embedding(1, self.model_channels)

    def forward(self, x, timesteps, class_cond, shift_cond, out_feat=False):
        """
        Args:
            x: input tensor must have isotropic spatial dimensions,
                such as ``[batch_size, channels, sp_size, sp_size[, sp_size]]``.
        """
        # print(x.shape)
        t_emb = timestep_embedding(timesteps, self.embed_dim, repeat_only=False)
        # print(t_emb.shape)
        t_emb = self.time_embed(t_emb)
        # print(t_emb.shape)

        shift_cond = shift_cond.unsqueeze(1)
        c_emb = self.class_embed(class_cond)
        s_emb = self.shift_embed(shift_cond)
        # print(c_emb.shape, s_emb.shape)


        t_emb = t_emb.unsqueeze(1)
        c_emb = c_emb.unsqueeze(1)
        s_emb = s_emb.unsqueeze(1)

        spatial_size = x.shape[2:]
        x = self.patch_embedding(x)
        patch_num = x.shape[1]
        # print(x.shape)

        x = torch.cat((t_emb, c_emb, s_emb, x), dim=1)

        hidden_states_out = []
        for blk in self.blocks:
            x = blk(x)
            hidden_states_out.append(x)
        x = x[:, -patch_num:, :]
        x = self.norm(x)
        x = x.transpose(1, 2)
        d = [s // p for s, p in zip(spatial_size, self.patch_size)]
        x = torch.reshape(x, [x.shape[0], x.shape[1], *d])
        # print(x.shape)
        x = self.conv3d_transpose(x)
        out = self.conv3d_transpose_1(x)
        if out_feat:
            return out, x, hidden_states_out
        else:
            return out
