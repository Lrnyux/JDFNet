import torch.nn as nn
from torch.nn import init
from torch.nn import functional as F
import torch
import time
import numpy as np
import math
from inspect import isfunction



def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv2d(dim, dim, 3, padding=1)

    def forward(self, x):
        return self.conv(self.up(x))


class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=32, dropout=0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(groups, dim),
            Swish(),
            nn.Dropout(dropout) if dropout != 0 else nn.Identity(),
            nn.Conv2d(dim, dim_out, 3, padding=1)
        )

    def forward(self, x):
        return self.block(x)


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, dropout=0.0, norm_groups=32):
        super().__init__()
        self.block1 = Block(dim, dim_out, groups=norm_groups)
        self.block2 = Block(dim_out, dim_out, groups=norm_groups, dropout=dropout)
        self.res_conv = nn.Conv2d(
            dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x):
        b, c, h, w = x.shape
        h = self.block1(x)
        h = self.block2(h)
        return h + self.res_conv(x)


class SelfAttention(nn.Module):
    def __init__(self, in_channel, n_head=1, norm_groups=32):
        super().__init__()

        self.n_head = n_head
        self.norm = nn.GroupNorm(norm_groups, in_channel)
        self.qkv = nn.Conv2d(in_channel, in_channel * 3, 1, bias=False)
        self.out = nn.Conv2d(in_channel, in_channel, 1)

    def forward(self, input):
        batch, channel, height, width = input.shape
        n_head = self.n_head
        head_dim = channel // n_head

        norm = self.norm(input)
        qkv = self.qkv(norm).view(batch, n_head, head_dim * 3, height, width)
        query, key, value = qkv.chunk(3, dim=2)  # bhdyx

        attn = torch.einsum(
            "bnchw, bncyx -> bnhwyx", query, key
        ).contiguous() / math.sqrt(channel)
        attn = attn.view(batch, n_head, height, width, -1)
        attn = torch.softmax(attn, -1)
        attn = attn.view(batch, n_head, height, width, height, width)

        out = torch.einsum("bnhwyx, bncyx -> bnchw", attn, value).contiguous()
        out = self.out(out.view(batch, channel, height, width))

        return out + input


class ResnetBlocWithAttn(nn.Module):
    def __init__(self, dim, dim_out, *, norm_groups=32, dropout=0.0, with_attn=False):
        super().__init__()
        self.with_attn = with_attn
        self.res_block = ResnetBlock(
            dim, dim_out, norm_groups=norm_groups, dropout=dropout)
        if with_attn:
            self.attn = SelfAttention(dim_out, norm_groups=norm_groups)

    def forward(self, x):
        x = self.res_block(x)
        if (self.with_attn):
            # print('========')
            # print(x.shape)
            x = self.attn(x)
            # print(x.shape)
            # print('========')
        return x


class SpecNet(nn.Module):
    def __init__(
            self,
            in_channel=3,
            out_channel=3,
            inner_channel=16,
            norm_groups=16,
            channel_mults=(1, 2, 4, 8),
            res_blocks=1,
            dropout=0.1,
            image_size=256
    ):
        super().__init__()

        kernel_size = 3
        num_mults = len(channel_mults)
        pre_channel = inner_channel
        feat_channels = [pre_channel]
        now_res = image_size
        downs = [nn.Conv2d(in_channel, inner_channel,
                           kernel_size=3, padding=1)]


        self.res_blocks = res_blocks

        for ind in range(num_mults):
            is_last = (ind == num_mults - 1)
            channel_mult = inner_channel * channel_mults[ind]
            for res_idx in range(0, res_blocks):
                downs.append(ResnetBlocWithAttn(
                    pre_channel, channel_mult, norm_groups=norm_groups,
                    dropout=dropout, with_attn=False))
                feat_channels.append(channel_mult)
                pre_channel = channel_mult
            if not is_last:
                downs.append(Downsample(pre_channel))
                feat_channels.append(pre_channel)
                now_res = now_res // 2
        self.downs = nn.ModuleList(downs)
        self.mid = nn.ModuleList([
            ResnetBlocWithAttn(pre_channel, pre_channel,
                               norm_groups=norm_groups,
                               dropout=dropout, with_attn=True),
            ResnetBlocWithAttn(pre_channel, pre_channel,
                               norm_groups=norm_groups,
                               dropout=dropout, with_attn=False)
        ])
        ups = []
        for ind in reversed(range(num_mults)):
            is_last = (ind < 1)
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks + 1):
                ups.append(ResnetBlocWithAttn(
                    pre_channel + feat_channels.pop(), channel_mult,
                    norm_groups=norm_groups,
                    dropout=dropout, with_attn=False))
                pre_channel = channel_mult
            if not is_last:

                ups.append(Upsample(pre_channel))
                now_res = now_res * 2

        self.ups = nn.ModuleList(ups)
        self.id = nn.Identity()

        # for mask branch
        self.layer_mask = nn.Sequential(
            nn.Conv2d(pre_channel, pre_channel//2, 3, 1, 1),
            nn.Conv2d(pre_channel//2, pre_channel//2, 3, 1, 1),
            nn.Conv2d(pre_channel//2, 1, 3, 1, 1),
            nn.Sigmoid()
        )

        self.layer_specular = nn.Sequential(
            nn.Conv2d(pre_channel + 1, pre_channel//2, 3, 1, 1),
            nn.Conv2d(pre_channel//2, 3, 3, 1, 1),
            nn.Conv2d(3, 3, 3, 1, 1)
        )

        self.layer_diffuse = nn.Sequential(
            nn.Conv2d(pre_channel + 1 + 3, 3, 3, 1, 1),
            nn.Conv2d(3, 3, 3, 1, 1),
            nn.Conv2d(3, 3, 3, 1, 1)
        )





    def forward(self, x):
        feats = []
        ori_img = self.id(x)
        for idx in range(len(self.downs)):
            layer = self.downs[idx]
            x = layer(x)
            feats.append(x)

        for layer in self.mid:
            x = layer(x)

        for layer in self.ups:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(torch.cat((x, feats.pop()), dim=1))
            else:
                x = layer(x)

        hl_feats = x
        mask = self.layer_mask(hl_feats)
        hl_m_feats = torch.concat([hl_feats,mask],dim=1)
        specular = self.layer_specular(hl_m_feats)
        hl_m_s_feats = torch.concat([hl_feats,mask,specular],dim=1)
        diffuse = ori_img - self.layer_diffuse(hl_m_s_feats)


        return mask, specular, diffuse

if __name__ == '__main__':
    net = SpecNet(in_channel=3, out_channel=3,inner_channel=16, norm_groups=16, channel_mults=[1,2,4,8], res_blocks=3)
    net = net.cuda()
    input = torch.ones((1,3,256,320))
    input = input.cuda()
    mask,specular,output = net(input)
    print(mask.shape, specular.shape, output.shape)


