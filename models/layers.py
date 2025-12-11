# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.ops import cus_sample

def _get_act_fn(act_name: str, inplace: bool = True):
    if act_name is None:
        return None
    act_name = act_name.lower()
    if act_name == "relu":
        return nn.ReLU(inplace=inplace)
    if act_name in ("leakyrelu", "leaklyrelu"):
        return nn.LeakyReLU(negative_slope=0.1, inplace=inplace)
    if act_name == "gelu":
        return nn.GELU()
    raise NotImplementedError(f"Unsupported activation: {act_name}")


def _to_2tuple(x):
    if isinstance(x, (tuple, list)):
        assert len(x) == 2
        return tuple(x)
    return (x, x)


class ConvBNReLU(nn.Sequential):
    """
    Convolution + BatchNorm + Activation
    Args:
        in_planes (int): input channels
        out_planes (int): output channels
        kernel_size (int|tuple)
        stride (int|tuple): default 1
        padding (int|tuple): default 0
        dilation (int|tuple): default 1
        groups (int): default 1
        bias (bool): default False
        act_name (str|None): 'relu'|'leakyrelu'|'gelu'|None
        is_transposed (bool): use ConvTranspose2d when True
    """
    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=False,
        act_name="relu",
        is_transposed=False,
    ):
        super().__init__()
        conv_module = nn.ConvTranspose2d if is_transposed else nn.Conv2d
        self.add_module(
            "conv",
            conv_module(
                in_planes,
                out_planes,
                kernel_size=kernel_size,
                stride=_to_2tuple(stride),
                padding=_to_2tuple(padding),
                dilation=_to_2tuple(dilation),
                groups=groups,
                bias=bias,
            ),
        )
        self.add_module("bn", nn.BatchNorm2d(out_planes))
        act = _get_act_fn(act_name) if act_name is not None else None
        if act is not None:
            self.add_module(act_name, act)

class FFN(nn.Module):
    def __init__(self, channels: int, expand: int = 4):
        super().__init__()
        self.linear1 = nn.Linear(channels, channels * expand)
        self.act = nn.GELU()
        self.linear2 = nn.Linear(channels * expand, channels)

    def forward(self, x):
        x = self.linear1(x)
        x = self.act(x)
        x = self.linear2(x)
        return x



class ASPP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.conv1 = ConvBNReLU(in_dim, out_dim, kernel_size=1)
        self.conv2 = ConvBNReLU(in_dim, out_dim, kernel_size=3, dilation=2, padding=2)
        self.conv3 = ConvBNReLU(in_dim, out_dim, kernel_size=3, dilation=5, padding=5)
        self.conv4 = ConvBNReLU(in_dim, out_dim, kernel_size=3, dilation=7, padding=7)
        self.conv5 = ConvBNReLU(in_dim, out_dim, kernel_size=1)
        self.fuse = ConvBNReLU(5 * out_dim, out_dim, 3, 1, 1)

    def forward(self, x: torch.Tensor):
        conv1 = self.conv1(x)
        conv2 = self.conv2(x)
        conv3 = self.conv3(x)
        conv4 = self.conv4(x)
        gp = x.mean((2, 3), keepdim=True)
        conv5 = self.conv5(cus_sample(gp, mode="size", factors=x.size()[2:]))
        return self.fuse(torch.cat((conv1, conv2, conv3, conv4, conv5), 1))


class SIU(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.conv_l_pre_down = ConvBNReLU(in_dim, in_dim, 5, stride=1, padding=2)
        self.conv_l_post_down = ConvBNReLU(in_dim, in_dim, 3, 1, 1)
        self.conv_m = ConvBNReLU(in_dim, in_dim, 3, 1, 1)
        self.conv_s_pre_up = ConvBNReLU(in_dim, in_dim, 3, 1, 1)
        self.conv_s_post_up = ConvBNReLU(in_dim, in_dim, 3, 1, 1)
        self.trans = nn.Sequential(
            ConvBNReLU(3 * in_dim, in_dim, 1),
            ConvBNReLU(in_dim, in_dim, 3, 1, 1),
            ConvBNReLU(in_dim, in_dim, 3, 1, 1),
            nn.Conv2d(in_dim, 3, 1),
        )

    def forward(self, l: torch.Tensor, m: torch.Tensor, s: torch.Tensor, return_feats: bool = False):
        tgt_size = m.shape[2:]
        # down
        l = self.conv_l_pre_down(l)
        l = F.adaptive_max_pool2d(l, tgt_size) + F.adaptive_avg_pool2d(l, tgt_size)
        l = self.conv_l_post_down(l)
        # same
        m = self.conv_m(m)
        # up
        s = self.conv_s_pre_up(s)
        s = cus_sample(s, mode="size", factors=m.shape[2:])
        s = self.conv_s_post_up(s)
        attn = self.trans(torch.cat([l, m, s], dim=1))
        attn_l, attn_m, attn_s = torch.softmax(attn, dim=1).chunk(3, dim=1)
        lms = attn_l * l + attn_m * m + attn_s * s

        if return_feats:
            return lms, dict(attn_l=attn_l, attn_m=attn_m, attn_s=attn_s, l=l, m=m, s=s)
        return lms


class MaskGen(nn.Module):
    def __init__(self, d_sal=64, d_cam=64):
        super().__init__()
        assert d_sal == d_cam
        self.d_model = d_cam
        self.relevance_norm = nn.BatchNorm2d(1)
        self.relevance_acti = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, sf, feats):
        """
        sf: [B,C,1,1]
        feats: [B,C,H,W]
        """
        assert sf.shape[0] == feats.shape[0]
        bs = sf.shape[0]
        assert feats.shape[1] == sf.shape[1] == self.d_model
        mask = torch.cat([F.conv2d(feats[i].unsqueeze(0), sf[i].unsqueeze(0)) for i in range(bs)], 0)
        mask = self.relevance_acti(self.relevance_norm(mask))
        return mask


class HMU(nn.Module):
    def __init__(self, in_c: int, num_groups: int = 6, hidden_dim: int = 32):
        super().__init__()
        self.num_groups = num_groups
        expand_dim = hidden_dim * num_groups
        self.expand_conv = ConvBNReLU(in_c, expand_dim, 1)
        self.gate_genator = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(num_groups * hidden_dim, hidden_dim, 1),
            nn.ReLU(True),
            nn.Conv2d(hidden_dim, num_groups * hidden_dim, 1),
            nn.Softmax(dim=1),
        )
        self.interact = nn.ModuleDict()
        self.interact["0"] = ConvBNReLU(hidden_dim, 3 * hidden_dim, 3, 1, 1)
        for group_id in range(1, num_groups - 1):
            self.interact[str(group_id)] = ConvBNReLU(2 * hidden_dim, 3 * hidden_dim, 3, 1, 1)
        self.interact[str(num_groups - 1)] = ConvBNReLU(2 * hidden_dim, 2 * hidden_dim, 3, 1, 1)

        self.fuse = nn.Sequential(nn.Conv2d(num_groups * hidden_dim, in_c, 3, 1, 1), nn.BatchNorm2d(in_c))
        self.final_relu = nn.ReLU(True)

    def forward(self, x: torch.Tensor):
        xs = self.expand_conv(x).chunk(self.num_groups, dim=1)
        outs = []
        branch_out = self.interact["0"](xs[0])
        outs.append(branch_out.chunk(3, dim=1))
        for group_id in range(1, self.num_groups - 1):
            branch_out = self.interact[str(group_id)](torch.cat([xs[group_id], outs[group_id - 1][1]], dim=1))
            outs.append(branch_out.chunk(3, dim=1))
        group_id = self.num_groups - 1
        branch_out = self.interact[str(group_id)](torch.cat([xs[group_id], outs[group_id - 1][1]], dim=1))
        outs.append(branch_out.chunk(2, dim=1))
        out = torch.cat([o[0] for o in outs], dim=1)
        gate = self.gate_genator(torch.cat([o[-1] for o in outs], dim=1))
        out = self.fuse(out * gate)
        return self.final_relu(out + x)
