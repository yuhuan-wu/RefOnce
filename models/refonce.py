# -*- coding: utf-8 -*-
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from models.layers import ConvBNReLU, FFN, ASPP, SIU, MaskGen, HMU
from utils.ops import cus_sample


class BAA(nn.Module):
    """
    BAA block:
    - update class representation (ref)
    - spatially-adaptive feature modulation for image features
    """
    def __init__(self, num_classes: int, embed_dims: int, init_std: float = 0.02):
        super().__init__()
        assert num_classes == 1
        self.num_classes = num_classes
        self.embed_dims = embed_dims

        # For updating ref
        self.cls_delta_proj = nn.Linear(embed_dims, embed_dims, bias=False)

        self.conv_cls = nn.Linear(embed_dims, embed_dims)
        self.conv_x = nn.Linear(embed_dims, embed_dims)
        self.conv_x_1 = nn.Linear(embed_dims, embed_dims)

        # Projections to get global gamma and beta
        self.gamma_proj = nn.Linear(embed_dims, embed_dims)
        self.beta_proj = nn.Linear(embed_dims, embed_dims)

        # Norms / FFN
        self.norm_x = nn.LayerNorm(embed_dims)
        self.norm_cls = nn.LayerNorm(embed_dims)
        self.norm_ffn1 = nn.LayerNorm(embed_dims)
        self.norm_ffn2 = nn.LayerNorm(embed_dims)
        self.ffn_cls = FFN(embed_dims, 4)
        self.ffn_feat = FFN(embed_dims, 4)

        # init: use pytorch defaults (no external init)
        for m in self.modules():
            if isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, inputs):
        x, ref = inputs # x: (b, c, h, w), ref: (b, 1, c)
        b, c, h, w = x.shape
        
        # --- 0. Prepare inputs ---
        ref_in = ref.view(b, -1, c)
        img_feat_in = x.permute(0, 2, 3, 1).contiguous().view(b, h * w, c)
        
        # Normalize features
        ref_norm = self.conv_cls(self.norm_cls(ref_in))
        
        img_feat_norm = self.norm_x(img_feat_in)
        img_feat_norm_linear = self.conv_x_1(img_feat_norm)
        img_feat_norm = self.conv_x(img_feat_norm)

        # --- 1. Compute Spatial Attention Gate ---
        # Use normalized features for cleaner attention scores
        attn_dot_product = img_feat_norm @ ref_norm.transpose(1, 2) / (c ** 0.5)
        coupled_attn = attn_dot_product.permute(0, 2, 1) # b, k, hw
        spatial_gate = coupled_attn.softmax(dim=-1).view(b, 1, h, w) # b, 1, h, w

        # --- 2. Update Class Representation (ref) ---
        ref_context = coupled_attn.softmax(dim=-1) @ img_feat_norm_linear # b, k, c
        cls_delta = self.cls_delta_proj(ref_context)
        aligned_ref = ref_in + cls_delta
        aligned_ref = (aligned_ref + self.ffn_cls(self.norm_ffn1(aligned_ref))).permute(0, 2, 1).unsqueeze(-1)
        # aligned_ref = aligned_ref # .squeeze(-1) # .permute(0, 2, 1) # Back to (b, k, c)

        # --- 3. Spatially-Adaptive Feature Modulation for Image Features ---
        # Generate global channel-wise gamma and beta from the (original) reference vector
        # Using the un-normalized ref might carry more original signal
        ref_vector = ref_in.squeeze(1) # b, c
        gamma_global = self.gamma_proj(ref_vector).view(b, c, 1, 1)
        beta_global = self.beta_proj(ref_vector).view(b, c, 1, 1)
        
        # Use the spatial gate to make gamma/beta spatially-variant
        # Note: We add 1 to gamma so its base is identity transformation, making learning easier.
        gamma_spatial = (1 + torch.tanh(gamma_global)) * spatial_gate
        beta_spatial = torch.tanh(beta_global) * spatial_gate
        
        # Apply the spatially-adaptive modulation
        aligned_img_feat = gamma_spatial * x + beta_spatial + x # Add residual connection
        aligned_img_feat = aligned_img_feat.view(b, c, h*w).transpose(1,2)
        aligned_img_feat = aligned_img_feat + self.ffn_feat(self.norm_ffn2(aligned_img_feat))
        aligned_img_feat = aligned_img_feat.reshape(b, h, w, c).permute(0, 3, 1, 2)
        
        
        return [aligned_img_feat, aligned_ref]


class TransLayer(nn.Module):
    def __init__(self, out_c: int, last_module=ASPP):
        super().__init__()
        self.c5_down = nn.Sequential(last_module(in_dim=2048, out_dim=out_c))
        self.c4_down = nn.Sequential(ConvBNReLU(1024, out_c, 3, 1, 1))
        self.c3_down = nn.Sequential(ConvBNReLU(512, out_c, 3, 1, 1))
        self.c2_down = nn.Sequential(ConvBNReLU(256, out_c, 3, 1, 1))
        self.c1_down = nn.Sequential(ConvBNReLU(64, out_c, 3, 1, 1))

    def forward(self, xs: List[torch.Tensor]) -> List[torch.Tensor]:
        assert isinstance(xs, (tuple, list)) and len(xs) == 5
        c1, c2, c3, c4, c5 = xs
        c5 = self.c5_down(c5)
        c4 = self.c4_down(c4)
        c3 = self.c3_down(c3)
        c2 = self.c2_down(c2)
        c1 = self.c1_down(c1)
        return [c5, c4, c3, c2, c1]



class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class RefFeatCompute(nn.Module):
    def __init__(self, ref_type=64, channel=64, momentum=0.99, temperature=1):
        super().__init__()
        self.register_buffer('ref_proj', torch.zeros((ref_type, channel, 1, 1)))
        self.register_buffer("momentum", torch.tensor(momentum))
        self.ref_type = ref_type
        self.channel = channel
        self.temperature = temperature

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.feat_extract = nn.Sequential(
            BasicConv2d(channel, channel, kernel_size=3, padding=1),
            BasicConv2d(channel, channel, kernel_size=3, padding=1),
        )
        self.weight_mlp = nn.Sequential(
            nn.Linear(channel, channel * 2),
            nn.ReLU(inplace=True),
            nn.Linear(channel * 2, ref_type)
        )

    def tempered_softmax(self, logits, dim=1):
        logits = logits / self.temperature
        return F.softmax(logits, dim=dim)

    @torch.no_grad()
    def forward(self, x, ref_x=None, ref_types=None):
        bs, c, h, w = x.shape

        # Extract features from input x
        x_feat = self.feat_extract(x)
        x_feat = self.pool(x_feat).view(bs, -1)  # [bs, c]

        # Compute weights for each reference type (output is [bs, ref_type])
        type_logits = self.weight_mlp(x_feat)
        type_weights = self.tempered_softmax(type_logits, dim=1)  # <-- Use tempered softmax

        if self.training and ref_x is not None and ref_types is not None:
            # Update reference features during training
            with torch.no_grad():
                for curr_ref, curr_type in zip(ref_x, ref_types):
                    self.ref_proj[curr_type] = self.momentum * self.ref_proj[curr_type].detach() + (1 - self.momentum) * curr_ref

        # Compute weighted average of reference features
        ref_feats = self.ref_proj.squeeze(-1).squeeze(-1)  # [ref_type, channel]
        
        #top_indices = type_weights.argmax(dim=1)  # [bs]
        #weighted_ref = ref_feats.index_select(0, top_indices)  # [bs, channel]

        # [bs, ref_type] @ [ref_type, channel] -> [bs, channel]
        weighted_ref = torch.matmul(type_weights, ref_feats)

        # Ablation: if ref_x is None but ref_types are provided, use GT class to pick ref vector
        if (ref_x is None) and (ref_types is not None):
            # Select the reference vectors by ground-truth indices
            weighted_ref = ref_feats.index_select(0, ref_types)  # [bs, channel]

        # Reshape to match expected output format
        weighted_ref = weighted_ref.view(bs, -1, 1, 1)  # [bs, c, 1, 1]

        # Compute supervision loss if ref_types is provided
        if ref_types is not None and ref_x is not None:
            # ref_types should be [batch_size] with class indices
            # type_logits is [batch_size, ref_type] with logits for each class
            supervision_loss = F.cross_entropy(type_logits, ref_types)
            return (weighted_ref + ref_x), supervision_loss
        else:
            return weighted_ref * 2 # for consistency


class FeatFusion(nn.Module):
    """
    Dual-source Information Fusion (DSF) + simplified fusion
    """
    def __init__(self, channel=64, low_channel=64, embed_dim=64):
        super().__init__()
        self.beta_proj2 = nn.Linear(channel, low_channel)
        self.gamma_proj2 = nn.Linear(channel, low_channel)
        self.norm2 = nn.InstanceNorm2d(low_channel)
        self.fusion_process2 = BasicConv2d(low_channel, embed_dim, 3, padding=1)
        self.norm_res = nn.InstanceNorm2d(embed_dim)

    def forward(self, x, ref_x):
        x2 = self.norm2(x)
        bs = x.shape[0]
        beta2 = torch.tanh(self.beta_proj2(ref_x.squeeze())).view(bs, -1, 1, 1).expand_as(x2)
        gamma2 = torch.tanh(self.gamma_proj2(ref_x.squeeze())).view(bs, -1, 1, 1).expand_as(x2)
        x2 = self.fusion_process2(F.relu(gamma2 * x2 + beta2))
        x2 = self.norm_res(x2)
        return x2


class BAAFuse(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.feat_fusion = FeatFusion(channels, channels, channels)
        self.bidirect_fusion = BAA(num_classes=1, embed_dims=channels)

    def forward(self, x, ref_x):
        x, ref_x = self.bidirect_fusion([x, ref_x])
        x = self.feat_fusion(x, ref_x)
        return x


# ---------------------------
# RefOnce Model Based on ZoomNet
# ---------------------------
class RefOnce(nn.Module):
    def __init__(self, inner_channel: int = 64, ref_type: int = 64):
        super().__init__()
        self.shared_encoder = timm.create_model(model_name="resnet50", pretrained=True, in_chans=3, features_only=True)
        self.translayer = TransLayer(out_c=inner_channel)
        self.ref_proj = ConvBNReLU(2048, inner_channel, 1)

        self.l_ff = nn.ModuleList([BAAFuse(inner_channel) for _ in range(4)])
        self.m_ff = nn.ModuleList([BAAFuse(inner_channel) for _ in range(4)])
        self.s_ff = nn.ModuleList([BAAFuse(inner_channel) for _ in range(4)])

        self.l_baa = nn.Sequential(*[BAA(num_classes=1, embed_dims=inner_channel) for _ in range(3)])
        self.m_baa = nn.Sequential(*[BAA(num_classes=1, embed_dims=inner_channel) for _ in range(3)])
        self.s_baa = nn.Sequential(*[BAA(num_classes=1, embed_dims=inner_channel) for _ in range(3)])

        self.ref_feat = RefFeatCompute(ref_type=ref_type, channel=inner_channel)
        # self.feat_tune = ...  # not used in inference path

        self.merge_layers = nn.ModuleList([SIU(in_dim=inner_channel) for _ in range(5)])

        self.mask_gen = MaskGen(d_sal=inner_channel, d_cam=inner_channel)
        self.merge1 = ConvBNReLU(inner_channel + 1, inner_channel, 1)
        self.merge2 = ConvBNReLU(inner_channel + 1, inner_channel, 1)
        self.merge3 = ConvBNReLU(inner_channel + 1, inner_channel, 1)
        self.merge4 = ConvBNReLU(inner_channel + 1, inner_channel, 1)
        self.merge5 = ConvBNReLU(inner_channel + 1, inner_channel, 1)

        self.d5 = nn.Sequential(HMU(inner_channel, num_groups=6, hidden_dim=32))
        self.d4 = nn.Sequential(HMU(inner_channel, num_groups=6, hidden_dim=32))
        self.d3 = nn.Sequential(HMU(inner_channel, num_groups=6, hidden_dim=32))
        self.d2 = nn.Sequential(HMU(inner_channel, num_groups=6, hidden_dim=32))
        self.d1 = nn.Sequential(HMU(inner_channel, num_groups=6, hidden_dim=32))

        self.out_layer_00 = ConvBNReLU(inner_channel, 32, 3, 1, 1)
        self.out_layer_01 = nn.Conv2d(32, 1, 1)

    def encoder_translayer(self, x: torch.Tensor) -> List[torch.Tensor]:
        en_feats = self.shared_encoder(x)
        trans_feats = self.translayer(en_feats)  # list of 5 tensors
        return trans_feats

    def body(self, l_scale: torch.Tensor, m_scale: torch.Tensor, s_scale: torch.Tensor, ref_type: torch.Tensor):
        l_en_feats = self.shared_encoder(l_scale)
        m_en_feats = self.shared_encoder(m_scale)
        s_en_feats = self.shared_encoder(s_scale)
        l_trans_feats = self.translayer(l_en_feats)  # [c5,c4,c3,c2,c1]
        m_trans_feats = self.translayer(m_en_feats)
        s_trans_feats = self.translayer(s_en_feats)

        # Inference path: compute ref feature only from types/buffer
        ref_x = self.ref_feat(m_trans_feats[0], ref_x=None, ref_types=None)

        # propagate deltas
        m_trans_feats[0], ref_x = self.m_baa([m_trans_feats[0], ref_x])
        s_trans_feats[0], ref_x_s = self.s_baa([s_trans_feats[0], ref_x])
        l_trans_feats[0], ref_x_l = self.l_baa([l_trans_feats[0], ref_x])

        l_trans_feats[1:] = [l_ff(feats, ref_x_l) for l_ff, feats in zip(self.l_ff, l_trans_feats[1:])]
        m_trans_feats[1:] = [m_ff(feats, ref_x) for m_ff, feats in zip(self.m_ff, m_trans_feats[1:])]
        s_trans_feats[1:] = [s_ff(feats, ref_x_s) for s_ff, feats in zip(self.s_ff, s_trans_feats[1:])]

        feats = []
        for l, m, s, layer in zip(l_trans_feats, m_trans_feats, s_trans_feats, self.merge_layers):
            siu_outs = layer(l=l, m=m, s=s)
            feats.append(siu_outs)

        mask = self.mask_gen(ref_x, feats[0])

        x = self.d5(self.merge5(torch.cat((feats[0], mask), dim=1)))
        x = cus_sample(x, mode="scale", factors=2)
        x = self.d4(x + self.merge4(torch.cat((feats[1], F.interpolate(mask, scale_factor=2)), dim=1)))
        x = cus_sample(x, mode="scale", factors=2)
        x = self.d3(x + self.merge3(torch.cat((feats[2], F.interpolate(mask, scale_factor=4)), dim=1)))
        x = cus_sample(x, mode="scale", factors=2)
        x = self.d2(x + self.merge2(torch.cat((feats[3], F.interpolate(mask, scale_factor=8)), dim=1)))
        x = cus_sample(x, mode="scale", factors=2)
        x = self.d1(x + self.merge1(torch.cat((feats[4], F.interpolate(mask, scale_factor=16)), dim=1)))
        x = cus_sample(x, mode="scale", factors=2)

        logits = self.out_layer_01(self.out_layer_00(x))
        return dict(seg=logits)

    def forward(self, data: dict) -> torch.Tensor:
        return self.test_forward(data)

    @torch.no_grad()
    def test_forward(self, data: dict) -> torch.Tensor:
        output = self.body(
            l_scale=data["image1.5"],
            m_scale=data["image1.0"],
            s_scale=data["image0.5"],
            ref_type=None,
        )
        return output["seg"]