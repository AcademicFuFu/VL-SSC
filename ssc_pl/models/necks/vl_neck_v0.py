import os
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layers import *
from ..layers import TransformerLayer
from ..utils import flatten_multi_scale_feats, recover_multi_scale_feats

from debug.utils import print_detail as pd, mem


class VisionLanguageTransformer(nn.Module):

    def __init__(self, embed_dims, view_scales, img_pos=None):
        super().__init__()
        self.norm_img = nn.ModuleList([nn.LayerNorm(embed_dims) for _ in view_scales])
        self.norm_text = nn.LayerNorm(embed_dims)
        self.attn = nn.ModuleList(
            [nn.MultiheadAttention(embed_dims, num_heads=8, bias=False, batch_first=True) for _ in view_scales])
        self.img_pos = img_pos

    def forward(self, img_feats, text_feats):
        text_feats_norm = self.norm_text(text_feats)
        vl_feats = []
        for i, img_feat in enumerate(img_feats):
            b, c, h, w = img_feat.shape
            img_feat = img_feat.flatten(2).transpose(1, 2).reshape(b, h * w, c)
            if self.img_pos is not None:
                img_feat = img_feat + self.img_pos
            vl_feat = self.attn[i](
                self.norm_img[i](img_feat),
                text_feats_norm.unsqueeze(0),
                text_feats.unsqueeze(0),
            )[0]
            vl_feat = vl_feat.transpose(1, 2).reshape(b, c, h, w)
            vl_feats.append(vl_feat)
        return vl_feats


class VisionLanguageNeckV0(nn.Module):

    def __init__(self, domain_transformer, residual_transformers, img_pos):
        super().__init__()
        self.domain_transformer = domain_transformer
        self.residual_transformers = residual_transformers
        self.img_pos = img_pos

    def forward(self, img_feats, text_feats):
        vl_feats = self.domain_transformer(img_feats, text_feats) if self.domain_transformer else img_feats

        for i, vl_feat in enumerate(vl_feats):
            b, c, h, w = vl_feat.shape
            vl_feat = vl_feat.flatten(2).transpose(1, 2).reshape(b, h * w, c)
            for layer in self.residual_transformers:
                vl_feat = layer(
                    vl_feat,
                    text_feats.unsqueeze(0),
                    text_feats.unsqueeze(0),
                )
            vl_feats[i] = vl_feat.transpose(1, 2).reshape(b, c, h, w)

        return vl_feats, text_feats

    @classmethod
    def from_conf(cls, conf, **kwargs):
        embed_dims = kwargs.get('embed_dims', None)
        view_scales = kwargs.get('view_scales', None)
        num_layers = conf.get('num_layers', 1)

        pos_embed_conf = conf.get('pos_embed', None)
        img_pos = None

        feat_domain = conf.get('feat_domain', 'text')
        if feat_domain == 'text':
            domain_transformer = VisionLanguageTransformer(embed_dims, view_scales, img_pos)
        elif feat_domain == 'fused':
            domain_transformer = None
            num_layers += 1

        residual_transformers = nn.ModuleList([TransformerLayer(embed_dims, 8) for _ in range(num_layers)])

        return cls(
            domain_transformer=domain_transformer,
            residual_transformers=residual_transformers,
            img_pos=img_pos,
        )


class VisionLanguageLoftrLayer(nn.Module):

    def __init__(self, embed_dims):
        super().__init__()
        self.text_self_attn = TransformerLayer(embed_dims, 8)
        self.vl_cross_attn = TransformerLayer(embed_dims, 8)
        self.lv_cross_attn = TransformerLayer(embed_dims, 8)

    def forward(self, img_feats, text_feats):
        text_feats = self.text_self_attn(text_feats, text_feats, text_feats)
        img_feats = self.vl_cross_attn(img_feats, text_feats, text_feats)
        text_feats = self.lv_cross_attn(text_feats, img_feats, img_feats)
        return img_feats, text_feats


class VisionLanguageNeckV2(nn.Module):

    def __init__(self, layers, img_pos):
        super().__init__()
        self.layers = layers
        self.img_pos = img_pos

    def forward(self, img_feats, text_feats):
        img_feats_flatten, feat_shapes = flatten_multi_scale_feats(img_feats)
        text_feats = text_feats.unsqueeze(0)
        for layer in self.layers:
            img_feats_flatten, text_feats = layer(img_feats_flatten, text_feats)
        text_feats = text_feats.squeeze(0)
        img_feats = recover_multi_scale_feats(img_feats_flatten, feat_shapes)

        return img_feats, text_feats

    @classmethod
    def from_conf(cls, conf, **kwargs):
        embed_dims = kwargs.get('embed_dims', None)
        view_scales = kwargs.get('view_scales', None)
        num_layers = conf.get('num_layers', 1)

        pos_embed_conf = conf.get('pos_embed', None)
        img_pos = None

        layers = nn.ModuleList([VisionLanguageLoftrLayer(embed_dims) for _ in range(num_layers)])
        return cls(
            layers=layers,
            img_pos=img_pos,
        )
