import os
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layers import *
from ..layers import TransformerLayer
from ..utils import flatten_multi_scale_feats, recover_multi_scale_feats

from debug.utils import print_detail as pd, mem


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


class VisionLanguageNeckV1(nn.Module):

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
