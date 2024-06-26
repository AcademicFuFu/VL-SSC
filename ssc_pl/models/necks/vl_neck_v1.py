import os
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layers import *
from ..layers import TransformerLayer
from ..utils import flatten_multi_scale_feats, recover_multi_scale_feats

from debug.utils import print_detail as pd, mem
from ..layers.pos_embed import LearnableSqueezePositionalEncoding


class VisionLanguageNeckV1(nn.Module):

    def __init__(self, layers, img_pos):
        super().__init__()
        self.layers = layers
        self.img_pos = img_pos

    def forward(self, img_feats, text_feats):
        img_feats_flatten, feat_shapes = flatten_multi_scale_feats(img_feats)
        bs = img_feats[0].shape[0]
        img_pos = self.img_pos().repeat(bs, 1, 1) if self.img_pos else None
        for layer in self.layers:
            img_feats_flatten = layer(
                img_feats_flatten,
                text_feats.unsqueeze(0),
                text_feats.unsqueeze(0),
                img_pos,
            )
        img_feats = recover_multi_scale_feats(img_feats_flatten, feat_shapes)

        return img_feats, text_feats

    @classmethod
    def from_conf(cls, conf, **kwargs):
        embed_dims = kwargs.get('embed_dims', None)
        num_layers = conf.get('num_layers', 1)

        pos_embed_conf = conf.get('img_pos_embed', None)
        img_pos = LearnableSqueezePositionalEncoding((37404, ), embed_dims, (1, )) if pos_embed_conf else None

        layers = nn.ModuleList([TransformerLayer(embed_dims, 8) for _ in range(num_layers)])
        return cls(
            layers=layers,
            img_pos=img_pos,
        )
