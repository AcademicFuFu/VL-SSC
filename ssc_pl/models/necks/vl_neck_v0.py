import os
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layers import *
from ..layers import TransformerLayer
from ..utils import flatten_multi_scale_feats, recover_multi_scale_feats

from debug.utils import print_detail as pd, mem


class VisionLanguageNeckV0(nn.Module):

    def __init__(self, layers, img_pos):
        super().__init__()
        self.layers = layers
        self.img_pos = img_pos

    def forward(self, img_feats, text_feats):

        for i, img_feat in enumerate(img_feats):
            b, c, h, w = img_feat.shape
            img_feat = img_feat.flatten(2).transpose(1, 2).reshape(b, h * w, c)
            img_pos = self.img_pos[i]().repeat(b, 1, 1) if self.img_pos else None
            for layer in self.layers:
                img_feat = layer[i](
                    img_feat,
                    text_feats.unsqueeze(0),
                    text_feats.unsqueeze(0),
                    img_pos,
                )
            img_feats[i] = img_feat.transpose(1, 2).reshape(b, c, h, w)

        return img_feats, text_feats

    @classmethod
    def from_conf(cls, conf, **kwargs):
        embed_dims = kwargs.get('embed_dims', None)
        view_scales = kwargs.get('view_scales', None)
        num_layers = conf.get('num_layers', 1)

        pos_embed_conf = conf.get('img_pos_embed', None)
        img_pos = nn.ModuleList([
            LearnableSqueezePositionalEncoding((93, 305), embed_dims, (1, 1)),
            LearnableSqueezePositionalEncoding((47, 153), embed_dims, (1, 1)),
            LearnableSqueezePositionalEncoding((24, 77), embed_dims, (1, 1)),
        ]) if pos_embed_conf else None

        transformers = nn.ModuleList(
            nn.ModuleList([TransformerLayer(embed_dims, 8) for _ in range(len(view_scales))]) for _ in range(num_layers))

        return cls(
            layers=transformers,
            img_pos=img_pos,
        )
