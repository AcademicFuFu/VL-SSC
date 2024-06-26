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

    def __init__(self, embed_dims):
        super().__init__()
        self.norm_img = nn.LayerNorm(embed_dims)
        self.norm_text = nn.LayerNorm(embed_dims)
        self.attn = nn.MultiheadAttention(embed_dims, num_heads=8, bias=False, batch_first=True)

    def forward(self, img_feats, text_feats, img_pos):
        text_feats_norm = self.norm_text(text_feats)
        if img_pos is not None:
            img_feats = img_feats + img_pos
        vl_feats = self.attn(
            self.norm_img(img_feats),
            text_feats_norm.unsqueeze(0),
            text_feats.unsqueeze(0),
        )[0]
        return vl_feats


class VisionLanguageNeckV3(nn.Module):

    def __init__(self, domain_transformer, residual_transformers, img_pos):
        super().__init__()
        self.domain_transformer = domain_transformer
        self.residual_transformers = residual_transformers
        self.img_pos = img_pos

    def forward(self, img_feats, text_feats):
        b = img_feats[0].shape[0]
        img_pos = self.img_pos().repeat(b, 1, 1) if self.img_pos else None
        img_feats_flatten, feat_shapes = flatten_multi_scale_feats(img_feats)

        img_feats_flatten = self.domain_transformer(img_feats_flatten, text_feats, img_pos)
        for layer in self.residual_transformers:
            img_feat = layer(
                img_feat,
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

        domain_transformer = VisionLanguageTransformer(embed_dims)
        residual_transformers = nn.ModuleList([TransformerLayer(embed_dims, 8) for _ in range(num_layers)])

        return cls(
            domain_transformer=domain_transformer,
            residual_transformers=residual_transformers,
            img_pos=img_pos,
        )
