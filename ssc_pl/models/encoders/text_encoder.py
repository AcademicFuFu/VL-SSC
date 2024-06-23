import clip
import os
import copy
import torch
import torch.nn as nn

import pdb

from debug.utils import print_detail as pd, mem


def get_labels(path):
    all_labels = []
    with open(os.path.join(path)) as ff:
        lines = ff.readlines()
        for line in lines:
            label = line.strip()
            all_labels.append(label)
    return all_labels


class ClipTransformerEncoder(nn.Module):

    def __init__(self, text_token, token_embedding, positional_embedding, transformer, text_projection, ln_final,
                 text_embedding_ch, out_projs):
        super().__init__()
        self.text_token = text_token
        self.dtype = torch.float16
        self.token_embedding = token_embedding
        self.positional_embedding = positional_embedding
        self.transformer = transformer
        self.text_projection = text_projection
        self.ln_final = ln_final
        self.text_embedding_ch = text_embedding_ch
        self.out_projs = out_projs
        return

    def forward(self, device):
        text = self.text_token.to(device)
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        x = self.out_projs(x.float())
        return x

    @classmethod
    def from_conf(cls, conf: dict, **kwargs):
        text_label = conf.get('text_label', None)
        text_token = clip.tokenize(get_labels(text_label)) if text_label else None
        text_encoder, _ = clip.load(conf.get('type'), device='cuda', jit=False)
        token_embedding = copy.deepcopy(text_encoder.token_embedding)
        positional_embedding = copy.deepcopy(text_encoder.positional_embedding)
        transformer = copy.deepcopy(text_encoder.transformer)
        text_projection = copy.deepcopy(text_encoder.text_projection)
        ln_final = copy.deepcopy(text_encoder.ln_final)
        text_embedding_ch = conf.get('text_embedding_ch')
        del text_encoder

        embed_dims = kwargs.get('embed_dims', None)
        out_projs = nn.Sequential(
            nn.Linear(text_embedding_ch, embed_dims),
            nn.BatchNorm1d(embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims, embed_dims),
        )
        return cls(
            text_token=text_token,
            token_embedding=token_embedding,
            positional_embedding=positional_embedding,
            transformer=transformer,
            text_projection=text_projection,
            ln_final=ln_final,
            text_embedding_ch=text_embedding_ch,
            out_projs=out_projs,
        )
