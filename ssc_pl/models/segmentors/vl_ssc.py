import torch.nn as nn

from ... import build_from_configs
from .. import encoders
from ..encoders.text_encoder import ClipTransformerEncoder
from ..necks import *
from ..decoders import SymphoniesDecoder
from ..losses import ce_ssc_loss, frustum_proportion_loss, geo_scal_loss, sem_scal_loss


class VL_SSC(nn.Module):

    def __init__(
            self,
            vision_encoder,
            text_encoder,
            embed_dims,
            scene_size,
            view_scales,
            volume_scale,
            num_classes,
            num_layers=3,
            image_shape=(370, 1220),
            neck=None,
            voxel_size=0.2,
            downsample_z=2,
            class_weights=None,
            criterions=None,
            **kwargs,
    ):
        super().__init__()
        self.volume_scale = volume_scale
        self.num_classes = num_classes
        self.class_weights = class_weights
        self.criterions = criterions

        self.vision_encoder = build_from_configs(encoders, vision_encoder, embed_dims=embed_dims, scales=view_scales)
        self.text_encoder = ClipTransformerEncoder.from_conf(text_encoder, embed_dims=embed_dims)

        neck_type = globals()[neck.get('type', None)] if neck else None
        self.neck = neck_type.from_conf(
            neck,
            embed_dims=embed_dims,
            view_scales=view_scales,
        ) if neck else None

        self.decoder = SymphoniesDecoder(embed_dims,
                                         num_classes,
                                         num_layers=num_layers,
                                         num_levels=len(view_scales),
                                         scene_shape=scene_size,
                                         project_scale=volume_scale,
                                         image_shape=image_shape,
                                         voxel_size=voxel_size,
                                         downsample_z=downsample_z)

    def forward(self, inputs):
        pred_insts = self.vision_encoder(inputs['img'])
        pred_masks = pred_insts.pop('pred_masks', None)
        img_feats = pred_insts.pop('feats')

        text_feats = self.text_encoder(inputs['img'].device)

        img_feats, text_feats = self.neck(img_feats, text_feats) if self.neck else (img_feats, text_feats)

        depth, K, E, voxel_origin, projected_pix, fov_mask = list(
            map(lambda k: inputs[k], ('depth', 'cam_K', 'cam_pose', 'voxel_origin', f'projected_pix_{self.volume_scale}',
                                      f'fov_mask_{self.volume_scale}')))

        outs = self.decoder(pred_insts, img_feats, pred_masks, depth, K, E, voxel_origin, projected_pix, fov_mask)
        return {'ssc_logits': outs[-1], 'aux_outputs': outs}

    def loss(self, preds, target):
        loss_map = {
            'ce_ssc': ce_ssc_loss,
            'sem_scal': sem_scal_loss,
            'geo_scal': geo_scal_loss,
            'frustum': frustum_proportion_loss
        }

        target['class_weights'] = self.class_weights.type_as(preds['ssc_logits'])
        losses = {}
        if 'aux_outputs' in preds:
            for i, pred in enumerate(preds['aux_outputs']):
                scale = 1 if i == len(preds['aux_outputs']) - 1 else 0.5
                for loss in self.criterions:
                    losses['loss_' + loss + '_' + str(i)] = loss_map[loss]({'ssc_logits': pred}, target) * scale
        else:
            for loss in self.criterions:
                losses['loss_' + loss] = loss_map[loss](preds, target)
        return losses
