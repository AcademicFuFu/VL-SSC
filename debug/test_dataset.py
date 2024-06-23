import os
import cv2
import pdb

import hydra
import lightning as L
from omegaconf import DictConfig, OmegaConf

import sys
import torch

sys.path.append('/home/wangruoyu/workspace/Symphonies')
from ssc_pl import LitModule, build_data_loaders, pre_build_callbacks

from debug.utils import *


@hydra.main(config_path='../configs', config_name='config', version_base=None)
def main(cfg: DictConfig):
    if os.environ.get('LOCAL_RANK', 0) == 0:
        print(OmegaConf.to_yaml(cfg))
    cfg, callbacks = pre_build_callbacks(cfg)

    dls, meta_info = build_data_loaders(cfg.data)

    # train
    data_iter = iter(dls[0])
    # # val
    # data_iter = iter(dls[1])

    desired_index = 0
    for _ in range(desired_index + 1):
        batch = next(data_iter)
    print_detail(batch[0], 'data')
    print_detail(batch[1], 'gt')
    print_detail(meta_info, 'meta_info')

    gt_2d = batch[1].get('target_2d', None)
    if gt_2d:
        folder = os.path.join(save_root, 'input and gt')
        folder_input = os.path.join(folder, 'input')
        folder_gt = os.path.join(folder, 'gt')
        os.makedirs(folder_input, exist_ok=True)
        os.makedirs(os.path.join(folder_gt, 'depth'), exist_ok=True)
        os.makedirs(os.path.join(folder_gt, 'semantic'), exist_ok=True)
        # gt
        depth_gt = gt_2d['depth_gt'][0]

        for i in range(depth_gt.shape[0]):
            depth_map = tensor2depthmap(depth_gt[i])
            cv2.imwrite(os.path.join(folder_gt, 'depth', '{0:05d}'.format(i) + ".png"), depth_map)

        semantic_gt = gt_2d['semantic_gt'][0]
        for i in range(semantic_gt.shape[0]):
            semantic_map = tensor2semanticmap(semantic_gt[i])
            cv2.imwrite(os.path.join(folder_gt, 'semantic', '{0:05d}'.format(i) + ".png"), semantic_map)


if __name__ == '__main__':
    main()
