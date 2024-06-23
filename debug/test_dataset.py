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
    print('---------------------train----------------------')
    data_iter = iter(dls[0])
    desired_index = 0
    for _ in range(desired_index + 1):
        batch = next(data_iter)
    print('num_samples: ', len(dls[0]))
    print_detail(batch[0], 'data')
    print_detail(batch[1], 'gt')
    print_detail(meta_info, 'meta_info')

    # val
    print('--------------------- val ----------------------')
    data_iter = iter(dls[1])
    desired_index = 0
    for _ in range(desired_index + 1):
        batch = next(data_iter)
    print('num_samples: ', len(dls[1]))
    print_detail(batch[0], 'data')
    print_detail(batch[1], 'gt')
    print_detail(meta_info, 'meta_info')


if __name__ == '__main__':
    main()
