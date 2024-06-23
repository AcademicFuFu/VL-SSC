import os
import pdb

import hydra
import lightning as L
from omegaconf import DictConfig, OmegaConf

import sys
import torch

sys.path.append('/home/wangruoyu/workspace/Symphonies')
from ssc_pl import LitModule, build_data_loaders, pre_build_callbacks

from debug.utils import print_detail, mem


@hydra.main(config_path='../configs', config_name='config', version_base=None)
def main(cfg: DictConfig):
    if os.environ.get('LOCAL_RANK', 0) == 0:
        print(OmegaConf.to_yaml(cfg))
    cfg, callbacks = pre_build_callbacks(cfg)

    dls, meta_info = build_data_loaders(cfg.data)
    model = LitModule(**cfg, **meta_info)

    # data_iter = iter(dls[0])
    # desired_index = 2
    data_iter = iter(dls[1])
    desired_index = 0
    for _ in range(desired_index + 1):
        batch = next(data_iter)

    for i in range(len(batch)):
        for k in batch[i].keys():
            if type(batch[i][k]) == torch.Tensor:
                batch[i][k] = batch[i][k]
    memory = 0.95
    torch.cuda.set_per_process_memory_fraction(memory)

    model_out = model(batch[0])
    loss = model.model.loss(model_out, batch[1])
    print(loss)
    mem()
    loss = torch.sum(torch.stack([torch.sum(value) for value in loss.values()]))
    pdb.set_trace()
    loss.backward()
    mem()


if __name__ == '__main__':
    main()
