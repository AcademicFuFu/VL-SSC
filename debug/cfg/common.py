import yaml
import platform
import os

save_root = './save/'

cfg = open('./debug/semantickitti.yaml', 'r')
cfg = yaml.safe_load(cfg)

learning_map = cfg['learning_map']
learning_ignore = cfg['learning_ignore']
color_map = cfg['color_map']
learning_map_inv = cfg['learning_map_inv']
