import os
import pdb
import sys
import cv2

import numpy as np

import mmcv
import torch

sys.path.append('/home/wangruoyu/workspace/voxformer-nerfhead')

from debug.cfg import save_root
from debug.cfg import color_map, learning_ignore, learning_map, learning_map_inv


def label2img(coor: np.ndarray, label_seg: np.ndarray, label_depth: np.ndarray, seq, select_id, img_size=[376, 1241]):

    depth_map = np.zeros(img_size)
    seg_map = np.zeros((img_size[0], img_size[1], 3), dtype=float)
    for i in range(coor.shape[0]):
        point = coor[i, :]
        u, v = point[0:2]
        depth_map[int(v)][int(u)] = label_depth[i]
        seg_map[int(v)][int(u)] = color_map[learning_map_inv[label_seg[i]]]

    depth_colored = cv2.applyColorMap(np.uint8(depth_map * 255 / label_depth.max()), cv2.COLORMAP_VIRIDIS)
    # depth
    mmcv.mkdir_or_exist(save_root + '/read_gt/')

    mmcv.imwrite(depth_colored, save_root + '/read_gt/depth{}_{}.jpg'.format(seq, select_id))
    # mmcv.imwrite(img_depth, debug_path + '/lidar2sem/img_with_depth.jpg')

    # label
    mmcv.imwrite(seg_map, save_root + '/read_gt/label{}_{}.jpg'.format(seq, select_id))
    # mmcv.imwrite(img_label, debug_path + '/lidar2sem/img_with_label.jpg')
    return


def visulize_gt(sequence, frame_id):
    data_root = '/home/komorebi/workspace/researches/01_datasets/semantickitti'
    depth_gt_path = '/home/komorebi/workspace/researches/01_datasets/semanticKITTI/gt_renderocc/depth_gt_lidar'
    semantic_gt_path_lidar = '/home/komorebi/workspace/researches/01_datasets/semanticKITTI/gt_renderocc/label_gt_lidar'
    semantic_gt_path_segformer = '/home/komorebi/workspace/researches/01_datasets/semanticKITTI/gt_renderocc/kitti_semantics'

    # rgb_path = os.path.join(data_root, "dataset", "sequences", sequence, "image_2", frame_id + ".png")

    filename = str(sequence).zfill(2) + '_' + str(frame_id).zfill(6) + '.bin'
    filepath = os.path.join(depth_gt_path, filename)
    cam_depth = np.fromfile(filepath, dtype=np.float32, count=-1).reshape(-1, 3)

    filename = str(sequence).zfill(2) + '_' + str(frame_id).zfill(6) + '.bin'
    filepath = os.path.join(semantic_gt_path_lidar, filename)
    cam_label = np.fromfile(filepath, dtype=np.float32, count=-1).reshape(-1, 3)
    coords = cam_depth[:, :2].astype(np.int16)
    depth = cam_depth[:, 2]
    label = cam_label[:, 2]

    gt_type = 'lidar+segformer'
    if gt_type == 'lidar':
        return coords, depth, label
    elif gt_type == 'lidar+segformer':
        assert semantic_gt_path_segformer is not None
        filepath = os.path.join(semantic_gt_path_segformer, str(sequence).zfill(2), str(frame_id).zfill(6) + '.npy')
        semantic_segformer = np.load(filepath)
        mask = semantic_segformer == 10  # sky
        v, u = np.where(mask == True)
        coords_sky = np.stack((u, v)).T
        depth_sky = np.ones_like(u) * 51
        label_sky = np.zeros_like(u)
        coords = np.concatenate((coords, coords_sky))
        depth = np.concatenate((depth, depth_sky))
        label = np.concatenate((label, label_sky))
    label2img(coords, label, depth, sequence, frame_id)
    return


def ray2depthmap(rays_d, rays_depth_gt, rays_depth_pred):
    tr = torch.tensor([
        -1.857739385241e-03, -9.999659513510e-01, -8.039975204516e-03, -4.784029760483e-03, -6.481465826011e-03,
        8.051860151134e-03, -9.999466081774e-01, -7.337429464231e-02, 9.999773098287e-01, -1.805528627661e-03,
        -6.496203536139e-03, -3.339968064433e-01, 0, 0, 0, 1
    ]).reshape((4, 4)).to(rays_d.device)
    k = torch.tensor([
        7.188560000000e+02, 0.000000000000e+00, 6.071928000000e+02, 0.000000000000e+00, 7.188560000000e+02, 1.852157000000e+02,
        0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00
    ]).reshape((3, 3)).to(rays_d.device)

    print(k)
    rays_d = torch.sum(rays_d[..., np.newaxis, :] * tr[:3, :3], -1)
    print(rays_d[:5, ...])
    # k[1][2] = -k[1][2]
    print(k)
    uv = torch.sum(rays_d[..., np.newaxis, :] * k, -1).cpu().numpy()
    print(uv[:5, ...])
    uv = uv[:, :2].t
    print(uv.shape)
    mask = np.ones(uv.shape[1], dtype=bool)
    mask = np.logical_and(mask, uv[0, :] > 1)
    mask = np.logical_and(mask, uv[0, :] < 1241 - 1)
    mask = np.logical_and(mask, uv[1, :] > 1)
    mask = np.logical_and(mask, uv[1, :] < 371 - 1)
    print(np.sum(mask))
    uv = uv[:, mask]
    rays_depth_gt = rays_depth_gt.cpu().numpy()
    rays_depth_pred = rays_depth_pred.cpu().numpy()
    rays_depth_gt = rays_depth_gt[mask]
    rays_depth_pred = rays_depth_pred[mask]
    print(rays_depth_gt.shape)

    import cv2
    debug_path = '/home/wangruoyu/workspace/voxformer/debug'
    mmcv.mkdir_or_exist(debug_path + '/render_depth/')
    depth_map_gt = np.zeros((371, 1241))
    depth_map_pred = np.zeros((371, 1241))

    for i in range(uv.shape[1]):
        point = uv[:, i]
        u, v = point[0:2]
        depth_map_gt[int(v)][int(u)] = rays_depth_gt[i]
    depth_map_gt = cv2.applycolormap(np.uint8(depth_map_gt * 255 / rays_depth_gt.max()), cv2.colormap_viridis)
    mmcv.imwrite(depth_map_gt, debug_path + '/render_depth/depth_map_gt.jpg')

    for i in range(uv.shape[1]):
        point = uv[:, i]
        u, v = point[0:2]
        depth_map_pred[int(v)][int(u)] = rays_depth_pred[i]
    depth_map_pred = cv2.applycolormap(np.uint8(depth_map_pred * 255 / rays_depth_pred.max()), cv2.colormap_viridis)
    mmcv.imwrite(depth_map_pred, debug_path + '/render_depth/depth_map_pred.jpg')

    return


def tensor2depthmap(depth: torch.Tensor, mask: torch.Tensor = None):
    depth = depth.cpu().numpy()
    if mask is not None:
        mask = mask.cpu().numpy()
    else:
        # mask = depth > 0
        mask = np.ones_like(depth, dtype=bool)
    depth_map = cv2.applyColorMap(np.uint8(depth * 255 / depth.max()), cv2.COLORMAP_VIRIDIS)
    depth_map_ = np.zeros_like(depth_map)
    depth_map_[mask] = depth_map[mask]
    return depth_map_


def tensor2semanticmap(semantic: torch.Tensor, mask: torch.Tensor = None):
    semantic = semantic.cpu().numpy()
    if mask is not None:
        mask = mask.cpu().numpy()
    else:
        mask = semantic != 255
    semantic_map = np.ones((mask.shape[0], mask.shape[1], 3)) * 255
    for value in np.unique(semantic[mask]):
        m = semantic == value
        m = np.logical_and(m, mask)
        semantic_map[m] = color_map[learning_map_inv[value]]
    return semantic_map


def ndarray2depthmap(depth: np.ndarray, mask: np.ndarray):
    depth_map = cv2.applyColorMap(np.uint8(depth * 255 / depth.max()), cv2.COLORMAP_VIRIDIS)
    depth_map_ = np.zeros_like(depth_map)
    depth_map_[mask] = depth_map[mask]
    return depth_map_


def ndarray2semanticmap(semantic: np.ndarray, mask: np.ndarray, sky_mask: np.ndarray = None):
    semantic_map = np.ones((mask.shape[0], mask.shape[1], 3)) * 255
    for value in np.unique(semantic[mask]):
        m = semantic == value
        m = np.logical_and(m, mask)
        semantic_map[m] = color_map[learning_map_inv[value]]
    if sky_mask is not None:
        semantic_map[sky_mask] = [0, 0, 0]
    return semantic_map
