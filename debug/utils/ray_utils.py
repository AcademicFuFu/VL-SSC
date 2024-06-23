import mmcv
import torch
import numpy as np
import matplotlib.pyplot as plt

from debug.cfg import *
from .common_utils import print_detail
'''
coor,
label_depth.unsqueeze(1),
label_seg.unsqueeze(1),  # 0-1, 2, 3
rays_o,
rays_d,
viewdirs  # 4:7,7:10,10:13
'''
from .common_utils import class_names

class_names = [
    "empty", "car", "bicycle", "motorcycle", "truck", "other-vehicle", "person", "bicyclist", "motorcyclist", "road",
    "parking", "sidewalk", "other-ground", "building", "fence", "vegetation", "trunk", "terrain", "pole", "traffic-sign"
]
kitti_class_ray_nums = torch.Tensor([
    25203950, 4088374, 9874, 37731, 208711, 206232, 33682, 16948, 16402, 21626205, 914066, 8994914, 241960, 6956458,
    5180859, 18193943, 525661, 4828831, 273109, 59692
])


def get_kitti_class_ratio():
    ratio = kitti_class_ray_nums / torch.sum(kitti_class_ray_nums)
    return ratio


def show_kitti_class_info():
    ratio = get_kitti_class_ratio()
    for i in range(20):
        print('class {}: {}, num: {}, ratio: {}'.format(i, class_names[i], kitti_class_ray_nums[i], ratio[i]))
    return


def check_ray_semantic_frequency(rays: torch.Tensor):
    # rays.shape: B, N, 13
    semantics = rays[..., 3].to(torch.int8).squeeze()
    print(torch.unique(semantics))
    freq = torch.bincount(semantics)
    for i in range(20):
        if i < len(freq):
            num = freq[i]
        else:
            num = 0
        print('class {}: {}, {} times'.format(i, class_names[i], num))
    return


def get_ray_semantic_num(rays: torch.Tensor):
    # rays.shape: B, N, 13
    s = torch.zeros(20, dtype=torch.int16).to(rays)
    semantics = rays[..., 3].to(torch.int16).squeeze()
    freq = torch.bincount(semantics)
    for i in range(20):
        if i < len(freq):
            s[i] += freq[i]
    return s.to(torch.int16)


def show_num_of_each_class(num: torch.Tensor):
    for i in range(20):
        print('class {}: {}, num: {}'.format(i, class_names[i], num[i]))


def show_class_info(num: torch.Tensor, class_weights=None, ratio=None):
    for i in range(20):
        out = ''
        out += 'class {}: {}, num: {}'.format(i, class_names[i], num[i])
        out += ' weights: {}'.format(class_weights[i]) if class_weights is not None else ''
        out += ' ratio: {}'.format(ratio[i]) if ratio is not None else ''
        print(out)
    return


def show_rays_info(rays, weights=None):
    n = get_ray_semantic_num(rays)
    show_class_info(n, weights)


def visulize_rays(rays):
    # 创建一个3D坐标系
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 定义射线的起点和方向向量
    rays = rays.cpu().squeeze().numpy()
    ray_o = rays[..., 4:7]
    ray_d = rays[..., 7:10]

    # 绘制射线
    for start, direction in zip(ray_o, ray_d):
        X, Y, Z = zip(start, start + direction)
        ax.plot3D(X, Y, Z, 'blue')

    # 显示坐标轴
    ax.quiver(0, 0, 0, 1, 0, 0, color='red', arrow_length_ratio=0.1)
    ax.quiver(0, 0, 0, 0, 1, 0, color='green', arrow_length_ratio=0.1)
    ax.quiver(0, 0, 0, 0, 0, 1, color='blue', arrow_length_ratio=0.1)
    # 设置图形范围
    ax.set_xlim([-3, 3])
    ax.set_ylim([-3, 3])
    ax.set_zlim([-3, 3])

    # 显示图形
    plt.show()


def ray2depthMap(rays_d, rays_depth_gt, rays_depth_pred):
    tr = torch.tensor([
        -1.857739385241e-03, -9.999659513510e-01, -8.039975204516e-03, -4.784029760483e-03, -6.481465826011e-03,
        8.051860151134e-03, -9.999466081774e-01, -7.337429464231e-02, 9.999773098287e-01, -1.805528627661e-03,
        -6.496203536139e-03, -3.339968064433e-01, 0, 0, 0, 1
    ]).reshape((4, 4)).to(rays_d.device)
    K = torch.tensor([
        7.188560000000e+02, 0.000000000000e+00, 6.071928000000e+02, 0.000000000000e+00, 7.188560000000e+02,
        1.852157000000e+02, 0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00
    ]).reshape((3, 3)).to(rays_d.device)

    print(K)
    rays_d = torch.sum(rays_d[..., np.newaxis, :] * tr[:3, :3], -1)
    print(rays_d[:5, ...])
    uv = torch.sum(rays_d[..., np.newaxis, :] * K, -1).cpu().numpy()
    print(uv[:5, ...])
    uv = uv[:, :2].T
    print(uv.shape)
    print(rays_d.shape)
    print(rays_depth_gt.shape)
    print(rays_depth_pred.shape)

    mask = np.ones(uv.shape[1], dtype=bool)
    mask = np.logical_and(mask, uv[0, :] > 1)
    mask = np.logical_and(mask, uv[0, :] < 1241 - 1)
    mask = np.logical_and(mask, uv[1, :] > 1)
    mask = np.logical_and(mask, uv[1, :] < 370 - 1)
    print(np.sum(mask))
    uv = uv[:, mask]
    rays_depth_gt = rays_depth_gt.cpu().numpy()
    rays_depth_pred = rays_depth_pred.cpu().numpy()
    rays_depth_gt = rays_depth_gt[mask]
    rays_depth_pred = rays_depth_pred[mask]

    import cv2
    mmcv.mkdir_or_exist(debug_root + '/render_depth/')
    depth_map_gt = np.zeros((371, 1241))
    depth_map_pred = np.zeros((371, 1241))

    for i in range(uv.shape[1]):
        point = uv[:, i]
        u, v = point[0:2]
        depth_map_gt[int(v)][int(u)] = rays_depth_gt[i]
    depth_map_gt = cv2.applyColorMap(np.uint8(depth_map_gt * 255 / rays_depth_gt.max()), cv2.COLORMAP_VIRIDIS)
    mmcv.imwrite(depth_map_gt, debug_root + '/render_depth/depth_map_gt.jpg')

    for i in range(uv.shape[1]):
        point = uv[:, i]
        u, v = point[0:2]
        depth_map_pred[int(v)][int(u)] = rays_depth_pred[i]
    depth_map_pred = cv2.applyColorMap(np.uint8(depth_map_pred * 255 / rays_depth_pred.max()), cv2.COLORMAP_VIRIDIS)
    mmcv.imwrite(depth_map_pred, debug_root + '/render_depth/depth_map_pred.jpg')

    return


def ray2pts(rays: torch.Tensor, T_velo2cam: torch.Tensor, visulize=False):
    '''
    coor,
    label_depth.unsqueeze(1),
    label_seg.unsqueeze(1),  # 0-1, 2, 3
    rays_o,
    rays_d,
    viewdirs  # 4:7,7:10,10:13
    '''
    rays = rays.squeeze()
    print_detail(rays)
    rays_o = rays[..., 4:7]
    rays_d = rays[..., 7:10]
    rays_depth = rays[..., 2].unsqueeze(0)
    print_detail(rays_o)
    print_detail(rays_d)
    print_detail(rays_depth)
    # print(rays_o[:5, ...])
    pts = rays_o + rays_depth.T * rays_d
    print(pts.shape)

    if visulize is True:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        import numpy as np

        pts = pts.cpu().numpy()
        x = pts[:, 0]
        y = pts[:, 1]
        z = pts[:, 2]

        # 创建3D图形对象
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # 绘制点云
        ax.scatter(x, y, z, c='b', marker='o')

        # 设置坐标轴标签
        ax.set_xlabel('X轴')
        ax.set_ylabel('Y轴')
        ax.set_zlabel('Z轴')

        # 显示图形
        plt.show()
    return pts
