import numpy as np


def points2voxels(
    pts: np.ndarray,
    point_cloud_range=[0, -25.6, -2.0, 51.2, 25.6, 4.4],
    voxel_size=[0.2, 0.2, 0.2],
):
    # pts = np.array([
    #     [4, 4, 0],
    #     [6, -7, 4],
    #     [0, 0, 0],
    #     [3, -3, 3],
    #     [3, -3, 5],
    #     [-3, -3, 5],
    #     [52, -3, 5],
    #     [5, -35, 5],
    #     [5, 35, 5],
    #     [5, 3, -5],
    # ])

    mask = pts[..., 0] >= point_cloud_range[0]
    mask = np.logical_and(mask, pts[..., 0] < point_cloud_range[3])
    mask = np.logical_and(mask, pts[..., 1] >= point_cloud_range[1])
    mask = np.logical_and(mask, pts[..., 1] < point_cloud_range[4])
    mask = np.logical_and(mask, pts[..., 2] >= point_cloud_range[2])
    mask = np.logical_and(mask, pts[..., 2] < point_cloud_range[5])

    pts = pts[mask]

    point_cloud_range = np.array(point_cloud_range)
    range_min = point_cloud_range[0:3]
    range_max = point_cloud_range[3:6]
    pc_range = range_max - range_min

    voxel_shape = (pc_range / voxel_size).astype(int)
    voxels = np.zeros(voxel_shape)
    print(voxels.shape)

    # x = ((pts[:, 0] - range_min[0]) / voxel_size[0]).astype(int)
    # y = ((pts[:, 1] - range_min[1]) / voxel_size[1]).astype(int)
    # z = ((pts[:, 2] - range_min[2]) / voxel_size[2]).astype(int)
    # print(x)
    # print(y)
    # print(z)
    # voxels[x, y, z] = 1
    coords = ((pts - range_min) / voxel_size).astype(int)
    x_coords = coords[:, 0]
    y_coords = coords[:, 1]
    z_coords = coords[:, 2]
    voxels[x_coords, y_coords, z_coords] = 1
    return voxels
