import numpy as np
# import open3d as o3d
import torch

import time


from data_utils.ModelNetDataLoader import ModelNetDataLoader

from models.CubeCNN_cls import CubeCNN
from models.CubeConv_utils import split_cube, valid_cube, build_chicube_fts, combine_chicube_fts


# from visualization_utils import o3d_show

# import provider

def o3d_show(points, color=None):
    if len(points) == 0:
        return None
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, 0:3])
    # if voxel_size != 0:
    #     pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    if color is not None:
        pcd.paint_uniform_color(color)
    o3d.visualization.draw_geometries([pcd])
    return pcd


'''
def data_augmentation(data_batches):
    data_batches = data_batches.numpy()

    data_batches = provider.random_point_dropout(data_batches)
    data_batches[:, :, 0:3] = provider.random_scale_point_cloud(data_batches[:, :, 0:3])
    data_batches[:, :, 0:3] = provider.shift_point_cloud(data_batches[:, :, 0:3])

    data_batches = torch.from_numpy(data_batches)
    return data_batches
'''

def test_01():
    """
    test_04 : 专用于时间检测
    :return:
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_point = 1024
    normal = False
    # augment = PointCloudAugment(normal)

    DATA_PATH = 'data/modelnet40_normal_resampled/'

    TRAIN_DATASET = ModelNetDataLoader(root=DATA_PATH, npoint=num_point, split='train',
                                       normal_channel=normal)

    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=16, shuffle=True,
                                                  num_workers=4)


    points_batches, shapes_cls = next(iter(trainDataLoader))
    points_batches = points_batches.to(device)

    split_list = [5, 3, 3]
    reversed_split_list = list(reversed(split_list))

    print(f"Points_batches: {points_batches.shape}")

    time_start_split_cube = time.time()
    side_list, chunks_list = split_cube([0, 0, 0], L=2, split_list=split_list)
    time_end_split_cube = time.time()
    print(f"time_split_cube: {time_end_split_cube - time_start_split_cube}")

    B = points_batches.shape[0]
    n_layers = len(split_list) + 1

    for b in range(B):
        points = points_batches[b]
        print(f"    Leaves_pts : {points.shape}")

        time_start_valid_cube = time.time()
        valcub_list, table_list = valid_cube(points, side_list, chunks_list)
        time_end_valid_cube = time.time()
        print(f"    time_valid_cube: {time_end_valid_cube - time_start_valid_cube}")
        for i in range(n_layers):
            print(f"valcub_list[{i}] : {valcub_list[i].shape}, table_list[{i}] : {table_list[i].shape}")

        shape_occu = list(points.shape)
        shape_occu[-1] = 1
        points_occu = torch.ones(*shape_occu).to(device)
        # print(f"build -> points_occu : {points_occu.shape}")

        for i in range(n_layers):
            time_start_build_chicube_fts = time.time()
            chicube_fts = build_chicube_fts(down_pts=points, down_fts=points_occu, cubes=valcub_list[i], lookup_table=table_list[i], side=side_list[i], chunks=chunks_list[i], split=5)
            time_end_build_chicube_fts = time.time()
            print(f"        time_build_chicube_fts: {time_end_build_chicube_fts - time_start_build_chicube_fts}")

            if i == 0:
                # print(f"cbconv -> chicube_fts : {chicube_fts.shape}")
                print(f"        time_combine_chicube_fts: 0")
                continue
            else:
                shape_fts = list(valcub_list[i - 1].shape)
                shape_fts[-1] = 1
                down_fts = torch.ones(*shape_fts).to(device)
                # print(f"combine -> down_fts : {down_fts.shape}")

                time_start_combine_chicube_fts = time.time()
                chicube_fts = combine_chicube_fts(down_cubes=valcub_list[i - 1], down_fts=down_fts, cubes=valcub_list[i],
                                                lookup_table=table_list[i], chunks=chunks_list[i],
                                                split=reversed_split_list[i - 1])
                time_end_combine_chicube_fts = time.time()
                print(f"        time_combine_chicube_fts: {time_end_combine_chicube_fts - time_start_combine_chicube_fts}")



def test_02():
    """
    test_04 : 专用于时间检测
    :return:
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_point = 1024
    normal = False
    # augment = PointCloudAugment(normal)

    DATA_PATH = 'data/modelnet40_normal_resampled/'

    TRAIN_DATASET = ModelNetDataLoader(root=DATA_PATH, npoint=num_point, split='train',
                                       normal_channel=normal)

    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=16, shuffle=True,
                                                  num_workers=4)


    points_batches, shapes_cls = next(iter(trainDataLoader))
    points_batches = points_batches.to(device)

    split_list = [5, 5, 5, 9]
    reversed_split_list = list(reversed(split_list))

    print(f"Points_batches: {points_batches.shape}")

    time_start_split_cube = time.time()
    side_list, chunks_list = split_cube([0, 0, 0], L=2, split_list=split_list)
    time_end_split_cube = time.time()
    print(f"time_split_cube: {time_end_split_cube - time_start_split_cube}")

    B = points_batches.shape[0]
    n_layers = len(split_list) + 1

    for b in range(B):
        points = points_batches[b]
        print(f"    Leaves_pts : {points.shape}")

        time_start_valid_cube = time.time()
        valcub_list, table_list = valid_cube(points, side_list, chunks_list)
        time_end_valid_cube = time.time()
        print(f"    time_valid_cube: {time_end_valid_cube - time_start_valid_cube}")
        # for i in range(n_layers):
        #     print(f"valcub_list[{i}] : {valcub_list[i].shape}, table_list[{i}] : {table_list[i].shape}")

        shape_occu = list(points.shape)
        shape_occu[-1] = 1
        points_occu = torch.ones(*shape_occu).to(device)
        # print(f"build -> points_occu : {points_occu.shape}")

        for i in range(n_layers):
            time_start_build_chicube_fts = time.time()
            chicube_fts = build_chicube_fts(down_pts=points, down_fts=points_occu, cubes=valcub_list[i], lookup_table=table_list[i], side=side_list[i], chunks=chunks_list[i], split=5)
            time_end_build_chicube_fts = time.time()
            print(f"        time_build_chicube_fts: {time_end_build_chicube_fts - time_start_build_chicube_fts}")

            if i == 0:
                # print(f"cbconv -> chicube_fts : {chicube_fts.shape}")
                print(f"        time_combine_chicube_fts: 0")
                continue
            else:
                shape_fts = list(valcub_list[i - 1].shape)
                shape_fts[-1] = 1
                down_fts = torch.ones(*shape_fts).to(device)
                # print(f"combine -> down_fts : {down_fts.shape}")

                time_start_combine_chicube_fts = time.time()
                chicube_fts = combine_chicube_fts(down_cubes=valcub_list[i - 1], down_fts=down_fts, cubes=valcub_list[i],
                                                lookup_table=table_list[i], chunks=chunks_list[i],
                                                split=reversed_split_list[i - 1])
                time_end_combine_chicube_fts = time.time()
                print(f"        time_combine_chicube_fts: {time_end_combine_chicube_fts - time_start_combine_chicube_fts}")



if __name__ == '__main__':
    test_01()
