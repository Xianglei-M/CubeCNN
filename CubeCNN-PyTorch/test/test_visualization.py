import numpy as np
import open3d as o3d
import torch

from data_utils.ModelNetDataLoader import ModelNetDataLoader
from data_utils.ShapeNetDataLoader import PartNormalDataset

from models.CubeCNN_cls import CubeCNN
from models.CubeConv_utils import split_cube, valid_cube, build_chicube_fts, combine_chicube_fts


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
    test_01 : 专用于数据形状 .shape 检测
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

    print(f"Inputs shape: {points_batches.shape}")  # torch.Size([32, 10])
    print(f"Targets shape: {points_batches.shape}")  # torch.Size([32])

    split_list = [3, 3, 3]
    reversed_split_list = list(reversed(split_list))
    cubes_list, side_list, chunks_list = split_cube([0, 0, 0], L=2, split_list=split_list)
    for i, chunks in enumerate(chunks_list):
        cubes = cubes_list[i]
        side = side_list[i]
        print(f"cubes : {cubes.shape}, side = {side}, chunks = {chunks}")

    B = points_batches.shape[0]

    colors = [[0, 0, 1], [1, 0, 0], [0, 0.5, 0.5], [1, 0, 1], [0, 1, 1]]
    colors_N = len(colors)

    print(f"Points_batches: {points_batches.shape}")

    for b in range(B):
        points = points_batches[b]
        print(f"    Leaves_pts : {points.shape}")

        valcub_list, table_list = valid_cube(cubes_list, side_list, chunks_list, points)

        geometry_list = []
        color_i = 0
        for i, val in enumerate(valcub_list):
            print(f"val : {val.shape}")

            pcd = o3d_show(val.cpu().numpy(), color=colors[color_i % colors_N])
            color_i += 1
            geometry_list.append(pcd)

def test_02():
    """
    test_01 : 专用于数据形状 .shape 检测
    :return:
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_point = 1024
    normal = False
    # augment = PointCloudAugment(normal)

    root = 'data/shapenetcore_partanno_segmentation_benchmark_v0_normal/'

    TRAIN_DATASET = PartNormalDataset(root=root, npoints=num_point, split='trainval',
                                      normal_channel=normal)
    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=4, shuffle=True,
                                                  num_workers=8)

    TEST_DATASET = PartNormalDataset(root=root, npoints=num_point, split='test',
                                     normal_channel=normal)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=4, shuffle=False,
                                                 num_workers=8)
    print("The number of training data is: %d" % len(TRAIN_DATASET))
    print("The number of test data is: %d" % len(TEST_DATASET))


    points_batches, shapes_cls, targets = next(iter(trainDataLoader))
    points_batches = points_batches.to(device)

    print(f"Points shape: {points_batches.shape}")  # torch.Size([32, 10])
    print(f"Classes shape: {shapes_cls.shape}")  # torch.Size([32])
    print(f"Targets shape: {targets.shape}")  # torch.Size([32])
    print(f"Targets[:, 0] shape: {targets[:, :, 0].shape}")  # torch.Size([32])
    print(f"Targets.squeeze shape: {targets.squeeze(dim=1).shape}")  # torch.Size([32])

    split_list = [3, 3, 3]
    reversed_split_list = list(reversed(split_list))
    cubes_list, side_list, chunks_list = split_cube([0, 0, 0], L=2, split_list=split_list)
    for i, chunks in enumerate(chunks_list):
        cubes = cubes_list[i]
        side = side_list[i]
        print(f"cubes : {cubes.shape}, side = {side}, chunks = {chunks}")

    B = points_batches.shape[0]

    colors = [[0, 0, 1], [1, 0, 0], [0, 0.5, 0.5], [1, 0, 1], [0, 1, 1]]
    colors_N = len(colors)

    print(f"Points_batches: {points_batches.shape}")

    for b in range(B):
        points = points_batches[b]
        print(f"    Leaves_pts : {points.shape}")

        valcub_list, table_list = valid_cube(cubes_list, side_list, chunks_list, points)

        geometry_list = []
        color_i = 0
        for i, val in enumerate(valcub_list):
            print(f"val : {val.shape}")

            pcd = o3d_show(val.cpu().numpy(), color=colors[color_i % colors_N])
            color_i += 1
            geometry_list.append(pcd)


if __name__ == '__main__':
    test_02()
