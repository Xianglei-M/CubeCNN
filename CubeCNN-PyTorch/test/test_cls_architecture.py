import numpy as np
import torch

import time


from data_utils.ModelNetDataLoader import ModelNetDataLoader
from data_utils.ShapeNetDataLoader import PartNormalDataset

from models.CubeCNN_cls import CubeCNN
from models.CubeCNN_seg import CubeCNN_PartSeg
from models.CubeConv_utils import split_cube, valid_cube, build_chicube_fts, combine_chicube_fts

from data_utils.data_augmentation import DataAugmentation


# from visualization_utils import o3d_show

# import provider

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

    DATA_PATH = '../data/modelnet40_normal_resampled/'

    TRAIN_DATASET = ModelNetDataLoader(root=DATA_PATH, npoint=num_point, split='train',
                                       normal_channel=normal)

    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=16, shuffle=True,
                                                  num_workers=4)


    points_batches, shapes_cls = next(iter(trainDataLoader))
    points_batches = points_batches.to(device)

    augmentater = DataAugmentation(npoints=num_point, scale_low=0.5, scale_high=1.0, normal=normal)
    points_batches = augmentater(points_batches)

    print(f"Inputs shape: {points_batches.shape}")  # torch.Size([32, 10])
    print(f"Targets shape: {points_batches.shape}")  # torch.Size([32])

    split_list = [50]
    reversed_split_list = list(reversed(split_list))
    cubes_list, side_list, chunks_list = split_cube([0, 0, 0], L=2, split_list=split_list)
    for i, chunks in enumerate(chunks_list):
        cubes = cubes_list[i]
        side = side_list[i]
        print(f"cubes : {cubes.shape}, side = {side}, chunks = {chunks}")

    B = points_batches.shape[0]
    n_layers = len(split_list) + 1

    print(f"Points_batches: {points_batches.shape}")

    for b in range(B):
        points = points_batches[b]
        print(f"    Leaves_pts : {points.shape}")

        idx_list, valcub_list, table_list = valid_cube(cubes_list, side_list, chunks_list, points)
        for i, val in enumerate(valcub_list):
            print(f"idx : {idx_list[i].shape}, val : {val.shape}")

        shape_occu = list(points.shape)
        shape_occu[-1] = 1
        points_occu = torch.ones(*shape_occu).to(device)
        # print(f"build -> points_occu : {points_occu.shape}")

        for i in range(n_layers):
            chicube_fts = build_chicube_fts(down_pts=points[:, 0:3], down_fts=points, down_paridx=idx_list[i], cubes=valcub_list[i], lookup_table=table_list[i], side=side_list[i], chunks=chunks_list[i], split=5)
            print(f"lvconv -> chicube_fts : {chicube_fts.shape}")

            if i == 0:
                print(f"cbconv -> chicube_fts : {chicube_fts.shape}")
                continue
            else:
                shape_fts = list(valcub_list[i - 1].shape)
                shape_fts[-1] = 1
                down_fts = torch.ones(*shape_fts).to(device)
                # print(f"combine -> down_fts : {down_fts.shape}")

                chicube_fts = combine_chicube_fts(down_cubes=valcub_list[i - 1], down_fts=down_fts, cubes=valcub_list[i],
                                                lookup_table=table_list[i], chunks=chunks_list[i],
                                                split=reversed_split_list[i - 1])
                print(f"cbconv -> chicube_fts : {chicube_fts.shape}")


def test_02():
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

    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=32, shuffle=True,
                                                  num_workers=12)

    for epoch in range(100):
        points_batches, shapes_cls = next(iter(trainDataLoader))
        points_batches = points_batches.to(device)

        print(f"Inputs shape: {points_batches.shape}")  # torch.Size([32, 10])
        print(f"Targets shape: {points_batches.shape}")  # torch.Size([32])

        split_list = [5, 3, 3]
        reversed_split_list = list(reversed(split_list))
        cubes_list, side_list, chunks_list = split_cube([0, 0, 0], L=2, split_list=split_list)
        for i, chunks in enumerate(chunks_list):
            side = side_list[i]
            print(f"side = {side}, chunks = {chunks}")

        B = points_batches.shape[0]
        n_layers = len(split_list) + 1

        print(f"Points_batches: {points_batches.shape}")

        for b in range(B):
            points = points_batches[b]
            print(f"    Leaves_pts : {points.shape}")

            valcub_list, table_list = valid_cube(cubes_list, side_list, chunks_list, points)
            for i, val in enumerate(valcub_list):
                print(f"val : {val.shape}")

            shape_occu = list(points.shape)
            shape_occu[-1] = 1
            points_occu = torch.ones(*shape_occu).to(device)
            # print(f"build -> points_occu : {points_occu.shape}")

            for i in range(n_layers):
                chicube_fts = build_chicube_fts(down_pts=points, down_fts=points_occu, cubes=valcub_list[i], lookup_table=table_list[i], side=side_list[i], chunks=chunks_list[i], split=3)
                print(f"lvconv -> chicube_fts : {chicube_fts.shape}")

                if i == 0:
                    print(f"cbconv -> chicube_fts : {chicube_fts.shape}")
                    continue
                else:
                    shape_fts = list(valcub_list[i - 1].shape)
                    shape_fts[-1] = 1
                    down_fts = torch.ones(*shape_fts).to(device)
                    # print(f"combine -> down_fts : {down_fts.shape}")

                    chicube_fts = combine_chicube_fts(down_cubes=valcub_list[i - 1], down_fts=down_fts, cubes=valcub_list[i],
                                                    lookup_table=table_list[i], chunks=chunks_list[i],
                                                    split=reversed_split_list[i - 1])
                    print(f"cbconv -> chicube_fts : {chicube_fts.shape}")

def test_03():
    """
    test_03 : 专用于检测神经网络构造是否正确
    :return:
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_point = 1024
    num_classes = 40
    normal = False
    # augment = PointCloudAugment(normal)

    '''DATA LOADING'''
    DATA_PATH = 'data/modelnet40_normal_resampled/'

    TRAIN_DATASET = ModelNetDataLoader(root=DATA_PATH, npoint=num_point, split='train',
                                       normal_channel=normal)

    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=32, shuffle=True,
                                                  num_workers=8)

    '''MODEL BUILDING'''
    model = CubeCNN(num_classes=num_classes, split_list=[5, 3, 3], cbconv_list=[256, 512, 1024],
                            samples_list=[[2, 3, 4], [2, 3, 4], [3, 5, 7], [5, 7, 9]],
                            lvconv_list=[32, 64, 128, 256],
                            normal_channel=normal).to(device)

    '''TEST TRAINING'''
    model.train()

    points_batches, shapes_cls = next(iter(trainDataLoader))
    points_batches = points_batches.to(device)

    root_fts, root_logits = model(points_batches)

def test_03():
    """
    test_03 : 专用于检测神经网络构造是否正确
    :return:
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_point = 1024
    num_classes = 40
    normal = False
    # augment = PointCloudAugment(normal)

    '''DATA LOADING'''
    DATA_PATH = 'data/modelnet40_normal_resampled/'

    TRAIN_DATASET = ModelNetDataLoader(root=DATA_PATH, npoint=num_point, split='train',
                                       normal_channel=normal)

    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=32, shuffle=True,
                                                  num_workers=8)

    '''MODEL BUILDING'''
    num_class = 10
    parameters_list = []
    params = {
        'in_channel': 3,
        'split_list': [4, 4, 4],
        'cbconv_list': [128, 256, 512],
        'samples_list': [[2, 3, 4], [2, 3, 4], [3, 5, 7], [5, 7, 9]],
        'lvconv_list': [32, 32, 64, 128],
        'out_channel': 896  # out_channel = lvconv_list[-1] * sample_times + cbconv_list[-1]
    }
    parameters_list.append(params)

    params = {
        'in_channel': 128 + additional_channel,  # in_channel = lvconv_list[-1] * (n_layers + 1)
        'split_list': [5, 5, 3],
        'cbconv_list': [128, 256, 512],
        'samples_list': [[2, 3, 4], [2, 3, 4], [3, 5, 7], [5, 7, 9]],
        'lvconv_list': [32, 32, 64, 128],
        'out_channel': 896
    }
    parameters_list.append(params)
    self.model = CubeCNN(num_class=num_class,
                         parameters_list=parameters_list,
                         normal_channel=normal).to(device)

    '''TEST TRAINING'''
    model.train()

    points_batches, shapes_cls = next(iter(trainDataLoader))
    points_batches = points_batches.to(device)

    root_fts, root_logits = model(points_batches)



if __name__ == '__main__':
    test_01()
    '''
    chunks = 25
    side = 2 / 25
    split = 3
    xyz = torch.tensor([-0.6800000071525574, 0.123, 0.234])
    # xyz = np.array([-0.6800000071525574, 0.123, 0.234])
    # xyz = [-0.6800000071525574, 0.123, 0.234]

    points_idx = torch.tensor((xyz + 1.0) // side, dtype=torch.int32)
    print(f"points_idx = ((xyz + 1.0) // side).int() = {points_idx}")

    s = side / split
    pts_idx = torch.tensor((xyz + 1.0) // s, dtype=torch.int32)
    # pts_idx = np.array((xyz + 1.0) // s, dtype=int)
    cube_idx3d = pts_idx // split
    print(f"cube_idx3d = {cube_idx3d}")
    '''
