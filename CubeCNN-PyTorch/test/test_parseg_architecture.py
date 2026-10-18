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

def test_04():
    """
    test_01 : 专用于数据形状 .shape 检测
    :return:
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_point = 2400
    normal = False
    # augment = PointCloudAugment(normal)

    root = '../data/shapenetcore_partanno_segmentation_benchmark_v0_normal/'

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

    num_class = 16
    num_part = 50
    model = CubeCNN_PartSeg(num_class=num_class, num_part=num_part, split_list=[5, 3, 3],
                                 cbconv_list=[256, 512, 1024],
                                 samples_list=[[2, 3, 4], [2, 3, 4], [3, 5, 7], [5, 7, 9]],
                                 lvconv_list=[32, 64, 128, 256],
                                 normal_channel=normal).to(device)

    model.train()

    points_batches, shapes_cls, targets = next(iter(trainDataLoader))
    points_batches = points_batches.to(device)

    print(f"Points shape: {points_batches.shape}")  # torch.Size([32, 10])
    print(f"Classes shape: {shapes_cls.shape}")  # torch.Size([32])
    print(f"Targets shape: {targets.shape}")  # torch.Size([32])

    leaves_fts, leaves_logits = model(points_batches)


if __name__ == '__main__':
    test_04()
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
