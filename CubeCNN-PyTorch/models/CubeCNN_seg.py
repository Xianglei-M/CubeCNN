import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.CubeConv_blocks import CubeBlock_P2P, CubeBlock_P2All

class CubeCNN(nn.Module):
    def __init__(self, num_class, parameters_list, normal_channel=True):
        super(CubeCNN, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.num_classes = num_class
        '''
        self.normal_channel = normal_channel
        if normal_channel:
            additional_channel = 3
        else:
            additional_channel = 0
        '''

        self.cubecnn_block = nn.ModuleList()
        self.n_cnn = len(parameters_list)
        for i, params in enumerate(parameters_list, start=1):
            p2p = CubeBlock_P2P(num_class=num_class, parameters=params, normal_channel=normal_channel)
            self.cubecnn_block.append(p2p)

        # -------------------------
        # classifier构建
        # -------------------------
        last_channel = 3 + parameters_list[0]['conv_list'][0] * self.n_cnn
        self.mlp_classifier = nn.Sequential(
            nn.BatchNorm1d(last_channel),

            nn.Linear(in_features=last_channel, out_features=1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(in_features=1024, out_features=512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(in_features=512, out_features=256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(in_features=256, out_features=128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(in_features=128, out_features=self.num_classes)
        )

    def forward(self, points_batches):
        """

        :param points_batches:
        :return:
        """
        B = points_batches.shape[0]

        fts_batches = []
        for b in range(B):
            points = points_batches[b]
            # print(f"    Leaves_pts : {points.shape}")

            new_points = points.clone()
            root_fts = None
            rootfts_list = []
            leavesfts_list = []
            for i, cnn in enumerate(self.cubecnn_block, start=1):
                cnn = self.cubecnn_block[i - 1]

                root_fts, new_points = cnn(new_points)
                # rootfts_list.append(root_fts)
                leavesfts_list.append(new_points)

                new_points = torch.cat([points[:, 0:3], new_points], dim=-1)
            # root_fts = torch.cat(rootfts_list, dim=1)  # [N_cnn, C]
            leaves_fts = new_points  # [N_cnn, C]

            print(f"    -> Root_fts : {leaves_fts.shape}")
            fts_batches.append(leaves_fts)

        fts_batches = torch.cat(fts_batches, dim=0)
        print(f"-> Fts_batches : {fts_batches.shape}")

        logits_batches = self.mlp_classifier(fts_batches)
        print(f"-> Logits_batches : {logits_batches.shape}")

        return fts_batches, logits_batches

