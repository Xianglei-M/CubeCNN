import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.CubeConv_modules import CubeConv, CubeConvLeaves, CCFeatProp, CCLeavFProp

from models.CubeConv_utils import split_cube, valid_cube


class CubeBlock_P2P(nn.Module):
    def __init__(self, num_class, parameters, normal_channel=True):
        super(CubeBlock_P2P, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.num_class = num_class
        self.normal_channel = normal_channel
        if normal_channel:
            additional_channel = 3
        else:
            additional_channel = 0
        self.split_list = np.array(parameters['split_list'], dtype=np.int32)

        # -------------------------
        # cbconv_block构建
        # -------------------------
        self.kernel_list = np.flip(self.split_list)
        self.conv_list = np.array(parameters['conv_list'], dtype=np.int32)
        self.samples = np.array(parameters['samples'], dtype=np.int32)
        sample_times = len(self.samples)

        conv_channels = []
        self.conv_block = nn.ModuleList()
        last_channel = parameters['in_channel']
        for i, out_channel in enumerate(self.conv_list):
            if i == 0:
                lvconv = CubeConvLeaves(in_channel=last_channel, out_channel=out_channel, sample_list=self.samples)
                self.conv_block.append(lvconv)

                last_channel = out_channel * sample_times

                conv_channels.append(last_channel)
            else:
                kernel = self.kernel_list[i - 1]

                cbconv = CubeConv(in_channel=last_channel, out_channel=out_channel, split=kernel)
                self.conv_block.append(cbconv)

                last_channel = out_channel

                conv_channels.append(last_channel)

        # -------------------------
        # CubeConvFP构建
        # -------------------------
        conv_channels_reversed = list(reversed(conv_channels))
        conv_list_reversed = list(reversed(self.conv_list))
        self.n_layers = len(conv_list_reversed)

        self.featprop_block = nn.ModuleList()
        for i, out_channel in enumerate(conv_list_reversed):
            if i == self.n_layers - 1:
                leavesfp = CCLeavFProp(in_channel=last_channel + parameters['in_channel'], out_channel=out_channel)
                self.featprop_block.append(leavesfp)

            else:
                cubesfp = CCFeatProp(in_channel=last_channel + conv_channels_reversed[i + 1], out_channel=out_channel)
                self.featprop_block.append(cubesfp)

            last_channel = out_channel


    def forward(self, points):
        """

        :param points:
        :return:
        """
        points = points
        print(f"    Leaves_pts : {points.shape}")

        cubes_list, side_list, chunks_list = split_cube([0, 0, 0], L=2, split_list=self.split_list)

        idx_list, valcub_list, table_list = valid_cube(cubes_list, side_list, chunks_list, points)

        pts_fts = None
        if self.normal_channel:
            pts_fts = points[:, 3:]
        else:
            pts_fts = points[:, 0:3]

        convfts_layers = []
        # n_layers = len(self.split_list) + 1
        for i in range(self.n_layers):
            if i == 0:
                lvconv = self.conv_block[i]

                lvconv_fts = lvconv(down_pts=points[:, 0:3], down_fts=pts_fts, parent_idx=idx_list[i],
                                    cubes=valcub_list[i],
                                    lookup_table=table_list[i], side=side_list[i], chunks=chunks_list[i])
                convfts_layers.append(lvconv_fts)

                print(f"        -> CubeConvLeaves : {lvconv_fts.shape}")
            else:
                cbconv = self.conv_block[i]

                cbconv_fts = cbconv(down_pts=valcub_list[i - 1], down_fts=convfts_layers[i - 1], cubes=valcub_list[i],
                                    lookup_table=table_list[i], chunks=chunks_list[i])
                convfts_layers.append(cbconv_fts)

                print(f"        -> CubeConv : {cbconv_fts.shape}")

        root_fts = convfts_layers[-1]

        convfts_layers_reversed = list(reversed(convfts_layers))

        idx_list_reversed = list(reversed(idx_list))
        valcub_list_reversed = list(reversed(valcub_list))
        table_list_reversed = list(reversed(table_list))
        chunks_list_reversed = list(reversed(chunks_list))
        side_list_reversed = list(reversed(side_list))

        for j in range(self.n_layers):

            if j == self.n_layers - 1:
                lvfprop = self.featprop_block[j]
                lvfprop_fts = lvfprop(up_fts=convfts_layers_reversed[j], lookup_table=table_list_reversed[j],
                                      chunks=chunks_list_reversed[j],
                                      down_paridx=idx_list_reversed[j], down_fts=pts_fts)  # lvfprop_fts : [N_points, C]

                pts_fts = torch.cat([pts_fts, lvfprop_fts], dim=-1)
                # pts_fts = lvfprop_fts
                print(f"        -> CCLeavFProp : {pts_fts.shape}")
            else:
                cbfprop = self.featprop_block[j]

                cbfprop_fts = cbfprop(up_fts=convfts_layers_reversed[j], lookup_table=table_list_reversed[j],
                                      chunks=chunks_list_reversed[j], split=self.split_list[j],
                                      down_cubes=valcub_list_reversed[j + 1], down_fts=convfts_layers_reversed[j + 1])
                convfts_layers_reversed[j + 1] = cbfprop_fts

                print(f"        -> CCFeatProp : {cbfprop_fts.shape}")

        return root_fts, pts_fts


class CubeBlock_P2All(nn.Module):
    def __init__(self, num_class, parameters, normal_channel=True):
        super(CubeBlock_P2All, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.num_class = num_class
        self.normal_channel = normal_channel
        if normal_channel:
            additional_channel = 3
        else:
            additional_channel = 0
        self.split_list = np.array(parameters['split_list'], dtype=np.int32)

        # -------------------------
        # cbconv_block构建
        # -------------------------
        self.kernel_list = np.flip(self.split_list)
        self.conv_list = np.array(parameters['conv_list'], dtype=np.int32)
        self.samples = np.array(parameters['samples'], dtype=np.int32)
        sample_times = len(self.samples)
        self.n_layers = len(self.conv_list)

        self.conv_block = nn.ModuleList()
        last_channel = parameters['in_channel']
        for i, out_channel in enumerate(self.conv_list):
            if i == 0:
                lvconv = CubeConvLeaves(in_channel=last_channel, out_channel=out_channel, sample_list=self.samples)
                self.conv_block.append(lvconv)

                last_channel = out_channel * sample_times
            else:
                kernel = self.kernel_list[i - 1]

                cbconv = CubeConv(in_channel=last_channel, out_channel=out_channel, split=kernel)
                self.conv_block.append(cbconv)

                last_channel = out_channel

    def forward(self, points):
        """

        :param points:
        :return:
        """
        points = points
        print(f"    Leaves_pts : {points.shape}")

        cubes_list, side_list, chunks_list = split_cube([0, 0, 0], L=2, split_list=self.split_list)

        idx_list, valcub_list, table_list = valid_cube(cubes_list, side_list, chunks_list, points)

        pts_fts = None
        if self.normal_channel:
            pts_fts = points[:, 3:]
        else:
            pts_fts = points[:, 0:3]

        convfts_layers = []
        # n_layers = len(self.split_list) + 1
        for i in range(self.n_layers):
            if i == 0:
                lvconv = self.conv_block[i]

                lvconv_fts = lvconv(down_pts=points[:, 0:3], down_fts=pts_fts, parent_idx=idx_list[i],
                                    cubes=valcub_list[i],
                                    lookup_table=table_list[i], side=side_list[i], chunks=chunks_list[i])
                convfts_layers.append(lvconv_fts)

                print(f"        -> CubeConvLeaves : {lvconv_fts.shape}")
            else:
                cbconv = self.conv_block[i]

                cbconv_fts = cbconv(down_pts=valcub_list[i - 1], down_fts=convfts_layers[i - 1], cubes=valcub_list[i],
                                    lookup_table=table_list[i], chunks=chunks_list[i])
                convfts_layers.append(cbconv_fts)

                print(f"        -> CubeConv : {cbconv_fts.shape}")

        root_fts = convfts_layers[-1]

        return root_fts