import numpy as np


class PointCloudAugment:

    def __init__(self, normal):
        self.normal = normal

    def __call__(self, points):

        points = points.copy()

        points = self.random_rotate_z(points)

        points = self.random_scale(points)

        points = self.random_shift(points)

        # points = self.random_jitter(points)

        # points = self.random_dropout(points)

        # print(f"augment over.")

        return points

    def random_rotate_z(self, points):
        """
        points: (B, N, 3) 或 (B, N, C)
        """
        B = points.shape[0]

        theta = np.random.uniform(
            0,
            2 * np.pi,
            B
        )

        cos = np.cos(theta)
        sin = np.sin(theta)

        R = np.zeros((B, 3, 3))

        R[:, 0, 0] = cos
        R[:, 0, 1] = -sin
        R[:, 1, 0] = sin
        R[:, 1, 1] = cos
        R[:, 2, 2] = 1

        xyz = points[:, :, :3]

        points[:, :, :3] = np.matmul(
            xyz,
            R.transpose(0, 2, 1)
        )

        if self.normal:
            points[:, :, 3:6] = np.matmul(
                points[:, :, 3:6],
                R.transpose(0, 2, 1)
            )

        return points

    def random_scale(self, points,
                     scale_low=0.9,
                     scale_high=1.1):
        B = points.shape[0]

        scales = np.random.uniform(
            scale_low,
            scale_high,
            (B, 1, 1)
        )

        points[:, :, :3] *= scales

        return points

    def random_shift(self, points,
                     shift=0.05):
        B = points.shape[0]

        shifts = np.random.uniform(
            -shift,
            shift,
            (B, 1, 3)
        )

        points[:, :, :3] += shifts

        return points

    def random_jitter(self, points,
                      sigma=0.005,
                      clip=0.02):
        noise = np.clip(
            sigma * np.random.randn(
                *points[:, :, :3].shape
            ),
            -clip,
            clip
        )

        points[:, :, :3] += noise

        return points

    def random_dropout(self, points,
                       max_dropout_ratio=0.3):

        B, N, _ = points.shape

        for i in range(B):

            ratio = np.random.random() * max_dropout_ratio

            drop_idx = np.where(
                np.random.random(N) <= ratio
            )[0]

            if len(drop_idx) > 0:
                points[i, drop_idx, :3] = points[i, 0, :3]

        return points

    def pc_normalize(self, points):
        """
        对多 batch 点云进行归一化，使每个样本的点云中心位于原点，并缩放到单位球内。

        参数:
            pc: ndarray, 形状为 (B, N, C)
                B - batch size
                N - 每个点云的点数
                C - 坐标维度 (通常是 3)

        返回:
            pc_normalized: ndarray, 形状为 (B, N, C)，归一化后的点云
        """
        pc = points[:, :, :3]

        # 1. 计算每个 batch 的质心，形状 (B, 1, C)
        centroid = np.mean(pc, axis=1, keepdims=True)

        # 2. 平移到原点
        pc_centered = pc - centroid

        # 3. 计算每个点到原点的距离，形状 (B, N)
        dist = np.sqrt(np.sum(pc_centered ** 2, axis=2))

        # 4. 每个 batch 的最大距离，形状 (B,)
        max_dist = np.max(dist, axis=1)

        # 5. 缩放，max_dist 变形为 (B, 1, 1) 以便广播
        #    为避免全零样本导致除零，加一个小 epsilon（可根据需要去除）
        epsilon = 1e-8
        pc_normalized = pc_centered / (max_dist[:, np.newaxis, np.newaxis] + epsilon)

        points[:, :, :3] = pc_normalized

        return points
