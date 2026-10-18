import math
import torch


def farthest_point_sample(xyz, npoint, start_zero=False):
    """
    最远点采样（FPS）。

    Args:
        xyz: [B, N, 3]
        npoint: 要采样的点数，要求 1 <= npoint <= N

    Returns:
        centroids: [B, npoint]，采样点在原始点云中的索引
    """
    if xyz.ndim != 3 or xyz.shape[-1] != 3:
        raise ValueError("xyz 应为 [B, N, 3]")

    B, N, _ = xyz.shape

    if not 1 <= npoint <= N:
        raise ValueError(f"npoint 必须在 [1, {N}] 内，实际为 {npoint}")

    device = xyz.device
    batch_indices = torch.arange(B, device=device)

    centroids = torch.empty(
        B, npoint, dtype=torch.long, device=device
    )

    # 每个样本随机选择一个起始点。
    if start_zero:
        farthest = torch.zeros(B, dtype=torch.long, device=device)
    else:
        farthest = torch.randint(N, (B,), device=device)

    # distance[b, i]：第 b 个样本的第 i 个点
    # 到目前已选点集合的最短距离平方。
    distance = torch.full(
        (B, N), float("inf"), device=device, dtype=xyz.dtype
    )

    for i in range(npoint):
        centroids[:, i] = farthest

        selected_xyz = xyz[batch_indices, farthest].unsqueeze(1)
        dist = ((xyz - selected_xyz) ** 2).sum(dim=-1)

        distance = torch.minimum(distance, dist)
        farthest = distance.max(dim=1).indices

    return centroids


class DataAugmentation:
    def __init__(
        self,
        npoints=1024,
        scale_low=0.8,
        scale_high=1.0,
        shift_range=0.1,
        normal=False
    ):
        if npoints <= 0:
            raise ValueError("npoints 必须大于 0")
        if not (0 < scale_low <= scale_high <= 1.0):
            raise ValueError(
                "为了保证增强后仍在单位球内，需要满足 "
                "0 < scale_low <= scale_high <= 1"
            )

        self.npoints = npoints
        self.scale_low = scale_low
        self.scale_high = scale_high
        self.shift_range = shift_range
        self.normal = normal

        self.jitter_sigma = 0.01
        self.jitter_clip = 0.05

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __call__(self, points):
        """
        points: [B, N, 3]；normal=True 时为 [B, N, 6]，
                后三列依次是 nx、ny、nz。
        """
        points = points.to(self.device)

        if points.ndim != 3:
            raise ValueError("points 应为 [B, N, C]")
        if not points.is_floating_point():
            raise TypeError("points 必须是浮点张量")
        if points.shape[1] == 0:
            raise ValueError("点云不能为空")
        if self.normal and points.shape[-1] < 6:
            raise ValueError("normal=True 时，points 至少需要 6 个通道")
        if points.shape[-1] < 3:
            raise ValueError("points 至少需要 XYZ 三个通道")

        points = self.resample(points)
        points = self.normalize(points)
        # points = self.random_rotation(points)
        points = self.random_translate(points)
        points = self.random_scale(points)
        # points = self.random_scale_and_translate(points)

        '''
        points[..., :3] = points[..., :3].clamp(
            -0.999,
            0.999
        )
        '''

        return points

    def resample(self, points):
        """先 FPS 到 point_all，再随机选取 npoints 个点。"""
        B, N, C = points.shape
        npoints = self.npoints

        if N == npoints:
            return points

        if N < npoints:
            # 点数不足时重复随机点，以保证输出始终是 npoints 个点。
            extra_idx = torch.randint(
                N, (B, npoints - N), device=points.device
            )
            extra = torch.gather(
                points,
                dim=1,
                index=extra_idx.unsqueeze(-1).expand(-1, -1, C),
            )
            return torch.cat((points, extra), dim=1)

        point_all_map = {
            1024: 1200,
            2048: 2300,
            4096: 4800,
            8192: 8192,
        }
        if npoints not in point_all_map:
            raise NotImplementedError(
                f"尚未定义 npoints={npoints} 的 point_all"
            )

        point_all = min(point_all_map[npoints], N)

        fps_idx = farthest_point_sample(
            points[..., :3].contiguous(),
            point_all,
        )

        # 整个 batch 共用同一组 FPS 结果的“位置序号”；
        # fps_idx 中每个样本对应的原始点索引仍然不同。
        random_idx = torch.randperm(
            point_all, device=points.device
        )[:npoints]
        selected_idx = fps_idx[:, random_idx]

        return torch.gather(
            points,
            dim=1,
            index=selected_idx.unsqueeze(-1).expand(-1, -1, C),
        )

    def normalize(self, points):
        """仅对 XYZ 做中心化和单位球归一化。"""
        points = points.clone()

        xyz = points[..., :3]
        centroid = xyz.mean(dim=1, keepdim=True)
        xyz = xyz - centroid

        radius = torch.linalg.vector_norm(
            xyz, dim=-1
        ).amax(dim=1, keepdim=True).unsqueeze(-1)

        # 所有点重合时，避免除以 0。
        xyz = xyz / radius.clamp_min(torch.finfo(xyz.dtype).eps)

        points[..., :3] = xyz
        return points

    def random_rotation(self, points):
        """每个样本独立绕 Z 轴旋转；存在法向量时同步旋转。"""
        points = points.clone()

        B = points.shape[0]
        theta = (
            torch.rand(
                B, device=points.device, dtype=points.dtype
            ) * 2.0 - 1.0
        ) * math.pi

        cos_theta = torch.cos(theta).view(B, 1)
        sin_theta = torch.sin(theta).view(B, 1)

        def rotate_z(vectors):
            x = vectors[..., 0]
            y = vectors[..., 1]
            z = vectors[..., 2]

            return torch.stack(
                (
                    x * cos_theta - y * sin_theta,
                    x * sin_theta + y * cos_theta,
                    z,
                ),
                dim=-1,
            )

        points[..., :3] = rotate_z(points[..., :3])

        if self.normal:
            points[..., 3:6] = rotate_z(points[..., 3:6])

        return points

    def random_translate(self, points):
        points = points.clone()

        xyz = points[..., :3]

        B, N, _ = xyz.shape
        device = xyz.device
        dtype = xyz.dtype
        # ============================================================
        # 3. Random translation
        #
        # tx, ty, tz ~ U(-0.1, 0.1)
        # ============================================================

        translation = (
                              torch.rand(
                                  B, 1, 3,
                                  device=device,
                                  dtype=dtype
                              ) * 2.0 - 1.0
                      ) * self.shift_range

        xyz = xyz + translation

        # 写回 xyz
        points[..., :3] = xyz

        return points

    def random_scale(self, points):
        points = points.clone()

        xyz = points[..., :3]

        B, N, _ = xyz.shape
        device = xyz.device
        dtype = xyz.dtype

        # ============================================================
        # 2. Anisotropic random scaling
        #
        # PointNeXt:
        # anisotropic=True
        #
        # sx, sy, sz independently sampled from [0.8, 1.25]
        # ============================================================
        dist = torch.norm(xyz, dim=2, keepdim=True)
        maxdist, _ = torch.max(dist, dim=1, keepdim=True)

        xyz = xyz / maxdist

        # norm_scale = 1 / maxdist

        scale = torch.empty(
            B, 1, 1,
            device=device,
            dtype=dtype
        ).uniform_(
            self.scale_low,
            self.scale_high
        )

        xyz = xyz * scale

        # 写回 xyz
        points[..., :3] = xyz

        return points

    def random_scale_and_translate(self, points):
        """
        每个样本的 XYZ 同比缩放，再随机平移。

        归一化和旋转后，所有点满足 ||xyz|| <= 1。
        若缩放系数为 s，则令平移向量满足 ||t|| <= 1-s，
        从而保证 ||s*xyz + t|| <= 1。
        """
        points = points.clone()

        B = points.shape[0]
        scale = torch.empty(
            B, 1, 1,
            device=points.device,
            dtype=points.dtype,
        ).uniform_(self.scale_low, self.scale_high)

        # 每个轴独立采样，但三个分量合成后的平移向量长度
        # 不超过 1-scale。无需 shift_range 参数。
        translation_limit = (1.0 - scale) / math.sqrt(3.0)

        translation = (
            torch.rand(
                B, 1, 3,
                device=points.device,
                dtype=points.dtype,
            ) * 2.0 - 1.0
        ) * translation_limit

        points[..., :3] = points[..., :3] * scale + translation

        # 正的同比缩放和平移都不改变法向量方向。
        return points

    def random_jitter(self, points):
        """备用函数：只扰动 XYZ，不在 __call__ 中调用。"""
        points = points.clone()

        noise = (
            torch.randn_like(points[..., :3]) * self.jitter_sigma
        ).clamp(-self.jitter_clip, self.jitter_clip)

        points[..., :3] = points[..., :3] + noise
        return points

    def fps_sample_and_normalize(self, points):
        """
        points: [B, N, 3]；normal=True 时为 [B, N, 6]，
                后三列依次是 nx、ny、nz。
        """
        points = points.to(self.device)

        if points.ndim != 3:
            raise ValueError("points 应为 [B, N, C]")
        if not points.is_floating_point():
            raise TypeError("points 必须是浮点张量")
        if points.shape[1] == 0:
            raise ValueError("点云不能为空")
        if self.normal and points.shape[-1] < 6:
            raise ValueError("normal=True 时，points 至少需要 6 个通道")
        if points.shape[-1] < 3:
            raise ValueError("points 至少需要 XYZ 三个通道")

        points = self.fps_sample(points)
        points = self.normalize(points)

        return points

    def fps_sample(self, points):
        """先 FPS 到 point_all，再随机选取 npoints 个点。"""
        B, N, C = points.shape
        npoints = self.npoints

        if N == npoints:
            return points

        if N < npoints:
            # 点数不足时重复随机点，以保证输出始终是 npoints 个点。
            extra_idx = torch.randint(
                N, (B, npoints - N), device=points.device
            )
            extra = torch.gather(
                points,
                dim=1,
                index=extra_idx.unsqueeze(-1).expand(-1, -1, C),
            )
            return torch.cat((points, extra), dim=1)

        fps_idx = farthest_point_sample(
            points[..., :3].contiguous(),
            npoints,
            start_zero=True
        )

        fps_pts = torch.gather(
            points,
            dim=1,
            index=fps_idx.unsqueeze(-1).expand(-1, -1, C),
        )

        # fps_pts = self.normalize(fps_pts)

        return fps_pts
