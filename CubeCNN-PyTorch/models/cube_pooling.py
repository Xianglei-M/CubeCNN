"""Aggregate points sharing a child voxel without overwriting features."""

import operator

import torch


def pool_chicube_fts(features, parent_row, local_idx, num_cubes, split,
                     reduction="mean"):
    """Return (features [M,S,S,S,C], counts [M,S,S,S]).

    Inputs: floating features [N,C], integer parent_row [N], and integer
    local_idx [N,3]. All tensors must be on the same device. Empty cells are
    zero. Accumulation uses float32, or float64 for float64 input. Counts are
    int64. Supported reductions: mean, sum, max (independently per channel).
    Range checks intentionally synchronize CUDA to reject invalid indices
    before scatter. CUDA floating reductions may differ in their last bits
    when input order changes; mathematical permutation invariance does not
    imply bitwise reproducibility.
    """
    num_cubes, split = operator.index(num_cubes), operator.index(split)
    if num_cubes < 0 or split < 1:
        raise ValueError("num_cubes must be >= 0 and split must be >= 1")
    if reduction not in {"mean", "sum", "max"}:
        raise ValueError("reduction must be mean, sum or max")
    if features.ndim != 2 or not features.is_floating_point():
        raise ValueError("features must be a floating tensor [N,C]")
    n, channels = features.shape
    if parent_row.shape != (n,) or local_idx.shape != (n, 3):
        raise ValueError("indices must have shapes [N] and [N,3]")
    if any(t.dtype not in (torch.int32, torch.int64)
           for t in (parent_row, local_idx)):
        raise TypeError("indices must be int32 or int64")
    if any(t.device != features.device for t in (parent_row, local_idx)):
        raise ValueError("features and indices must be on the same device")
    if ((parent_row < 0) | (parent_row >= num_cubes)).any().item():
        raise ValueError("parent_row is outside [0, num_cubes)")
    if ((local_idx < 0) | (local_idx >= split)).any().item():
        raise ValueError("local_idx is outside [0, split)")

    parent_row, local_idx = parent_row.long(), local_idx.long()
    key = (((parent_row * split + local_idx[:, 0]) * split
            + local_idx[:, 1]) * split + local_idx[:, 2])
    cells = num_cubes * split ** 3
    dtype = torch.float64 if features.dtype == torch.float64 else torch.float32
    source = features.to(dtype)
    counts = torch.zeros(cells, dtype=torch.long, device=features.device)
    counts.index_add_(0, key, torch.ones_like(key))

    if reduction == "max":
        # Zero initialization would incorrectly clip all-negative features.
        pooled = source.new_full((cells, channels), -float("inf"))
        pooled.scatter_reduce_(0, key[:, None].expand(-1, channels), source,
                               reduce="amax", include_self=True)
        pooled = torch.where(counts[:, None] > 0, pooled,
                             torch.zeros_like(pooled))
    else:
        pooled = source.new_zeros((cells, channels))
        pooled.index_add_(0, key, source)
        if reduction == "mean":
            pooled = pooled / counts.clamp_min(1).to(dtype)[:, None]

    shape = (num_cubes, split, split, split)
    return pooled.reshape(*shape, channels), counts.reshape(shape)
