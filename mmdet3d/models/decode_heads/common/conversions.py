import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor


def accumulated_depth_to_expected_depth(
    accumulated_depth: Tensor,
    alpha: Tensor,
    depth_unit_scale=1.0,
    alpha_threshold: float = 1e-7,
):
    """Convert alpha-accumulated rasterizer depth to expected metric depth.

    The CUDA rasterizer returns ``sum(T_i * alpha_i * z_i)``. Dividing by the
    accumulated alpha makes the result independent of foreground opacity. The
    optional scale restores metric units when the scene was normalized by the
    near plane before rasterization.
    """
    if alpha_threshold <= 0:
        raise ValueError("alpha_threshold must be positive")

    unit_scale = torch.as_tensor(
        depth_unit_scale,
        dtype=accumulated_depth.dtype,
        device=accumulated_depth.device,
    )
    while unit_scale.ndim < accumulated_depth.ndim:
        unit_scale = unit_scale.unsqueeze(-1)

    valid = (
        torch.isfinite(accumulated_depth)
        & torch.isfinite(alpha)
        & (alpha >= alpha_threshold)
    )
    expected_depth = accumulated_depth / alpha.clamp_min(alpha_threshold)
    expected_depth = expected_depth * unit_scale
    valid = valid & torch.isfinite(expected_depth) & (expected_depth >= 0)
    return torch.where(valid, expected_depth, torch.zeros_like(expected_depth))


def resize_depth_with_alpha(
    depth: Tensor,
    alpha: Tensor,
    size,
    alpha_threshold: float = 1e-3,
):
    """Resize expected depth without bleeding zero-background into edges."""
    if alpha_threshold <= 0:
        raise ValueError("alpha_threshold must be positive")
    if depth.shape != alpha.shape:
        raise ValueError(
            f"depth and alpha must have the same shape, got "
            f"{tuple(depth.shape)} and {tuple(alpha.shape)}")

    weighted_depth = F.interpolate(
        depth * alpha, size=size, mode="bilinear", align_corners=False)
    resized_alpha = F.interpolate(
        alpha, size=size, mode="bilinear", align_corners=False)
    resized_depth = weighted_depth / resized_alpha.clamp_min(1e-7)
    valid = (
        torch.isfinite(resized_depth)
        & torch.isfinite(resized_alpha)
        & (resized_alpha >= alpha_threshold)
        & (resized_depth > 0)
    )
    resized_depth = torch.where(
        valid, resized_depth, torch.zeros_like(resized_depth))
    return resized_depth, resized_alpha, valid


def relative_disparity_to_depth(
    relative_disparity: Float[Tensor, "*#batch"],
    near: Float[Tensor, "*#batch"],
    far: Float[Tensor, "*#batch"],
    eps: float = 1e-10,
) -> Float[Tensor, " *batch"]:
    """Convert relative disparity, where 0 is near and 1 is far, to depth."""
    disp_near = 1 / (near + eps)
    disp_far = 1 / (far + eps)
    return 1 / ((1 - relative_disparity) * (disp_near - disp_far) + disp_far + eps)


def depth_to_relative_disparity(
    depth: Float[Tensor, "*#batch"],
    near: Float[Tensor, "*#batch"],
    far: Float[Tensor, "*#batch"],
    eps: float = 1e-10,
) -> Float[Tensor, " *batch"]:
    """Convert depth to relative disparity, where 0 is near and 1 is far"""
    disp_near = 1 / (near + eps)
    disp_far = 1 / (far + eps)
    disp = 1 / (depth + eps)
    return 1 - (disp - disp_far) / (disp_near - disp_far + eps)
