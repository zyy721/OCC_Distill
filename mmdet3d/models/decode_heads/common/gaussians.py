import math

import torch
from einops import rearrange
from jaxtyping import Float
from torch import Tensor, nn


# https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py
def quaternion_to_matrix(
    quaternions,# : Float[Tensor, "*batch 4"],
    eps: float = 1e-8,
):
    # Order changed to match scipy format!
    i, j, k, r = torch.unbind(quaternions, dim=-1)
    two_s = 2 / ((quaternions * quaternions).sum(dim=-1) + eps)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return rearrange(o, "... (i j) -> ... i j", i=3, j=3)


def build_covariance(
    scale, # : Float[Tensor, "*#batch 3"],
    rotation_xyzw, # : Float[Tensor, "*#batch 4"],
):
    scale = scale.diag_embed()
    rotation = quaternion_to_matrix(rotation_xyzw)
    return (
        rotation
        @ scale
        @ rearrange(scale, "... i j -> ... j i")
        @ rearrange(rotation, "... i j -> ... j i")
    )


def broadcast_covariances_to_views(covariances: Tensor, num_views: int):
    """Broadcast world/key-ego covariances across camera views.

    The rasterizer receives world-space covariance together with a camera view
    matrix and performs the world-to-camera covariance transform internally.
    """
    if covariances.ndim != 4 or covariances.shape[-2:] != (3, 3):
        raise ValueError(
            "covariances must have shape (batch, gaussian, 3, 3), got "
            f"{tuple(covariances.shape)}")
    if num_views <= 0:
        raise ValueError("num_views must be positive")
    return covariances[:, None].expand(-1, num_views, -1, -1, -1)


def initialize_density_head(
    head: nn.Module,
    initial_probability: float = 0.01,
    weight_std: float = 1e-3,
):
    """Initialize the last density layer to a sparse sigmoid opacity prior."""
    if not 0 < initial_probability < 1:
        raise ValueError("initial_probability must be strictly between 0 and 1")
    if weight_std < 0:
        raise ValueError("weight_std must be non-negative")

    linear_layers = [module for module in head.modules()
                     if isinstance(module, nn.Linear)]
    if not linear_layers:
        raise ValueError("density head must contain at least one nn.Linear layer")
    final_layer = linear_layers[-1]
    if final_layer.bias is None:
        raise ValueError("the final density layer must have a bias")

    nn.init.normal_(final_layer.weight, mean=0.0, std=weight_std)
    initial_logit = math.log(initial_probability / (1 - initial_probability))
    nn.init.constant_(final_layer.bias, initial_logit)
