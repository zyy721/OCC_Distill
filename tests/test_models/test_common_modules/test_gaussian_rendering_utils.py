import importlib.util
from pathlib import Path

import pytest
import torch
from torch import nn


COMMON_PATH = (
    Path(__file__).resolve().parents[3]
    / 'mmdet3d/models/decode_heads/common'
)


def _load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, COMMON_PATH / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CONVERSIONS = _load_module(
    'visionpad_conversions_for_test', 'conversions.py')
GAUSSIANS = _load_module('visionpad_gaussians_for_test', 'gaussians.py')

DEPTH_SSL_PATH = (
    Path(__file__).resolve().parents[3]
    / 'mmdet3d/models/detectors/depth_ssl.py'
)
DEPTH_SSL_SPEC = importlib.util.spec_from_file_location(
    'visionpad_depth_ssl_for_test', DEPTH_SSL_PATH)
DEPTH_SSL = importlib.util.module_from_spec(DEPTH_SSL_SPEC)
DEPTH_SSL_SPEC.loader.exec_module(DEPTH_SSL)


def test_accumulated_depth_is_normalized_by_alpha_and_differentiable():
    accumulated_depth = torch.tensor(
        [[[[2.0, 0.0, 3.0]]]], requires_grad=True)
    alpha = torch.tensor(
        [[[[0.5, 0.0, 0.75]]]], requires_grad=True)

    depth = CONVERSIONS.accumulated_depth_to_expected_depth(
        accumulated_depth, alpha)

    torch.testing.assert_close(
        depth, torch.tensor([[[[4.0, 0.0, 4.0]]]]))
    depth.sum().backward()
    assert torch.isfinite(accumulated_depth.grad).all()
    assert torch.isfinite(alpha.grad).all()


def test_accumulated_depth_restores_pre_rasterization_scale():
    depth = CONVERSIONS.accumulated_depth_to_expected_depth(
        torch.tensor([[[[1.5]]]]),
        torch.tensor([[[[0.5]]]]),
        depth_unit_scale=torch.tensor(0.1),
    )
    torch.testing.assert_close(depth, torch.tensor([[[[0.3]]]]))


def test_depth_resize_is_alpha_weighted_at_foreground_boundary():
    depth = torch.tensor([[[[10.0, 0.0]]]])
    alpha = torch.tensor([[[[1.0, 0.0]]]])

    resized_depth, resized_alpha, valid = (
        CONVERSIONS.resize_depth_with_alpha(
            depth, alpha, size=(1, 1), alpha_threshold=0.1))

    torch.testing.assert_close(resized_depth, torch.tensor([[[[10.0]]]]))
    torch.testing.assert_close(resized_alpha, torch.tensor([[[[0.5]]]]))
    assert valid.item()


def test_depth_resize_masks_low_alpha_pixels():
    depth = torch.tensor([[[[8.0]]]])
    alpha = torch.tensor([[[[1e-4]]]])
    resized_depth, _, valid = CONVERSIONS.resize_depth_with_alpha(
        depth, alpha, size=(1, 1), alpha_threshold=1e-3)
    assert resized_depth.item() == 0.0
    assert not valid.item()


def test_disparity_smoothness_ignores_invalid_depth_boundaries():
    disparity = torch.tensor([[[[1.0, 100.0, 3.0]]]])
    image = torch.zeros(1, 3, 1, 3)
    valid = torch.tensor([[[[True, False, True]]]])

    smoothness = DEPTH_SSL.get_smooth_loss(disparity, image, valid)

    assert smoothness.item() == 0.0


def test_world_covariance_is_broadcast_without_camera_prerotation():
    covariance = torch.tensor([[
        [[4.0, 0.2, 0.0], [0.2, 1.0, 0.0], [0.0, 0.0, 0.25]],
        [[1.0, 0.0, 0.0], [0.0, 2.0, 0.1], [0.0, 0.1, 3.0]],
    ]])

    per_view = GAUSSIANS.broadcast_covariances_to_views(covariance, 6)

    assert per_view.shape == (1, 6, 2, 3, 3)
    for view_index in range(6):
        torch.testing.assert_close(per_view[:, view_index], covariance)


def test_density_head_starts_from_sparse_sigmoid_prior():
    head = nn.Sequential(
        nn.Linear(4, 8),
        nn.Softplus(),
        nn.Linear(8, 1),
    )
    GAUSSIANS.initialize_density_head(
        head, initial_probability=0.01, weight_std=0.0)

    opacity = torch.sigmoid(head(torch.randn(32, 4)))
    torch.testing.assert_close(opacity, torch.full_like(opacity, 0.01))


@pytest.mark.parametrize('initial_probability', [0.0, 1.0, -0.1, 1.1])
def test_density_head_rejects_invalid_probability(initial_probability):
    with pytest.raises(ValueError):
        GAUSSIANS.initialize_density_head(
            nn.Linear(2, 1), initial_probability=initial_probability)
