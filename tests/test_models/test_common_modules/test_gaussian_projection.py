import importlib.util
from pathlib import Path

import pytest
import torch


PROJECTION_PATH = (
    Path(__file__).resolve().parents[3]
    / 'mmdet3d/models/decode_heads/common/projection.py'
)
SPEC = importlib.util.spec_from_file_location(
    'visionpad_projection_for_test', PROJECTION_PATH)
PROJECTION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROJECTION)

get_projection_matrix_from_intrinsics = (
    PROJECTION.get_projection_matrix_from_intrinsics)
get_tanfov_from_intrinsics = PROJECTION.get_tanfov_from_intrinsics


def _make_intrinsics(fx, fy, cx, cy, dtype=torch.float64):
    return torch.tensor([[
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0],
    ]], dtype=dtype)


def _project_with_matrix(points, projection):
    points_h = torch.cat(
        [points, torch.ones_like(points[..., :1])], dim=-1)
    # This is the row-vector equivalent of the transposed matrix passed to
    # diff_gauss by cuda_splatting.py.
    clip = points_h @ projection.transpose(-1, -2)
    ndc = clip[..., :2] / clip[..., 3:4]
    return (ndc + 1.0) / 2.0


def test_centered_intrinsics_matches_legacy_projection():
    intrinsics = _make_intrinsics(0.8, 1.2, 0.5, 0.5)
    near = torch.tensor([0.1], dtype=intrinsics.dtype)
    far = torch.tensor([60.0], dtype=intrinsics.dtype)

    projection = get_projection_matrix_from_intrinsics(
        intrinsics, near, far)
    expected = torch.zeros_like(projection)
    expected[:, 0, 0] = 1.6
    expected[:, 1, 1] = 2.4
    expected[:, 3, 2] = 1.0
    expected[:, 2, 2] = far / (far - near)
    expected[:, 2, 3] = -(far * near) / (far - near)

    torch.testing.assert_close(projection, expected)


def test_off_center_optical_axis_projects_to_principal_point():
    width, height = 640.0, 360.0
    intrinsics = _make_intrinsics(
        509.04 / width,
        787.42 / height,
        330.65 / width,
        99.97 / height,
    )
    near = torch.tensor([0.1], dtype=intrinsics.dtype)
    far = torch.tensor([60.0], dtype=intrinsics.dtype)
    projection = get_projection_matrix_from_intrinsics(
        intrinsics, near, far)

    point_on_optical_axis = torch.tensor(
        [[[0.0, 0.0, 10.0]]], dtype=intrinsics.dtype)
    uv = _project_with_matrix(point_on_optical_axis, projection)
    expected = intrinsics[:, None, :2, 2]
    torch.testing.assert_close(uv, expected)


@pytest.mark.parametrize('principal_point', [(0.5, 0.5), (0.517, 0.278)])
def test_projection_matches_direct_intrinsics(principal_point):
    torch.manual_seed(0)
    cx, cy = principal_point
    intrinsics = _make_intrinsics(0.79, 2.18, cx, cy)
    near = torch.tensor([0.1], dtype=intrinsics.dtype)
    far = torch.tensor([60.0], dtype=intrinsics.dtype)
    projection = get_projection_matrix_from_intrinsics(
        intrinsics, near, far)

    points = torch.empty((1, 128, 3), dtype=intrinsics.dtype)
    points[..., 2].uniform_(2.0, 50.0)
    points[..., 0].uniform_(-0.5, 0.5).mul_(points[..., 2])
    points[..., 1].uniform_(-0.1, 0.1).mul_(points[..., 2])

    uv_projection = _project_with_matrix(points, projection)
    image_homogeneous = points @ intrinsics.transpose(-1, -2)
    uv_intrinsics = (
        image_homogeneous[..., :2] / image_homogeneous[..., 2:3])
    torch.testing.assert_close(uv_projection, uv_intrinsics)


def test_tanfov_recovers_focal_and_ignores_principal_point():
    width, height = 640.0, 360.0
    fx, fy = 509.04, 787.42
    centered = _make_intrinsics(
        fx / width, fy / height, 0.5, 0.5)
    off_center = _make_intrinsics(
        fx / width, fy / height, 330.65 / width, 99.97 / height)

    centered_tan = get_tanfov_from_intrinsics(centered)
    off_center_tan = get_tanfov_from_intrinsics(off_center)
    torch.testing.assert_close(centered_tan, off_center_tan)
    assert centered_tan[0, 0].item() == pytest.approx(width / (2.0 * fx))
    assert centered_tan[0, 1].item() == pytest.approx(height / (2.0 * fy))
    assert width / (2.0 * centered_tan[0, 0]).item() == pytest.approx(fx)
    assert height / (2.0 * centered_tan[0, 1]).item() == pytest.approx(fy)
