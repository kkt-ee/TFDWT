import os
import sys

import numpy as np

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if PKG_ROOT not in sys.path:
    sys.path.insert(0, PKG_ROOT)

import tensorflow as tf

from TFDWT.DWT1DFB import DWT1D, IDWT1D
from TFDWT.DWT2DFB import DWT2D, IDWT2D
from TFDWT.DWT3DFB import DWT3D, IDWT3D
from TFDWT.multilevel.dwt import dwt as dwt1d, idwt as idwt1d
from TFDWT.multilevel.dwt2 import dwt2, idwt2
from TFDWT.multilevel.dwt3 import dwt3, idwt3


def _normalize01(x: np.ndarray) -> np.ndarray:
    x_min = float(np.min(x))
    x_max = float(np.max(x))
    if x_max > x_min:
        x = (x - x_min) / (x_max - x_min)
    else:
        x = np.zeros_like(x)
    return x.astype(np.float32, copy=False)


def _make_signal_1d(N: int = 64) -> np.ndarray:
    t = np.linspace(0.0, 1.0, N, dtype=np.float32)
    x = (
        0.50 * np.sin(2.0 * np.pi * 3.0 * t)
        + 0.25 * np.cos(2.0 * np.pi * 7.0 * t)
        + 0.15 * (t > 0.45).astype(np.float32)
        + 0.30 * np.exp(-0.5 * ((t - 0.28) / 0.06) ** 2)
    )
    return _normalize01(x)[None, :, None]


def _make_image_2d(N: int = 32) -> np.ndarray:
    yy, xx = np.meshgrid(
        np.linspace(-1.0, 1.0, N, dtype=np.float32),
        np.linspace(-1.0, 1.0, N, dtype=np.float32),
        indexing="ij",
    )
    img = (
        0.45 * np.sin(3.0 * np.pi * xx)
        + 0.30 * np.cos(4.0 * np.pi * yy)
        + 0.20 * ((xx + yy) > 0.10).astype(np.float32)
        + 0.35 * np.exp(-((xx - 0.15) ** 2 + (yy + 0.20) ** 2) / 0.12)
    )
    return _normalize01(img)[None, :, :, None]


def _make_volume_3d(N: int = 16) -> np.ndarray:
    zz, yy, xx = np.meshgrid(
        np.linspace(-1.0, 1.0, N, dtype=np.float32),
        np.linspace(-1.0, 1.0, N, dtype=np.float32),
        np.linspace(-1.0, 1.0, N, dtype=np.float32),
        indexing="ij",
    )
    vol = (
        0.35 * np.sin(2.0 * np.pi * xx)
        + 0.25 * np.cos(2.0 * np.pi * yy)
        + 0.20 * np.sin(3.0 * np.pi * zz)
        + 0.15 * ((xx + yy + zz) > 0.20).astype(np.float32)
        + 0.30 * np.exp(-((xx - 0.20) ** 2 + (yy + 0.10) ** 2 + (zz - 0.15) ** 2) / 0.18)
    )
    return _normalize01(vol)[None, :, :, :, None]


def _assert_pr(x: tf.Tensor, xhat: tf.Tensor, tol: float) -> None:
    max_err = float(tf.reduce_max(tf.math.abs(x - xhat)))
    assert max_err < tol, f"Perfect reconstruction failed: max_err={max_err:.3e}, tol={tol:.1e}"


def test_single_level_pr_1d_clean_modes():
    x = tf.convert_to_tensor(_make_signal_1d(64), dtype=tf.float32)

    for clean, expected_shape in ((True, (1, 32, 2)), (False, (1, 64, 1))):
        subbands = DWT1D(wave="haar", clean=clean)(x)
        assert tuple(int(d) for d in subbands.shape) == expected_shape
        xhat = IDWT1D(wave="haar", clean=clean)(subbands)
        _assert_pr(x, xhat, tol=1e-6)


def test_single_level_pr_2d_clean_modes():
    x = tf.convert_to_tensor(_make_image_2d(32), dtype=tf.float32)

    for clean, expected_shape in ((True, (1, 16, 16, 4)), (False, (1, 32, 32, 1))):
        subbands = DWT2D(wave="haar", clean=clean)(x)
        assert tuple(int(d) for d in subbands.shape) == expected_shape
        xhat = IDWT2D(wave="haar", clean=clean)(subbands)
        _assert_pr(x, xhat, tol=1e-5)


def test_single_level_pr_3d_clean_modes():
    x = tf.convert_to_tensor(_make_volume_3d(16), dtype=tf.float32)

    for clean, expected_shape in ((True, (1, 8, 8, 8, 8)), (False, (1, 16, 16, 16, 1))):
        subbands = DWT3D(wave="haar", clean=clean)(x)
        assert tuple(int(d) for d in subbands.shape) == expected_shape
        xhat = IDWT3D(wave="haar", clean=clean)(subbands)
        _assert_pr(x, xhat, tol=1e-5)


def test_multilevel_pr_1d():
    level = 3
    x = tf.convert_to_tensor(_make_signal_1d(64), dtype=tf.float32)
    subbands = dwt1d(x, level=level, Ψ="haar")

    assert len(subbands) == level + 1
    expected_shapes = ((1, 32, 1), (1, 16, 1), (1, 8, 1), (1, 8, 1))
    assert tuple(tuple(int(d) for d in sb.shape) for sb in subbands) == expected_shapes

    xhat = idwt1d(subbands, level=level, Ψ="haar")
    _assert_pr(x, xhat, tol=1e-6)


def test_multilevel_pr_2d():
    level = 3
    x = tf.convert_to_tensor(_make_image_2d(32), dtype=tf.float32)
    subbands = dwt2(x, level=level, Ψ="haar")

    assert len(subbands) == level + 1
    expected_shapes = ((1, 16, 16, 3), (1, 8, 8, 3), (1, 4, 4, 3), (1, 4, 4, 1))
    assert tuple(tuple(int(d) for d in sb.shape) for sb in subbands) == expected_shapes

    xhat = idwt2(subbands, level=level, Ψ="haar")
    _assert_pr(x, xhat, tol=1e-5)


def test_multilevel_pr_3d():
    level = 3
    x = tf.convert_to_tensor(_make_volume_3d(16), dtype=tf.float32)
    subbands = dwt3(x, level=level, Ψ="haar")

    assert len(subbands) == level + 1
    expected_shapes = ((1, 8, 8, 8, 7), (1, 4, 4, 4, 7), (1, 2, 2, 2, 7), (1, 2, 2, 2, 1))
    assert tuple(tuple(int(d) for d in sb.shape) for sb in subbands) == expected_shapes

    xhat = idwt3(subbands, level=level, Ψ="haar")
    _assert_pr(x, xhat, tol=1e-5)
