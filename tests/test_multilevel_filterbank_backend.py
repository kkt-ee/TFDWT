import math
import os
import sys

import numpy as np
import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if PKG_ROOT not in sys.path:
    sys.path.insert(0, PKG_ROOT)

import tensorflow as tf

from TFDWT.DWT1DFB import DWT1D, IDWT1D
from TFDWT.dbFBimpulseResponse import FBimpulseResponses
from TFDWT.multilevel.dwt import (
    dwt,
    dwt_packed_axis,
    idwt,
    idwt_packed_axis,
)


def _assert_close(actual, expected, tolerance=1e-5):
    np.testing.assert_allclose(
        actual.numpy(),
        expected.numpy(),
        rtol=tolerance,
        atol=tolerance,
    )


@pytest.mark.parametrize("wave", ["haar", "db2", "bior2.2", "rbio2.2"])
@pytest.mark.parametrize("level", [1, 2, 3])
def test_list_api_filterbank_matches_matrix_and_reconstructs(wave, level):
    tf.random.set_seed(401 + level)
    x = tf.random.normal((2, 128, 3))
    matrix = dwt(x, level=level, Ψ=wave, backend="matrix")
    filterbank = dwt(x, level=level, Ψ=wave, backend="filterbank")

    assert len(matrix) == level + 1
    for matrix_subband, filterbank_subband in zip(matrix, filterbank):
        _assert_close(filterbank_subband, matrix_subband)

    matrix_reconstruction = idwt(
        matrix,
        level=level,
        Ψ=wave,
        backend="matrix",
    )
    filterbank_reconstruction = idwt(
        filterbank,
        level=level,
        Ψ=wave,
        backend="filterbank",
    )
    _assert_close(filterbank_reconstruction, matrix_reconstruction)
    _assert_close(filterbank_reconstruction, x)


@pytest.mark.parametrize(
    "shape,axis",
    [
        ((32, 2, 3), 0),
        ((2, 32, 3), 1),
        ((2, 3, 32), 2),
        ((2, 3, 32), -1),
    ],
)
@pytest.mark.parametrize("wave", ["haar", "bior2.2"])
def test_packed_arbitrary_axis_matches_matrix_and_reconstructs(
    shape, axis, wave
):
    tf.random.set_seed(405)
    x = tf.random.normal(shape)
    matrix = dwt_packed_axis(
        x,
        level=3,
        wave=wave,
        axis=axis,
        backend="matrix",
    )
    filterbank = dwt_packed_axis(
        x,
        level=3,
        wave=wave,
        axis=axis,
        backend="filterbank",
    )
    _assert_close(filterbank, matrix)
    _assert_close(
        idwt_packed_axis(
            filterbank,
            level=3,
            wave=wave,
            axis=axis,
            backend="filterbank",
        ),
        x,
    )
    _assert_close(
        idwt_packed_axis(
            matrix,
            level=3,
            wave=wave,
            axis=axis,
            backend="matrix",
        ),
        x,
    )


@pytest.mark.parametrize("backend", ["matrix", "filterbank"])
def test_packed_order_matches_existing_list_api(backend):
    tf.random.set_seed(406)
    x = tf.random.normal((2, 64, 3))
    subbands = dwt(x, level=3, Ψ="bior2.2", backend=backend)
    expected = tf.concat(
        [subbands[-1]] + list(reversed(subbands[:-1])),
        axis=1,
    )
    packed = dwt_packed_axis(
        x,
        level=3,
        wave="bior2.2",
        axis=1,
        backend=backend,
    )
    _assert_close(packed, expected, tolerance=0.0)
    _assert_close(
        idwt_packed_axis(
            packed,
            level=3,
            wave="bior2.2",
            axis=1,
            backend=backend,
        ),
        idwt(subbands, level=3, Ψ="bior2.2", backend=backend),
        tolerance=0.0,
    )


@pytest.mark.parametrize("backend", ["matrix", "filterbank"])
def test_level_one_packed_is_existing_raw_single_level_transform(backend):
    tf.random.set_seed(407)
    x = tf.random.normal((2, 32, 3))
    packed = dwt_packed_axis(
        x,
        level=1,
        wave="bior2.2",
        axis=1,
        backend=backend,
    )
    expected = DWT1D(
        wave="bior2.2",
        clean=False,
        backend=backend,
    )(x)
    _assert_close(packed, expected, tolerance=0.0)
    _assert_close(
        idwt_packed_axis(
            packed,
            level=1,
            wave="bior2.2",
            axis=1,
            backend=backend,
        ),
        IDWT1D(
            wave="bior2.2",
            clean=False,
            backend=backend,
        )(packed),
        tolerance=0.0,
    )


def test_packed_filterbank_gradients_match_matrix():
    tf.random.set_seed(408)
    x = tf.random.normal((2, 32, 3))
    probe = tf.random.normal(x.shape)

    def gradient(backend):
        variable = tf.Variable(x)
        with tf.GradientTape() as tape:
            coefficients = dwt_packed_axis(
                variable,
                level=3,
                wave="bior2.2",
                axis=1,
                backend=backend,
            )
            reconstruction = idwt_packed_axis(
                coefficients,
                level=3,
                wave="bior2.2",
                axis=1,
                backend=backend,
            )
            loss = tf.reduce_sum(reconstruction * probe)
        return tape.gradient(loss, variable)

    _assert_close(gradient("filterbank"), gradient("matrix"))


def test_level_two_filterbank_matches_matrix_for_every_wavelet():
    tf.random.set_seed(409)
    for wave, banks in FBimpulseResponses.items():
        largest_filter = max(len(filt) for bank in banks for filt in bank)
        minimum = 2 * largest_filter
        length = max(16, 2 ** math.ceil(math.log2(minimum)))
        x = tf.random.normal((1, length, 1))
        matrix = dwt_packed_axis(
            x,
            level=2,
            wave=wave,
            axis=1,
            backend="matrix",
        )
        filterbank = dwt_packed_axis(
            x,
            level=2,
            wave=wave,
            axis=1,
            backend="filterbank",
        )
        _assert_close(filterbank, matrix, tolerance=3e-5)
        _assert_close(
            idwt_packed_axis(
                filterbank,
                level=2,
                wave=wave,
                axis=1,
                backend="filterbank",
            ),
            x,
            tolerance=3e-5,
        )


def test_packed_multilevel_runs_in_tensorflow_graph():
    @tf.function
    def round_trip(x):
        coefficients = dwt_packed_axis(
            x,
            level=3,
            wave="bior2.2",
            axis=1,
            backend="filterbank",
        )
        return idwt_packed_axis(
            coefficients,
            level=3,
            wave="bior2.2",
            axis=1,
            backend="filterbank",
        )

    x = tf.random.normal((2, 32, 3))
    _assert_close(round_trip(x), x)


def test_multilevel_validation():
    x = tf.zeros((1, 32, 1))
    with pytest.raises(ValueError, match="level"):
        dwt_packed_axis(x, level=-1)
    with pytest.raises(ValueError, match="divisible"):
        dwt_packed_axis(tf.zeros((1, 30, 1)), level=2)
    with pytest.raises(ValueError, match="filter length"):
        dwt_packed_axis(x, level=3, wave="db10")
    with pytest.raises(ValueError, match="backend"):
        dwt(x, backend="unknown")
    subbands = dwt(x, level=2)
    with pytest.raises(
        ValueError,
        match="level=1 requires 2 subbands, but received 3",
    ):
        idwt(subbands, level=1)
