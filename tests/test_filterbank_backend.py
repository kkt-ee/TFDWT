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
from TFDWT.DWT2DFB import DWT2D, IDWT2D
from TFDWT.DWT3DFB import DWT3D, IDWT3D
from TFDWT.dbFBimpulseResponse import FBimpulseResponses


def _assert_close(actual, expected, tolerance=2e-6):
    np.testing.assert_allclose(
        actual.numpy(),
        expected.numpy(),
        rtol=tolerance,
        atol=tolerance,
    )


@pytest.mark.parametrize("wave", ["haar", "db2", "bior2.2", "rbio2.2"])
@pytest.mark.parametrize("clean", [False, True])
def test_1d_filterbank_matches_matrix_and_reconstructs(wave, clean):
    tf.random.set_seed(101)
    x = tf.random.normal((2, 32, 3))

    matrix_dwt = DWT1D(wave=wave, clean=clean, backend="matrix")
    filterbank_dwt = DWT1D(wave=wave, clean=clean, backend="filterbank")
    matrix_coefficients = matrix_dwt(x)
    filterbank_coefficients = filterbank_dwt(x)
    _assert_close(filterbank_coefficients, matrix_coefficients)

    matrix_idwt = IDWT1D(wave=wave, clean=clean, backend="matrix")
    filterbank_idwt = IDWT1D(wave=wave, clean=clean, backend="filterbank")
    _assert_close(
        filterbank_idwt(filterbank_coefficients),
        matrix_idwt(matrix_coefficients),
    )
    _assert_close(filterbank_idwt(filterbank_coefficients), x)

    assert filterbank_dwt.weights == []
    assert filterbank_idwt.weights == []


def test_filterbank_matches_matrix_for_every_supported_wavelet():
    tf.random.set_seed(100)
    for wave, banks in FBimpulseResponses.items():
        largest_filter = max(len(filt) for bank in banks for filt in bank)
        length = max(32, 2 ** math.ceil(math.log2(largest_filter)))
        x = tf.random.normal((1, length, 2))

        matrix_coefficients = DWT1D(
            wave=wave, clean=False, backend="matrix"
        )(x)
        filterbank_coefficients = DWT1D(
            wave=wave, clean=False, backend="filterbank"
        )(x)
        _assert_close(filterbank_coefficients, matrix_coefficients, tolerance=3e-6)

        matrix_reconstruction = IDWT1D(
            wave=wave, clean=False, backend="matrix"
        )(matrix_coefficients)
        filterbank_reconstruction = IDWT1D(
            wave=wave, clean=False, backend="filterbank"
        )(filterbank_coefficients)
        _assert_close(
            filterbank_reconstruction,
            matrix_reconstruction,
            tolerance=3e-6,
        )
        _assert_close(filterbank_reconstruction, x, tolerance=3e-6)


@pytest.mark.parametrize("wave", ["haar", "bior2.2"])
@pytest.mark.parametrize("clean", [False, True])
def test_2d_filterbank_matches_matrix_and_reconstructs(wave, clean):
    tf.random.set_seed(102)
    x = tf.random.normal((2, 16, 16, 2))

    matrix_coefficients = DWT2D(
        wave=wave, clean=clean, backend="matrix"
    )(x)
    filterbank_coefficients = DWT2D(
        wave=wave, clean=clean, backend="filterbank"
    )(x)
    _assert_close(filterbank_coefficients, matrix_coefficients, tolerance=5e-6)

    matrix_reconstruction = IDWT2D(
        wave=wave, clean=clean, backend="matrix"
    )(matrix_coefficients)
    filterbank_reconstruction = IDWT2D(
        wave=wave, clean=clean, backend="filterbank"
    )(filterbank_coefficients)
    _assert_close(filterbank_reconstruction, matrix_reconstruction, tolerance=5e-6)
    _assert_close(filterbank_reconstruction, x, tolerance=5e-6)


@pytest.mark.parametrize("wave", ["haar", "bior2.2"])
@pytest.mark.parametrize("clean", [False, True])
def test_3d_filterbank_matches_matrix_and_reconstructs(wave, clean):
    tf.random.set_seed(103)
    x = tf.random.normal((1, 16, 16, 16, 2))

    matrix_coefficients = DWT3D(
        wave=wave, clean=clean, backend="matrix"
    )(x)
    filterbank_coefficients = DWT3D(
        wave=wave, clean=clean, backend="filterbank"
    )(x)
    _assert_close(filterbank_coefficients, matrix_coefficients, tolerance=1e-5)

    matrix_reconstruction = IDWT3D(
        wave=wave, clean=clean, backend="matrix"
    )(matrix_coefficients)
    filterbank_reconstruction = IDWT3D(
        wave=wave, clean=clean, backend="filterbank"
    )(filterbank_coefficients)
    _assert_close(filterbank_reconstruction, matrix_reconstruction, tolerance=1e-5)
    _assert_close(filterbank_reconstruction, x, tolerance=1e-5)


@pytest.mark.parametrize("layer_class", [DWT1D, IDWT1D])
@pytest.mark.parametrize("wave", ["haar", "bior2.2"])
def test_1d_filterbank_input_gradient_matches_matrix(layer_class, wave):
    tf.random.set_seed(104)
    x = tf.random.normal((2, 32, 3))
    probe = tf.random.normal(x.shape)

    def input_gradient(backend):
        variable = tf.Variable(x)
        layer = layer_class(wave=wave, clean=False, backend=backend)
        with tf.GradientTape() as tape:
            output = layer(variable)
            loss = tf.reduce_sum(output * probe)
        return tape.gradient(loss, variable)

    _assert_close(input_gradient("filterbank"), input_gradient("matrix"))


def test_filterbank_backend_is_serialized_and_validated():
    layer = DWT1D(wave="bior2.2", clean=False, backend="filterbank")
    restored = DWT1D.from_config(layer.get_config())

    assert restored.wave == "bior2.2"
    assert restored.clean is False
    assert restored.backend == "filterbank"

    with pytest.raises(ValueError, match="backend must be one of"):
        DWT1D(backend="unknown")


def test_filterbank_round_trip_in_keras_graph():
    inputs = tf.keras.Input(shape=(32, 3))
    coefficients = DWT1D(
        wave="bior2.2", clean=True, backend="filterbank"
    )(inputs)
    outputs = IDWT1D(
        wave="bior2.2", clean=True, backend="filterbank"
    )(coefficients)
    model = tf.keras.Model(inputs, outputs)
    restored_model = tf.keras.models.clone_model(model)

    tf.random.set_seed(105)
    x = tf.random.normal((2, 32, 3))
    _assert_close(model(x), x)
    _assert_close(restored_model(x), x)
