import tensorflow as tf
# from keras import regularizers
from keras import ops
from TFDWT.DWT1DFB import DWT1D, IDWT1D
from TFDWT.DWTFilters import FetchAnalysisSynthesisFilters
from TFDWT.dwt_op import (
    analysis_filterbank_axis,
    make_dwt_operator_matrix_A,
    operator_matrix_axis,
    synthesis_filterbank_axis,
)
from tensorflow.keras.layers import Concatenate


_VALID_BACKENDS = ('matrix', 'filterbank')


def _validate_backend(backend):
    if backend not in _VALID_BACKENDS:
        choices = ', '.join(repr(value) for value in _VALID_BACKENDS)
        raise ValueError(f"backend must be one of: {choices}.")
    return backend


def _validate_level(level):
    if isinstance(level, bool) or not isinstance(level, int) or level < 0:
        raise ValueError("level must be a non-negative integer.")
    return level


def _normalize_axis(x, axis):
    rank = x.shape.rank
    if rank is None:
        raise ValueError("The input rank must be statically known.")
    if axis < 0:
        axis += rank
    if axis < 0 or axis >= rank:
        raise ValueError(f"axis={axis} is invalid for rank-{rank} input.")
    return axis


def _validate_packed_shape(x, level, filter_length, axis):
    axis = _normalize_axis(x, axis)
    length = x.shape[axis]
    if length is None:
        raise ValueError("The transformed length must be statically known.")
    length = int(length)
    divisor = 2 ** level
    if length % divisor:
        raise ValueError(
            f"The transformed length must be divisible by 2**level={divisor}."
        )
    if level and length // (2 ** (level - 1)) < filter_length:
        raise ValueError(
            "Every decomposition level must cover the wavelet-filter length."
        )
    return axis, length


def _wavelet_filters(wave):
    filters = FetchAnalysisSynthesisFilters(wave)
    analysis = filters.analysis()
    synthesis = (
        filters.synthesis()
        if 'bior' in wave or 'rbio' in wave
        else analysis
    )
    return analysis, synthesis


def _analysis_axis(x, filters, axis, backend):
    if backend == 'filterbank':
        return analysis_filterbank_axis(x, *filters, axis=axis)
    length = int(x.shape[axis])
    operator = make_dwt_operator_matrix_A(*filters, length)
    return operator_matrix_axis(x, operator, axis=axis)


def _synthesis_axis(x, filters, axis, backend):
    if backend == 'filterbank':
        return synthesis_filterbank_axis(x, *filters, axis=axis)
    length = int(x.shape[axis])
    operator = tf.transpose(
        make_dwt_operator_matrix_A(*filters, length)
    )
    return operator_matrix_axis(x, operator, axis=axis)


def dwt_packed_axis(x, level=3, wave='haar', axis=1, backend='matrix'):
    """Packed multilevel 1D DWT along an arbitrary tensor axis.

    The selected axis retains its length and is ordered as
    ``[L_level, H_level, H_(level-1), ..., H_1]``.
    """
    x = tf.convert_to_tensor(x)
    level = _validate_level(level)
    backend = _validate_backend(backend)
    analysis, _ = _wavelet_filters(wave)
    axis, _ = _validate_packed_shape(
        x,
        level,
        len(analysis[0]),
        axis,
    )
    if level == 0:
        return x

    highpasses = []
    current = x
    for _ in range(level):
        packed = _analysis_axis(current, analysis, axis, backend)
        current, highpass = tf.split(packed, 2, axis=axis)
        highpasses.append(highpass)
    return tf.concat([current] + list(reversed(highpasses)), axis=axis)


def idwt_packed_axis(x, level=3, wave='haar', axis=1, backend='matrix'):
    """Inverse of :func:`dwt_packed_axis`."""
    x = tf.convert_to_tensor(x)
    level = _validate_level(level)
    backend = _validate_backend(backend)
    _, synthesis = _wavelet_filters(wave)
    axis, length = _validate_packed_shape(
        x,
        level,
        len(synthesis[0]),
        axis,
    )
    if level == 0:
        return x

    lowest_length = length // (2 ** level)
    sizes = [lowest_length, lowest_length]
    sizes.extend(
        lowest_length * (2 ** exponent)
        for exponent in range(1, level)
    )
    lowpass, *highpasses = tf.split(x, sizes, axis=axis)
    current = lowpass
    for highpass in highpasses:
        packed = tf.concat([current, highpass], axis=axis)
        current = _synthesis_axis(
            packed,
            synthesis,
            axis,
            backend,
        )
    return current


def dwt(x, level=3, Ψ='haar', backend='matrix'):
    """ Multilevel 1D DWT
    
        TFDWT: Fast Discrete Wavelet Transform TensorFlow Layers.
        Copyright 2026 Kishore Kumar Tarafdar

        Licensed under the Apache License, Version 2.0 (the "License");
        you may not use this file except in compliance with the License.
        You may obtain a copy of the License at

            https://www.apache.org/licenses/LICENSE-2.0

        Unless required by applicable law or agreed to in writing, software
        distributed under the License is distributed on an "AS IS" BASIS,
        WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
        See the License for the specific language governing permissions and
        limitations under the License.
    """
    level = _validate_level(level)
    backend = _validate_backend(backend)
    subbands = []
    current = x
    channels_in = x.shape[-1]

    for _ in range(level):
        w = DWT1D(wave=Ψ, backend=backend)(current)
        lowpass = w[:, :, :channels_in]
        highpass = w[:, :, channels_in:]
        subbands.append(highpass)
        current = lowpass
    subbands.append(current)
    return subbands


def idwt(subbands, level=3, Ψ='haar', backend='matrix'):
    """ Multilevel 1D IDWT
    
        TFDWT: Fast Discrete Wavelet Transform TensorFlow Layers.
        Copyright 2026 Kishore Kumar Tarafdar

        Licensed under the Apache License, Version 2.0 (the "License");
        you may not use this file except in compliance with the License.
        You may obtain a copy of the License at

            https://www.apache.org/licenses/LICENSE-2.0

        Unless required by applicable law or agreed to in writing, software
        distributed under the License is distributed on an "AS IS" BASIS,
        WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
        See the License for the specific language governing permissions and
        limitations under the License.
    """
    level = _validate_level(level)
    backend = _validate_backend(backend)
    if len(subbands) != level + 1:
        raise ValueError(
            f"level={level} requires {level + 1} subbands, "
            f"but received {len(subbands)}."
        )
    *highpasses, lowpass = subbands  # unpack: [H1, H2, ..., Hn, ln]
    
    current = lowpass
    for H in reversed(highpasses):
        current = IDWT1D(
            wave=Ψ,
            backend=backend,
        )(Concatenate()([current, H]))
    
    return current

if __name__=='__main__':
    batch_size, N, channels = 1, 32, 2
    x = tf.random.normal((batch_size, N, channels))
    x.shape
    level = 4
    subbands = dwt(x, level=level)
    print([_.shape for _ in subbands])
    x_rec = idwt(subbands, level=level)
    print(np.allclose(x,x_rec, atol=1e-9))
