import tensorflow as tf

# @tf.function
def make_dwt_operator_matrix_A(h0, h1, N: int):
    """
    Returns DWT operator matrix A built from h0, h1 filters for signal of length N.
    Uses TensorArray to construct the row-shifted convolution matrices.
    """
    h0 = tf.convert_to_tensor(h0, dtype=tf.float32)
    h1 = tf.convert_to_tensor(h1, dtype=tf.float32)

    L = tf.shape(h0)[0]
    tf.debugging.assert_greater(L, 0, "Filter length must be positive")

    def H_branch_row(h):
        pad_len = N - L
        zeros = tf.zeros([pad_len], dtype=h.dtype)
        return tf.concat([h, zeros], axis=0)

    def H_start_row(row):
        return tf.roll(row, shift=-(L - 2), axis=0)

    def H_branch_tensorarray(row):
        num_rows = N // 2
        ta = tf.TensorArray(dtype=row.dtype, size=num_rows)
        def body(i, ta):
            shifted = tf.roll(row, shift=2 * i, axis=0)
            return i + 1, ta.write(i, shifted)
        _, ta_final = tf.while_loop(lambda i, _: i < num_rows, body, [0, ta])
        return ta_final.stack()

    h0_row = H_start_row(H_branch_row(h0))
    h1_row = H_start_row(H_branch_row(h1))

    H0 = H_branch_tensorarray(h0_row)
    H1 = H_branch_tensorarray(h1_row)

    A = tf.concat([H0, H1], axis=0)
    return A


def _move_axis_to_last(x, axis):
    """Move one statically known tensor axis to the last position."""
    rank = x.shape.rank
    if rank is None:
        raise ValueError("The input rank must be statically known.")
    if axis < 0:
        axis += rank
    if axis < 0 or axis >= rank:
        raise ValueError(f"axis={axis} is invalid for rank-{rank} input.")

    permutation = [index for index in range(rank) if index != axis] + [axis]
    inverse_permutation = [0] * rank
    for index, value in enumerate(permutation):
        inverse_permutation[value] = index
    return tf.transpose(x, permutation), inverse_permutation


def _validate_filterbank_inputs(x, h0, h1):
    """Convert filters and validate the constraints shared by DWT and IDWT."""
    h0 = tf.cast(tf.convert_to_tensor(h0), x.dtype)
    h1 = tf.cast(tf.convert_to_tensor(h1), x.dtype)
    filter_length = h0.shape[0]
    if filter_length is None:
        raise ValueError("The wavelet-filter length must be statically known.")
    if filter_length < 2:
        raise ValueError("The wavelet-filter length must be at least two.")
    if h1.shape[0] != filter_length:
        raise ValueError("Lowpass and highpass filters must have equal length.")
    return h0, h1, int(filter_length)


def operator_matrix_axis(x, operator, axis):
    """Apply a square linear operator along one arbitrary tensor axis."""
    x = tf.convert_to_tensor(x)
    operator = tf.cast(tf.convert_to_tensor(operator), x.dtype)
    moved, inverse_permutation = _move_axis_to_last(x, axis)
    moved_shape = tf.shape(moved)
    length = moved_shape[-1]
    checks = (
        tf.debugging.assert_equal(
            tf.shape(operator)[0],
            length,
            message="The operator output length must match the selected axis.",
        ),
        tf.debugging.assert_equal(
            tf.shape(operator)[1],
            length,
            message="The operator input length must match the selected axis.",
        ),
    )
    with tf.control_dependencies(checks):
        fibres = tf.reshape(moved, [-1, length])
    transformed = tf.einsum('ij,bj->bi', operator, fibres)
    restored = tf.reshape(transformed, moved_shape)
    return tf.transpose(restored, inverse_permutation)


def analysis_filterbank_axis(x, h0, h1, axis):
    """Apply the packed periodic analysis bank along one tensor axis.

    This is the matrix-free equivalent of multiplying by the operator made by
    ``make_dwt_operator_matrix_A``. The selected axis retains its length and is
    packed as ``[lowpass | highpass]``.
    """
    x = tf.convert_to_tensor(x)
    h0, h1, filter_length = _validate_filterbank_inputs(x, h0, h1)
    moved, inverse_permutation = _move_axis_to_last(x, axis)
    moved_shape = tf.shape(moved)
    length = moved_shape[-1]
    checks = (
        tf.debugging.assert_equal(
            tf.math.floormod(length, 2),
            0,
            message="The transformed length must be even.",
        ),
        tf.debugging.assert_greater_equal(
            length,
            filter_length,
            message="The transformed length must cover the wavelet filter.",
        ),
    )

    with tf.control_dependencies(checks):
        fibres = tf.reshape(moved, [-1, length, 1])

    pad_left = filter_length - 2
    if pad_left:
        fibres = tf.concat([fibres[:, -pad_left:, :], fibres], axis=1)

    filters = tf.expand_dims(tf.stack([h0, h1], axis=-1), axis=1)
    subbands = tf.nn.conv1d(
        fibres,
        filters,
        stride=2,
        padding='VALID',
        data_format='NWC',
    )
    packed = tf.concat([subbands[:, :, 0], subbands[:, :, 1]], axis=1)
    restored = tf.reshape(packed, moved_shape)
    return tf.transpose(restored, inverse_permutation)


def synthesis_filterbank_axis(x, g0, g1, axis):
    """Apply the packed periodic synthesis bank along one tensor axis.

    This is the matrix-free equivalent of multiplying by the transpose of an
    operator made from the synthesis filters. The selected input axis must be
    packed as ``[lowpass | highpass]``.
    """
    x = tf.convert_to_tensor(x)
    g0, g1, filter_length = _validate_filterbank_inputs(x, g0, g1)
    moved, inverse_permutation = _move_axis_to_last(x, axis)
    moved_shape = tf.shape(moved)
    length = moved_shape[-1]
    checks = (
        tf.debugging.assert_equal(
            tf.math.floormod(length, 2),
            0,
            message="The transformed length must be even.",
        ),
        tf.debugging.assert_greater_equal(
            length,
            filter_length,
            message="The transformed length must cover the wavelet filter.",
        ),
    )

    with tf.control_dependencies(checks):
        fibres = tf.reshape(moved, [-1, length])

    half = length // 2
    coefficients = tf.stack(
        [fibres[:, :half], fibres[:, half:]],
        axis=-1,
    )
    filters = tf.expand_dims(tf.stack([g0, g1], axis=-1), axis=1)
    pad_left = filter_length - 2
    output_shape = tf.stack(
        [tf.shape(fibres)[0], length + pad_left, 1]
    )
    padded = tf.nn.conv1d_transpose(
        coefficients,
        filters,
        output_shape=output_shape,
        strides=2,
        padding='VALID',
        data_format='NWC',
    )

    natural = padded[:, pad_left:, :]
    if pad_left:
        natural = tf.concat(
            [
                natural[:, :length - pad_left, :],
                natural[:, length - pad_left:, :] + padded[:, :pad_left, :],
            ],
            axis=1,
        )

    restored = tf.reshape(tf.squeeze(natural, axis=-1), moved_shape)
    return tf.transpose(restored, inverse_permutation)
