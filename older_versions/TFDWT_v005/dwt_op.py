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