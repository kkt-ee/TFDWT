import tensorflow as tf

def make_dwt_operator_matrix_A(h0, h1, N: int):
    """TFDWT: Fast Discrete Wavelet Transform TensorFlow Layers.
    Copyright (C) 2025 Kishore Kumar Tarafdar

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
    Create DWT operator matrix A from filters h0 and h1 (1D),
    for signal of length N.
    
    Returns Analysis or Synthesis Transform Matrix 'A'"""

    # Ensure inputs are tensors
    h0 = tf.convert_to_tensor(h0, dtype=tf.float32)
    h1 = tf.convert_to_tensor(h1, dtype=tf.float32)

    # Static filter length (positive integer)
    L = h0.shape[0]
    assert isinstance(L, int) and L > 0, "Filter must have known static length"

    def H_branch_row(h):
        pad_len = N - L
        return tf.concat([h, tf.zeros([pad_len], dtype=h.dtype)], axis=0)

    def H_start_row(row):
        return tf.roll(row, shift=-(L - 2), axis=0)

    def H_branch(row):
        num_rows = N // 2
        return tf.stack([tf.roll(row, shift=2 * k, axis=0) for k in range(num_rows)], axis=0)

    H0 = H_branch(H_start_row(H_branch_row(h0)))
    H1 = H_branch(H_start_row(H_branch_row(h1)))

    A = tf.concat([H0, H1], axis=0)
    return A

#%% Numpy version
# import numpy as np

# def make_dwt_operator_matrix_A(h0,h1,N:int):
#     """Returns Analysis or Synthesis Transform Matrix 'A'

#     TFDWT: Fast Discrete Wavelet Transform TensorFlow Layers.
#     Copyright (C) 2025 Kishore Kumar Tarafdar

#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.

#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.

#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <https://www.gnu.org/licenses/>."""
    
#     L = len(h0)
#     __H_branch_row = lambda h0,N: np.concatenate((h0, np.zeros(int(N - len(h0)))))
#     __H_start_row = lambda _row: np.roll(_row, shift=-L+2, axis=None)
#     __H_start_row(__H_branch_row(h0,N))
#     __H_branch= lambda _row: [np.roll(_row, shift=k*2, axis=None) for k in range(_row.shape[0]//2)]#-len(h0)+1)]
#     H0 = __H_branch(__H_start_row(__H_branch_row(h0,N)))
#     H1 = __H_branch(__H_start_row(__H_branch_row(h1,N)))
#     # return np.concatenate((H0[:int(len(H0)/2)], H1[:int(len(H1)/2)]))
#     return np.concatenate((H0, H1))

# # A = get_A_matrix_dwt_analysisFB_unit(h0,h1,N)
# # A.shape, A



