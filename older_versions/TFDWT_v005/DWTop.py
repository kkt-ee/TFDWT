import tensorflow as tf
import keras

@keras.saving.register_keras_serializable()
class DWTop(tf.keras.layers.Layer):
    """
    DWTop: Keras Layer that constructs the DWT operator matrix A.
    Serializes/Deserializes h0, h1, N for model saving/loading.

    TFDWT: Fast Discrete Wavelet Transform TensorFlow Layers.
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
    
    Returns Analysis or Synthesis Transform Matrix 'A'

    --kkt@23Jun2025"""
    def __init__(self, h0, h1, N, **kwargs):
        super().__init__(**kwargs)
        self.h0 = tf.convert_to_tensor(h0, dtype=tf.float32)
        self.h1 = tf.convert_to_tensor(h1, dtype=tf.float32)
        self.N = N
        # self.A = None
        # self.A = self.__make_dwt_operator_matrix_A(self.h0, self.h1, self.N)

        # def build(self, input_shape):
        # N = input_shape[1]
        def make_dwt_operator_matrix_A(h0, h1, N:int):
            # Ensure inputs are tensors
            # h0 = tf.convert_to_tensor(h0, dtype=tf.float32)
            # h1 = tf.convert_to_tensor(h1, dtype=tf.float32)

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
        
        self.A = make_dwt_operator_matrix_A(self.h0, self.h1, self.N)
        # self.A = tf.constant(A, name="DWT_matrix_A")  # no trainable weights
        # self.built = True
        # super().build(input_shape)
    
    def call(self, inputs=None):
        # A = self.add_weight(
        #         name="dwt_matrix_A",
        #         shape=A.shape,
        #         initializer=tf.constant_initializer(self.A),  # avoid .numpy()
        #         trainable=False)
        return self.A
        # pass
  
    def get_config(self):
        config = super().get_config()
        config.update({
            'h0': self.h0.numpy().tolist(),
            'h1': self.h1.numpy().tolist(),
            'N': self.N
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(
            h0=config['h0'],
            h1=config['h1'],
            N=config['N']
        )

if __name__=='__main__':
    from TFDWT.DWTFilters import FetchAnalysisSynthesisFilters
    w = FetchAnalysisSynthesisFilters('haar')
    h0, h1 = w.analysis()
    layer = DWTop(h0,h1,8)
    # layer.build(None)
    type(layer.A)
    print('A = ',layer.A)