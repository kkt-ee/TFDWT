import tensorflow as tf
import keras
from TFDWT.DWTFBlayout import DWTNDlayout, IDWTNDlayout
# from TFDWT.DWTFilters import FetchAnalysisSynthesisFilters
from TFDWT.dwt_op import make_dwt_operator_matrix_A
# from TFDWT.DWTop import DWTop

@keras.saving.register_keras_serializable()
class DWT3D(DWTNDlayout):
    """ TFDWT: Fast Discrete Wavelet Transform TensorFlow Layers.
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
        
    Note: if clean==True  then I/O (batch, N, N, N, channels) -> (batch, N/2, N/2, N/2, channels*8)
          if clean==False then I/O (batch, N, N, N, channels) -> (batch, N, N, N, channels)
    
    DWT3D layer  --kkt@4Jul2024"""
    def __init__(self, wave='haar', clean=True, **kwargs):
        super().__init__(wave=wave, clean=clean, **kwargs)

    def build(self, input_shape):
        # self.num_channels = input_shape[-1]
        self.N = input_shape[1]
        A = make_dwt_operator_matrix_A(self.h0,self.h1,self.N)
        # A = DWTop(self.h0,self.h1,self.N).A
        self.A = tf.cast(A, tf.float32) 
        super().build(input_shape)

    def call(self, inputs):
        # Inputs: (batch, row, col, depth, channel)
        A = self.A

        # Step 1: columns (axis 2)
        x = tf.transpose(inputs, [0, 2, 1, 3, 4])      # (batch, col, row, depth, ch)
        x = tf.einsum('ij,bjklc->biklc', A, x)         # apply S along columns
        x = tf.transpose(x, [0, 2, 1, 3, 4])           # (batch, row, col, depth, ch)

        # Step 2: rows (axis 1)
        x = tf.einsum('ij,bjklc->biklc', A, x)         # apply S along rows

        # Step 3: depth (axis 3)
        x = tf.transpose(x, [0, 3, 1, 2, 4])           # (batch, depth, row, col, ch)
        x = tf.einsum('ij,bjklc->biklc', A, x)         # apply S along depth
        x = tf.transpose(x, [0, 2, 3, 1, 4])           # back to (batch, row, col, depth, ch)
        if self.clean: return self.__extract_8subbands(x)
        else: return x

    def __extract_8subbands(self, LLLLLHLHLLHHHLLHLHHHLHHH):
        # arr shape: (batch, N, N, N, channels)
        mid = LLLLLHLHLLHHHLLHLHHHLHHH.shape[1] // 2

        LLL = LLLLLHLHLLHHHLLHLHHHLHHH[:, :mid, :mid, :mid, :]
        LLH = LLLLLHLHLLHHHLLHLHHHLHHH[:, :mid, :mid, mid:, :]
        LHL = LLLLLHLHLLHHHLLHLHHHLHHH[:, :mid, mid:, :mid, :]
        LHH = LLLLLHLHLLHHHLLHLHHHLHHH[:, :mid, mid:, mid:, :]

        HLL = LLLLLHLHLLHHHLLHLHHHLHHH[:, mid:, :mid, :mid, :]
        HLH = LLLLLHLHLLHHHLLHLHHHLHHH[:, mid:, :mid, mid:, :]
        HHL = LLLLLHLHLLHHHLLHLHHHLHHH[:, mid:, mid:, :mid, :]
        HHH = LLLLLHLHLLHHHLLHLHHHLHHH[:, mid:, mid:, mid:, :]

        return tf.concat([LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH], axis=-1)

        
#%%
@keras.saving.register_keras_serializable()
class IDWT3D(IDWTNDlayout):
    """ TFDWT: Fast Discrete Wavelet Transform TensorFlow Layers.
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
        
        
    Note: if clean==True  then I/O (batch, N/2, N/2, N/2, channels*8) -> (batch, N, N, N, channels)
          if clean==False then I/O (batch, N, N, N, channels) -> (batch, N, N, N, channels)

    IDWT3D layer --kkt@20Jun2024"""
    def __init__(self, wave='haar', clean=True, **kwargs):
        super().__init__(wave=wave, clean=clean, **kwargs)

    def build(self, input_shape):
        if self.clean: self.N = int(input_shape[1]*2)
        else: self.N = int(input_shape[1])
        A = make_dwt_operator_matrix_A(self.h0,self.h1,self.N)
        # A = DWTop(self.h0,self.h1,self.N).A
        self.S = tf.cast(tf.transpose(A), tf.float32)  # (N, N)
        super().build(input_shape)
    
    def call(self, inputs):
        # Inputs: (batch, row, col, depth, channel)
        if self.clean: inputs = self.__join_octants(inputs)
        # S = self.S
        # Step 1: columns (axis 2)
        x = tf.transpose(inputs, [0, 2, 1, 3, 4])      # (batch, col, row, depth, ch)
        x = tf.einsum('ij,bjklc->biklc', self.S, x)         # apply S along columns
        x = tf.transpose(x, [0, 2, 1, 3, 4])           # (batch, row, col, depth, ch)

        # Step 2: rows (axis 1)
        x = tf.einsum('ij,bjklc->biklc', self.S, x)         # apply S along rows

        # Step 3: depth (axis 3)
        x = tf.transpose(x, [0, 3, 1, 2, 4])           # (batch, depth, row, col, ch)
        x = tf.einsum('ij,bjklc->biklc', self.S, x)         # apply S along depth
        x = tf.transpose(x, [0, 2, 3, 1, 4])           # back to (batch, row, col, depth, ch)
        return x

    def __join_octants(self, concat_octants):
        # concat_octants: shape (batch, mid, mid, mid, channels*8)
        batch, mid, _, _, total_channels = concat_octants.shape.as_list()
        channels = total_channels // 8

        # Split along channel axis
        octants = tf.split(concat_octants, num_or_size_splits=8, axis=-1)
        LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH = octants

        # Stack along depth (axis=3)
        front_top = tf.concat([LLL, LLH], axis=3)
        front_bot = tf.concat([LHL, LHH], axis=3)
        back_top  = tf.concat([HLL, HLH], axis=3)
        back_bot  = tf.concat([HHL, HHH], axis=3)

        # Stack along columns (axis=2)
        front = tf.concat([front_top, front_bot], axis=2)
        back = tf.concat([back_top, back_bot], axis=2)

        # Stack along rows (axis=1)
        arr = tf.concat([front, back], axis=1)

        return arr  # shape: (batch, N, N, N, channels)
    

if __name__=='__main__':  
    import os
    os.environ["CUDA_VISIBLE_DEVICES"]="-1"    
    wavelet = 'bior1.5'
    # dwt_layer = DWT3D(wave=wavelet)
    # idwt_layer = IDWT3D(wave=wavelet)
    dwt_layer = DWT3D(wave=wavelet, clean=False)
    idwt_layer = IDWT3D(wave=wavelet, clean=False)

    ##Example input 1
    # x = tf.random.normal((1, 256, 256, 256, 2))  # batch=1, length=256, channels=2

    #Example input 2
    n = 256
    axis1 = tf.range(n, dtype=tf.int32)
    # x = tf.einsum('i,j->ij', axis1, axis1)
    x = tf.einsum('i,j,k->ijk', axis1, axis1, axis1)
    x = tf.cast(tf.expand_dims(tf.expand_dims(x,axis=-1), axis=0), tf.float32)
    del axis1, n
    print('\nx', x.shape)

    ## DWT
    lh = dwt_layer(x)
    print('lh', lh.shape)

    ## IDWT
    xhat = idwt_layer(lh)
    print("DWT output shape:", lh.shape)
    print("Max reconstruction error\n\n", tf.reduce_max(tf.math.abs(x-xhat)).numpy())


    ## Example 
    # Functional model
    N, channels, filters = 4, 1, 1
    input_shape = (N, N, N, channels)  # Replace N with the actual size of x            #1D
    inputs = tf.keras.Input(shape=input_shape)
    H1 = DWT3D(wave='db2')
    H2 = IDWT3D(wave='db2')
    lh = H1(inputs)
    outputs = H2(lh)
    # Build the model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mse', jit_compile=False)
    model.summary()
    # Random data
    ## 3D
    inputs_data = tf.random.normal((1, N, N, N, channels))
    targets = tf.random.normal((1, N, N, N, channels))
    # Training loop for 5 epochs
    epochs=5
    # for epoch in range(5):
    history = model.fit(inputs_data, targets, epochs=5, verbose=1)