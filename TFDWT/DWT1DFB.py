import tensorflow as tf
import keras
from TFDWT.DWTFBlayout import DWTNDlayout, IDWTNDlayout
# from TFDWT.DWTFilters import FetchAnalysisSynthesisFilters
from TFDWT.dwt_op import make_dwt_operator_matrix_A
# from TFDWT.DWTop import DWTop

@keras.saving.register_keras_serializable()
class DWT1D(DWTNDlayout):
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

    Note: if clean==True  then I/O (batch, N, channels) -> (batch, N/2, channels*2)
          if clean==False then I/O (batch, N, channels) -> (batch, N, channels)

    DWT1D layer  --kkt@04Jul2024"""
    def __init__(self, wave='haar', clean=True, **kwargs):
        super().__init__(wave=wave, clean=clean, **kwargs)

    def call(self, inputs):
        # inputs: (batch, N, channels)
        out = tf.einsum('bic,ij->bjc', inputs, self.A)
        if self.clean: return self.__extract_2subbands(out)
        else: return out

    def __extract_2subbands(self,LH_padded):
        """returns 2 subbands L, H from DWT Analysis bank o/p --@k"""
        mid = int(LH_padded.shape[1]/2)
        L = LH_padded[:,:mid,:]
        H = LH_padded[:,mid:,:]
        return tf.concat([L, H], axis=-1)


#%%
@keras.saving.register_keras_serializable()
class IDWT1D(IDWTNDlayout):
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
        

    Note: if clean==True  then I/O (batch, N/2, channels*2) -> (batch, N, channels)
          if clean==False then I/O (batch, N, channels) -> (batch, N, channels)  
    
    IDWT1D layer --kkt@04Jul2024"""
    def __init__(self, wave='haar', clean=True, **kwargs):
        super().__init__(wave=wave, clean=clean, **kwargs)
    
    def call(self, inputs):
        # inputs: (batch, N, channels)
        if self.clean: inputs = self.__join_2subbands(inputs)
        out = tf.einsum('bic,ij->bjc', inputs, self.S)
        return out

    def __join_2subbands(self, concat_subbands):
        """
        Inverts tf.concat([L, H], axis=-1) where L, H have shape (batch, mid, channels).
        Returns (batch, 2*mid, channels).
        """
        L, H = tf.split(concat_subbands, 2, axis=-1)
        return tf.concat([L, H], axis=1)

if __name__=='__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"]="-1"  
    wave = 'haar'
    dwt_layer = DWT1D(wave)
    idwt_layer = IDWT1D(wave)
    
    ## clean flag: (batch, N, channels)->(batch, N/2, channels)
    # dwt_layer = DWT1D(wave, clean=False)
    # idwt_layer = IDWT1D(wave,  clean=False)

    x = tf.random.normal((2, 256, 1))  # batch=1, length=256, channels=2
    print('\nx', x.shape)
    ## DWT
    lh = dwt_layer(x)
    print('lh', lh.shape)
    ## IDWT
    xhat = idwt_layer(lh)
    print("DWT output shape:", lh.shape)
    print("Reconstruction error (max.)", tf.reduce_max(tf.math.abs(x-xhat)))



    ## Example 2
    # Functional model
    N, channels, filters = 4, 1, 1
    input_shape = (N, channels)  # Replace N with the actual size of x            #1D
    inputs = tf.keras.Input(shape=input_shape)
    H1 = DWT1D(wave)
    H2 = IDWT1D(wave)
    lh = H1(inputs)
    outputs = H2(lh)
    # Build the model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mse', jit_compile=False)
    model.summary()
    # Random data
    ## 1D
    inputs_data = tf.random.normal((1, N, 1))
    targets = tf.random.normal((1, N, 1))
    # Training loop for 5 epochs
    epochs=5
    # for epoch in range(5):
    history = model.fit(inputs_data, targets, epochs=5, verbose=1)