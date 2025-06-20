import tensorflow as tf
import keras
from TFDWT.DWTFilters import FetchAnalysisSynthesisFilters
from TFDWT.dwt_operator import make_dwt_operator_matrix_A

@keras.saving.register_keras_serializable()
class DWT1D(tf.keras.layers.Layer):
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

    Note: if clean==True  then I/O (batch, N, channels) -> (batch, N/2, channels*2)
          if clean==False then I/O (batch, N, channels) -> (batch, N, channels)

    DWT1D layer  --kkt@20Jun2024"""
    def __init__(self, wave='haar', clean=True, **kwargs):
        super().__init__(**kwargs)
        self.wave = wave
        self.clean = clean  
        w = FetchAnalysisSynthesisFilters(wave)
        self.h0, self.h1 = w.analysis()
        self.L = len(self.h0)
    
    def build(self, input_shape):
        self.N = input_shape[1]
        A = make_dwt_operator_matrix_A(self.h0,self.h1,self.N)
        self.A = tf.cast(tf.transpose(A, perm=[1,0]), tf.float32)

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

    def get_config(self):
        config = super().get_config()
        config.update({
            'wave': self.wave,
            'clean': self.clean,
            # 'h0': list(self.h0),  # Converts TrackedList to regular list
            # add h1 if you also need it
            # 'h1': list(self.h1),
        })
        return config

#%%
@keras.saving.register_keras_serializable()
class IDWT1D(tf.keras.layers.Layer):
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
    

    Note: if clean==True  then I/O (batch, N/2, channels*2) -> (batch, N, channels)
          if clean==False then I/O (batch, N, channels) -> (batch, N, channels)  
    
    IDWT1D layer --kkt@20Jun2024"""
    def __init__(self, wave='haar', clean=True, **kwargs):
        super().__init__(**kwargs)
        self.wave = wave
        self.clean = clean
        w = FetchAnalysisSynthesisFilters(wave)
        if 'bior' in wave or 'rbio' in wave:
            """BIORTHOGONAL wavelets"""
            self.h0, self.h1 = w.synthesis()
            # print(f"Biothogonal wavelet {wave}")
        else:
            """ORTHOGONAL wavelets"""
            self.h0, self.h1 = w.analysis()
        
        self.L = len(self.h0)

    def build(self, input_shape):
        if self.clean: self.N = int(input_shape[1] * 2)
        else: self.N = int(input_shape[1])
        A = make_dwt_operator_matrix_A(self.h0,self.h1,self.N)
        self.S = tf.cast(A, tf.float32)  # No transpose needed

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

    def get_config(self):
        config = super().get_config()
        config.update({
            'wave': self.wave,
            'clean': self.clean,
            # 'h0': list(self.h0),  # Converts TrackedList to regular list
            # add h1 if you also need it
            # 'h1': list(self.h1),
        })
        return config


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