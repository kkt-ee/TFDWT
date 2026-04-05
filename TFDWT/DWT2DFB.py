import tensorflow as tf
import keras
from TFDWT.DWTFBlayout import DWTNDlayout, IDWTNDlayout
# from TFDWT.DWTFilters import FetchAnalysisSynthesisFilters
# from TFDWT.DWTop import DWTop

@keras.saving.register_keras_serializable()
class DWT2D(DWTNDlayout):
    """ TFDWT: Fast Discrete Wavelet Transform TensorFlow Layers.
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
        
    Note: if clean==True  then I/O (batch, N, N, channels) -> (batch, N/2, N/2 channels*4)
          if clean==False then I/O (batch, N, N, channels) -> (batch, N, N, channels)

    DWT2D layer  --kkt@20Jun2024"""
    def __init__(self, wave='haar', clean=True, **kwargs):
        super().__init__(wave=wave, clean=clean, **kwargs)

    def build(self, input_shape):
        # Defer to base layout which sets self.A = A
        super().build(input_shape)
       
    def call(self, inputs):        
        # Step 1: Transpose for each channel (swap axes 1 and 2)
        x = tf.transpose(inputs, [0, 2, 1, 3])  # (batch, N, N, channels)

        # Step 2: Row convolution (A @ input) along the 2nd dimension
        x = tf.einsum('ij,bjkc->bikc', self.A, x)  # (batch, N, N, channels)

        # Step 3: Transpose back
        x = tf.transpose(x, [0, 2, 1, 3])  # (batch, N, N, channels)

        # Step 4: Column convolution (A @ input) along the 2nd dimension again
        x = tf.einsum('ij,bjkc->bikc', self.A, x)  # (batch, N, N, channels)

        if self.clean: return self.__extract_4subbands(x)
        else: return x
        #   # (batch, N, N, channels)

    def __extract_4subbands(self,LLLHHLHH):
        """returns 4 image subbands LL, LH, HL, HH from DWT Analysis bank o/p --@k"""
        # global N
        mid = int(LLLHHLHH.shape[1]/2)
        LL = LLLHHLHH[:, :mid, :mid, :]
        LH = LLLHHLHH[:, mid:, :mid, :]
        HL = LLLHHLHH[:, :mid, mid:, :]
        HH = LLLHHLHH[:, mid:, mid:, :]
        # _i = int((LL.shape[1] - self.N/2)/2)
        # out = tf.concat([LL[:,_i:-_i,_i:-_i,:], LH[:,_i:-_i,_i:-_i,:], HL[:,_i:-_i,_i:-_i,:], HH[:,_i:-_i,_i:-_i,:]], axis=-1)
        out = tf.concat([LL, LH, HL, HH], axis=-1)
        return out

#%%
@keras.saving.register_keras_serializable()
class IDWT2D(IDWTNDlayout):
    """ TFDWT: Fast Discrete Wavelet Transform TensorFlow Layers.
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
    
    Note: if clean==True  then I/O (batch, N/2, N/2, channels*4) -> (batch, N, N channels)
          if clean==False then I/O (batch, N, N, channels) -> (batch, N, N, channels)

    IDWT2D layer --kkt@4Jul2024"""
    def __init__(self, wave='haar', clean=True, **kwargs):
        super().__init__(wave=wave, clean=clean, **kwargs)
    
    def build(self, input_shape):
        # Defer to base layout which sets self.S = A^T
        super().build(input_shape)

    def call(self, inputs):
        # inputs: (batch, N, N, channels)
        if self.clean: inputs = self.__join_quadrants(inputs)
        # Step 1: transpose N and N axes (1 and 2)
        x = tf.transpose(inputs, [0, 2, 1, 3])  # (batch, N, N, channels)
        # Step 2: matmul along axis 2 (columns after transpose)
        x = tf.einsum('ij,bjkc->bikc', self.S, x)
        # Step 3: transpose back
        x = tf.transpose(x, [0, 2, 1, 3])
        # Step 4: matmul along axis 2 again
        x = tf.einsum('ij,bjkc->bikc', self.S, x)
        return x

    def __join_quadrants(self, concat_quadrants):
        """
        Inverse of:
            LL = arr[:, :mid, :mid, :]
            LH = arr[:, mid:, :mid, :]
            HL = arr[:, :mid, mid:, :]
            HH = arr[:, mid:, mid:, :]
            out = tf.concat([LL, LH, HL, HH], axis=-1)
        """
        LL, LH, HL, HH = tf.split(concat_quadrants, num_or_size_splits=4, axis=-1)
        top = tf.concat([LL, HL], axis=2)     # [LL | HL]
        bottom = tf.concat([LH, HH], axis=2)  # [LH | HH]
        return tf.concat([top, bottom], axis=1)  # [top / bottom]


if __name__=='__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"]="-1"  
    wave = 'haar'
    dwt_layer = DWT2D(wave)
    idwt_layer = IDWT2D(wave)
    # dwt_layer = DWT2D(wave, clean=False)
    # idwt_layer = IDWT2D(wave, clean=False)

    # x = tf.random.normal((2, 256, 256, 5))  # batch=1, length=256, channels=2

    #Example input 2
    n = 256
    axis1 = tf.range(n, dtype=tf.int32)
    x = tf.einsum('i,j->ij', axis1, axis1)
    x = tf.cast(tf.expand_dims(tf.expand_dims(x,axis=-1), axis=0), tf.float32)
    print('\nx', x.shape)
    del axis1, n

    ## DWT
    lh = dwt_layer(x)
    print('lh', lh.shape)

    # IDWT
    xhat = idwt_layer(lh)
    print("DWT output shape:", lh.shape)
    print("Reconstruction error (max.)", tf.reduce_max(tf.math.abs(x-xhat)))


    ## Example 
    # Functional model
    N, channels, filters = 4, 1, 1
    input_shape = (N, N, channels)  # Replace N with the actual size of x            #1D
    inputs = tf.keras.Input(shape=input_shape)
    H1 = DWT2D(wave)
    H2 = IDWT2D(wave)
    lh = H1(inputs)
    outputs = H2(lh)
    # Build the model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mse', jit_compile=False)
    model.summary()
    # Random data
    ## 1D
    inputs_data = tf.random.normal((1, N, N, channels))
    targets = tf.random.normal((1, N, N, channels))
    # Training loop for 5 epochs
    epochs=5
    # for epoch in range(5):
    history = model.fit(inputs_data, targets, epochs=5, verbose=1)
