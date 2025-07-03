"""1D DWT and IDWT layers

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
"""



import tensorflow as tf
import keras


from TFDWT.GETDWTFiltersOrtho import GETDWTFiltersOrtho
from TFDWT.get_A_matrix_dwt_analysisFB_unit import get_A_matrix_dwt_analysisFB_unit


#%% OK
@keras.saving.register_keras_serializable()
class DWT1D(tf.keras.layers.Layer):
    """1D DWT layer

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
    along with this program.  If not, see <https://www.gnu.org/licenses/>."""
    def __init__(self, wave='haar', **kwargs):
        super(DWT1D, self).__init__(**kwargs)
        w = GETDWTFiltersOrtho(wave)
        self.h0, self.h1 = w.analysis()
        self.L = len(self.h0)
        
    def build(self, input_shape):
        self.num_channels = input_shape[-1]
        self.N = input_shape[1]
        self.A = get_A_matrix_dwt_analysisFB_unit(self.h0,self.h1,self.N)
        super(DWT1D, self).build(input_shape) 
       
    
    def call(self, inputs):
        ch_ = [self.__dwt(inputs[..., i:i+1]) for i in range(self.num_channels)]
        out = tf.concat(ch_, axis=-1)
        #
        out = self.__extract_2subbands(out)
        return out

    # @tf.function
    def __dwt(self, x):
        # print('>>',x.shape)
        x = tf.squeeze(x, axis=-1)
        dwtout = tf.cast(self.A, tf.float32)@tf.transpose(tf.cast(x, tf.float32), perm=[1,0])
        #OR einsum !!
        # dwtout = tf.einsum('ij,bjk->bik', tf.cast(self.A, tf.float32), tf.transpose(tf.cast(x, tf.float32), perm=[1,0]))
        dwtout = tf.transpose(dwtout,perm=[1,0])
        # print('++',dwtout.shape)
        return tf.expand_dims(dwtout, axis=-1)

    def __extract_2subbands(self,LH_padded):
        """returns 2 subbands L, H from DWT Analysis bank o/p --@k"""
     
        _mid = int(LH_padded.shape[1]/2)
        L = LH_padded[:,:_mid,:]
        H = LH_padded[:,_mid:,:]
        
        # _i = int((L.shape[1] - self.N/2)/2)
        # out = tf.concat([L[:,_i:-_i,:], H[:,_i:-_i,:]], axis=-1)
        out = tf.concat([L, H], axis=-1)
        return out

    # def compute_output_shape(self, input_shape):
    #     batch_size, height, width, channels = input_shape
    #     new_height, new_width = height // 2, width // 2
    #     return (batch_size, new_height, new_width, 4 * channels)
    
    def get_config(self):
        config = super(DWT1D, self).get_config()
        return config




#%% OK
@keras.saving.register_keras_serializable()
class IDWT1D(tf.keras.layers.Layer):
    """1D IDWT layer

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
    along with this program.  If not, see <https://www.gnu.org/licenses/>."""
    def __init__(self, wave='haar', **kwargs):
        super(IDWT1D, self).__init__(**kwargs)
        w = GETDWTFiltersOrtho(wave)
        
        # self.biortho_flag = 0
        if 'bior' in wave or 'rbio' in wave:
            """BIORTHOGONAL wavelets"""
            self.h0, self.h1 = w.synthesis()
            # self.biortho_flag = 1
            # print(f"Biothogonal wavelet {wave}")
        else:
            """ORTHOGONAL wavelets"""
            self.h0, self.h1 = w.analysis()
        
        self.L = len(self.h0)
       

    def build(self, input_shape):
        # self.num_channels = input_shape[-1]
        self.N = int(input_shape[1]*2)
        self.A = get_A_matrix_dwt_analysisFB_unit(self.h0,self.h1,self.N)
        super(IDWT1D, self).build(input_shape) 

    def call(self, inputs):
        __ = self.__splitch_for_idwt(inputs)
        __ = [self.__rejoin(_) for _ in __]# axis=-1)
        out = tf.stack([self.__idwt(_) for _ in __], axis=-1)
        return out#[:,L:-L,:]

    @tf.function
    def __idwt(self, x_dwtcoeffs):
        __ = x_dwtcoeffs
        idwtout = tf.cast(self.A.T, tf.float32) @ tf.transpose(__,perm=[1,0])
        idwtout = tf.transpose(idwtout,perm=[1,0])     
        return idwtout#[L:-L,L:-L,:]

    def __splitch_for_idwt(self,x):
        w = int(x.shape[-1]/2)
        __ = [tf.stack([x[:,:,k], x[:,:,k+1*w]],  axis=-1) for k in range(w)]
        return __
    #__ = __splitch_for_idwt(coeffs)

    def __rejoin(self,subbands):
        el = tf.unstack(subbands, axis=-1)
        # print('++++', el)
        _1 = tf.concat([el[0][:,:], el[1][:,:]], axis=-1)
        return _1
    #tf.stack([rejoin(_) for _ in __], axis=-1)

    # def compute_output_shape(self, input_shape):
    #     batch_size, height, width, channels = input_shape
    #     new_height, new_width, new_channels = height *2, width *2, channels //4
    #     return (batch_size, new_height, new_width, new_channels)
   
    def get_config(self):
        config = super(IDWT1D, self).get_config()
        return config
    
if __name__ == '__main__':

    # 1D dwt
    N = 256 # length of the sequence
    input_shape = (N,1)
    import numpy as np
    import matplotlib.pyplot as plt
    plt.rcParams['font.size'] = '18'
    x = np.random.rand(input_shape[0])
    # x = tmpx
    plt.figure(figsize=(15,2))
    plt.plot(x, 'g.-',label='$x$')
    plt.legend(), plt.grid()
    plt.title('$x$')
    plt.show()
    print(f"Raw x shape {x.shape}")



    mother_wavelet = 'bior3.1' # max 'db8' for lenght 16 signal
    # mother_wavelet = 'haar' # max 'db8' for lenght 16 signal
    # mother_wavelet = 'db10' # max 'db8' for lenght 16 signal
    newx = tf.expand_dims(tf.expand_dims(x,-1),0)
    print(f'x shape {newx.shape}')
    dwtout = DWT1D(wave=mother_wavelet)(newx)
    print(f'DWT(x) shape {dwtout.shape}, \nDWT(x) := {dwtout}')

    # print(dwtout.shape)
    idwtout = IDWT1D(wave=mother_wavelet)(dwtout)
    print(f'IDWT(DWT(x)) shape {idwtout.shape} \nIDWT(DWT(x)) := {idwtout}')

    print(f'Check perfect reconstruction \nIDWT(DWT(x)) := {idwtout.numpy()} \n\nInput x is {x} ')

    

    plt.figure(figsize=(16,2))
    plt.plot(x,'o-')
    plt.plot(idwtout.numpy()[0,:], '.--'), plt.title(f"reconstuction using {mother_wavelet}")



