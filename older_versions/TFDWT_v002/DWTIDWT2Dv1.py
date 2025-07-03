"""2D DWT and IDWT layers

The transform matrix is of same dimension as of the signal. No padding is applied.
Circular convolution takes care of the boundary effects.


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



import tensorflow as tf
import keras

from TFDWT.GETDWTFiltersOrtho import GETDWTFiltersOrtho
from TFDWT.get_A_matrix_dwt_analysisFB_unit import get_A_matrix_dwt_analysisFB_unit

#%% OK

@keras.saving.register_keras_serializable()
class DWT2D(tf.keras.layers.Layer):
    """2D DWT layer 

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
        super(DWT2D, self).__init__(**kwargs)
        w = GETDWTFiltersOrtho(wave)
        self.h0, self.h1 = w.analysis()
        # g0, g1 = w.synthesis()
        self.L = len(self.h0)
        
    def build(self, input_shape):
        self.num_channels = input_shape[-1]
        self.N = input_shape[1]
        self.A = get_A_matrix_dwt_analysisFB_unit(self.h0,self.h1,self.N)
        super(DWT2D, self).build(input_shape) 
       
    
    def call(self, inputs):
        ch_ = [self.__dwt(inputs[..., i:i+1]) for i in range(self.num_channels)]
        out = tf.concat(ch_, axis=-1)
        #
        out = self.__extract_4subbands(out)
        return out

    @tf.function
    def __dwt(self, x):
        ## einsum code
        _x2d = tf.squeeze(x, axis=-1)
        _x2d = tf.transpose(_x2d,[0,2,1])
        A = tf.cast(self.A, tf.float32)
        _rowconv_out = tf.einsum('ij,bjk->bik', A, _x2d)
        _rowconv_outT = tf.transpose(_rowconv_out,[0,2,1])
        _LLLHHLHH = tf.einsum('ij,bjk->bik', A, _rowconv_outT)
        _colconv_out = _LLLHHLHH
        ## einsum code end
        return tf.expand_dims(_colconv_out, axis=-1)

    def __extract_4subbands(self,LLLHHLHH):
        """returns 4 image subbands LL, LH, HL, HH from DWT Analysis bank o/p --@k"""
        # global N
        _mid = int(LLLHHLHH.shape[1]/2)
        LL = LLLHHLHH[:,:_mid,:_mid,:]
        LH = LLLHHLHH[:,_mid:,:_mid,:]
        HL = LLLHHLHH[:,:_mid,_mid:,:]
        HH = LLLHHLHH[:,_mid:,_mid:,:]
        # _i = int((LL.shape[1] - self.N/2)/2)
        # out = tf.concat([LL[:,_i:-_i,_i:-_i,:], LH[:,_i:-_i,_i:-_i,:], HL[:,_i:-_i,_i:-_i,:], HH[:,_i:-_i,_i:-_i,:]], axis=-1)
        out = tf.concat([LL, LH, HL, HH], axis=-1)
        return out

    # def compute_output_shape(self, input_shape):
    #     batch_size, height, width, channels = input_shape
    #     new_height, new_width = height // 2, width // 2
    #     return (batch_size, new_height, new_width, 4 * channels)
    
    def get_config(self):
        config = super(DWT2D, self).get_config()
        return config





#%% OK
@keras.saving.register_keras_serializable()
class IDWT2D(tf.keras.layers.Layer):
    """2D IDWT layer 
    
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
        super(IDWT2D, self).__init__(**kwargs)
        w = GETDWTFiltersOrtho(wave)
        if 'bior' in wave or 'rbio' in wave:
            """BIORTHOGONAL wavelets"""
            self.h0, self.h1 = w.synthesis()
            # print(f"Biothogonal wavelet {wave}")
        else:
            """ORTHOGONAL wavelets"""
            self.h0, self.h1 = w.analysis()
        # g0, g1 = w.synthesis()
        self.L = len(self.h0)
        

    def build(self, input_shape):
        # self.num_channels = input_shape[-1]
        self.N = int(input_shape[1]*2)
        self.A = get_A_matrix_dwt_analysisFB_unit(self.h0,self.h1,self.N)
        super(IDWT2D, self).build(input_shape) 

    def call(self, inputs):
        __ = self.__splitch_for_idwt(inputs)
        __ = [self.__rejoin(_) for _ in __]# axis=-1)

        out = tf.stack([self.__idwt(_) for _ in __], axis=-1)
        # out = ch_
        # L = self.L
        return out#[:,L:-L,L:-L,:]

    @tf.function
    def __idwt(self, x_dwtcoeffs):
        _LLLHHLHH = x_dwtcoeffs
        A = tf.cast(self.A, tf.float32)
        _ = tf.transpose(_LLLHHLHH,[0,2,1])
        _AT = tf.cast(tf.transpose(A,[1,0]), dtype=tf.float32)
        _AT.shape, _.shape
        _r = tf.einsum('ij,bjk->bik', _AT, _)
        _rT = tf.transpose(_r,[0,2,1])
        _r = tf.einsum('ij,bjk->bik', _AT, _rT)
        # plt.imshow(_r)
        # _LLLHHLHH.dtype, _AT.dtype, _r.shape
        # print('+>',_r.shape)
        return _r

    def __splitch_for_idwt(self,x):
        w = int(x.shape[-1]/4)
        __ = [tf.stack([x[:,:,:,k], x[:,:,:,k+1*w], x[:,:,:,k+2*w], x[:,:,:,k+3*w]],  axis=-1) for k in range(w)]
        return __
    #__ = __splitch_for_idwt(coeffs)

    def __rejoin(self,subbands):
        el = tf.unstack(subbands, axis=-1)
        _1 = tf.concat([el[0][:,:,:], el[2][:,:,:]], axis=-1)
        _1.shape
        _2 = tf.concat([el[1][:,:,:], el[3][:,:,:]], axis=-1)
        _1.shape
        _ = tf.concat([_1, _2], axis=-2)
        _.shape
        return _
    #tf.stack([rejoin(_) for _ in __], axis=-1)

    def compute_output_shape(self, input_shape):
        batch_size, height, width, channels = input_shape
        new_height, new_width, new_channels = height *2, width *2, channels //4
        return (batch_size, new_height, new_width, new_channels)
   
    def get_config(self):
        config = super(IDWT2D, self).get_config()
        return config
    
if __name__=='__main__':
    import pywt.data
    # Load image
    x1 = pywt.data.camera()

    import cv2
    x = cv2.imread(f'/home/k/brain.png',cv2.IMREAD_GRAYSCALE) #test.jpg
    print('raw x shape:', x.shape)
    x = cv2.resize(x, (512,512))
    print('x shape:', x.shape)
    #x = x/np.max(x)
    plt.imshow(x,label='$x$')
    plt.title('input $x$')

    x.shape
    xnew = tf.expand_dims(tf.expand_dims(x, axis=-1), axis=0)
    xnew.shape

    xnew1 = tf.expand_dims(tf.expand_dims(x1, axis=-1), axis=0)
    xnew1.shape
    # del x, x1

    xnew = tf.cast(xnew, dtype=tf.float32)
    xnew1 = tf.cast(xnew1, dtype=tf.float32)
    _1 = tf.concat([xnew,xnew1,xnew],axis=-1)
    _2 = tf.concat([xnew1,xnew,xnew1],axis=-1)
    xnew = tf.cast(tf.concat([_1,_2], axis=0), dtype=tf.float32)
    _1.shape,_2.shape, xnew.shape, _1.dtype,_2.dtype, xnew.dtype
    del _1, _2



    wave = 'haar'
    wave = 'db6'
    wave = 'bior3.1'
    coeffs = DWT2D(wave=wave)(xnew)
    print(coeffs.shape)
    import matplotlib.pyplot as plt
    _ = coeffs
    for b in range(_.shape[0]):
        plt.figure(figsize=(16,8))
        for c in range(_.shape[-1]):
            plt.subplot(1, _.shape[-1], c+1), plt.imshow(_[b,:,:,c]), plt.title(f'bat.{b}, ch.{c}')
        plt.show()    

    out = IDWT2D(wave=wave)(coeffs)
    # [_.shape for _ in out]
    print(out.shape, out.dtype)
    import matplotlib.pyplot as plt
    _ = out
    for b in range(_.shape[0]):
        plt.figure(figsize=(12,4))
        for c in range(_.shape[-1]):
            plt.subplot(1, _.shape[-1], c+1), plt.imshow(_[b,:,:,c]), plt.title(f'bat.{b}, ch.{c}')
        plt.show()
            






