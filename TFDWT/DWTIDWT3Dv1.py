"""3D DWT and IDWT layers

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
class DWT3D(tf.keras.layers.Layer):
    """3D DWT layer 

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
        super(DWT3D, self).__init__(**kwargs)
        w = GETDWTFiltersOrtho(wave)
        self.h0, self.h1 = w.analysis()
        self.L = len(self.h0)
        
    def build(self, input_shape):
        self.num_channels = input_shape[-1]
        self.N = input_shape[1]
        self.A = get_A_matrix_dwt_analysisFB_unit(self.h0,self.h1,self.N)
        # self.pw = (self.A.shape[0]-self.N)//2
        super(DWT3D, self).build(input_shape) 
       
    
    def call(self, inputs):
        ch_ = [self.__dwt(inputs[..., i:i+1]) for i in range(self.num_channels)]
        out = tf.concat(ch_, axis=-1)
        #
        out = self.__extract_8subbands(out)
        return out

    # @tf.function
    def __dwt(self, x):
        _x3d = tf.squeeze(x, axis=-1)
        A = tf.cast(self.A, tf.float32)
        ### DWT 3D core
        # _x3d = tf.cast(x3d, dtype=tf.float32)
        # _x3d = tf.transpose(_x3d,[1,0,2])
        # _rowconv3dout = tf.einsum('ij,jkl->ikl', A, _x3d)
        # _rowconv3doutT = tf.transpose(_rowconv3dout,[1,0,2])
        # _colconv3dout = tf.einsum('ij,jkl->ikl', A, _rowconv3doutT)
        # _colconv3doutT = tf.transpose(_colconv3dout,[0,2,1]) #!!
        # _depthconv3dout = tf.einsum('ik,jkl->jil', A, _colconv3doutT) ##!!
        # _dwt3dout = tf.transpose(_depthconv3dout,[0,2,1])

        ## batched inputs
        _x3d = tf.transpose(_x3d,[0,2,1,3])
        _rowconv3dout = tf.einsum('ij,bjkl->bikl', A, _x3d)
        _rowconv3doutT = tf.transpose(_rowconv3dout,[0,2,1,3])
        _colconv3dout = tf.einsum('ij,bjkl->bikl', A, _rowconv3doutT)
        _colconv3doutT = tf.transpose(_colconv3dout,[0,1,3,2]) #!!
        _depthconv3dout = tf.einsum('ik,bjkl->bjil', A, _colconv3doutT) ##!!
        _dwt3dout = tf.transpose(_depthconv3dout,[0,1,3,2])
        return tf.expand_dims(_dwt3dout, axis=-1)

    def __extract_8subbands(self, LLLLLHLHLLHHHLLHLHHHLHHH):
        _mid = int(LLLLLHLHLLHHHLLHLHHHLHHH.shape[2]/2)
        # print('++',_mid)
        # print('+++',LLLLLHLHLLHHHLLHLHHHLHHH.shape)
        ## front half 4 subbands
        LLL = LLLLLHLHLLHHHLLHLHHHLHHH[:,:_mid,:_mid,:_mid,:]
        LLH = LLLLLHLHLLHHHLLHLHHHLHHH[:,_mid:,:_mid,:_mid,:]
        LHL = LLLLLHLHLLHHHLLHLHHHLHHH[:,:_mid,_mid:,:_mid,:]
        LHH = LLLLLHLHLLHHHLLHLHHHLHHH[:,_mid:,_mid:,:_mid,:]
        ## back half 4 subbands
        HLL = LLLLLHLHLLHHHLLHLHHHLHHH[:,:_mid,:_mid,_mid:,:]
        HLH = LLLLLHLHLLHHHLLHLHHHLHHH[:,_mid:,_mid:,_mid:,:]
        HHL = LLLLLHLHLLHHHLLHLHHHLHHH[:,:_mid,_mid:,_mid:,:]
        HHH = LLLLLHLHLLHHHLLHLHHHLHHH[:,_mid:,_mid:,_mid:,:]

        # _out = tf.stack([LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH], axis=-1)
        _out = tf.concat([LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH], axis=-1)
        return _out


    # def compute_output_shape(self, input_shape):
    #     batch_size, height, width, channels = input_shape
    #     new_height, new_width = height // 2, width // 2
    #     return (batch_size, new_height, new_width, 4 * channels)
    
    def get_config(self):
        config = super(DWT3D, self).get_config()
        return config




#%% OK
# ## WORKING IDWT similar to wavetf
# from GETDWTFiltersOrtho import GETDWTFiltersOrtho
@keras.saving.register_keras_serializable()
class IDWT3D(tf.keras.layers.Layer):
    """3D IDWT layer 
    
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
        super(IDWT3D, self).__init__(**kwargs)
        w = GETDWTFiltersOrtho(wave)
        if 'bior' in wave or 'rbio' in wave:
            """BIORTHOGONAL wavelets"""
            # self.g0, self.g1 = w.synthesis()
            # self.h0, self.h1 = self.g0, self.g1
            # del self.g0, self.g1
            self.h0, self.h1 = w.synthesis()
            # self.biortho_flag = 1
            print(f"Biothogonal wavelet {wave}")
        else:
            """ORTHOGONAL wavelets"""
            self.h0, self.h1 = w.analysis()
        # g0, g1 = w.synthesis()
        self.L = len(self.h0)
        # self.padding_mode = 'symmetric'

    def build(self, input_shape):
        # self.num_channels = input_shape[-1]
        self.N = int(input_shape[1]*2)
        self.A = get_A_matrix_dwt_analysisFB_unit(self.h0,self.h1,self.N)
        # self.pw = int(self.L/2)
        super(IDWT3D, self).build(input_shape) 

    def call(self, inputs):
        # self.N = int(inputs.shape[1]*2)
        # self.A = get_A_matrix_dwt_analysisFB_unit(self.h0,self.h1,self.N)
        # input_shape = tf.shape(inputs)

        #all input channels
        # ch_ = tf.unstack(inputs, axis=-1)
        # print(len(ch_))
        # ch_ = tf.stack([self.__pad(_) for _ in ch_], axis=-1)
        # __ = self.__splitch_for_idwt(ch_)
        __ = self.__splitch_for_idwt(inputs)
        # print('>++', len(__), __[0].shape)
        __ = [self.__rejoin(_) for _ in __]# axis=-1)
        # print('<>++', len(__), __[0].shape)

        out = tf.stack([self.__idwt(_) for _ in __], axis=-1)
        # out = ch_
        # L = self.L
        return out#[:,L:-L,L:-L,:]

    @tf.function
    def __idwt(self, x_dwtcoeffs):

        _LLLLLHLHLLHHHLLHLHHHLHHH = x_dwtcoeffs
        A = A = tf.cast(self.A, tf.float32)
        ## core        
        # _dwt3dout.shape
        # _LLLLLHLHLLHHHLLHLHHHLHHH = _dwt3dout ## !!
        # _AT = tf.cast(tf.transpose(A,[1,0]), dtype=tf.float32)

        # _ = tf.transpose(_LLLLLHLHLLHHHLLHLHHHLHHH,[0,2,1])
        # _ = tf.einsum('ik,jkl->jil', _AT, _) ##!!
        # _ = tf.transpose(_,[0,2,1])
        # _ = tf.einsum('ij,jkl->ikl', _AT, _)
        # _ = tf.transpose(_,[1,0,2])
        # _ = tf.einsum('ij,jkl->ikl', _AT, _)
        # _r3d = tf.transpose(_,[1,0,2])
        # _r3d.shape

        # _dwt3dout.shape
        # _LLLLLHLHLLHHHLLHLHHHLHHH = _dwt3dout ## !!
        _AT = tf.cast(tf.transpose(A,[1,0]), dtype=tf.float32)
        # print('>>',_LLLLLHLHLLHHHLLHLHHHLHHH.shape, _AT.shape, _AT.dtype)

        _ = tf.transpose(_LLLLLHLHLLHHHLLHLHHHLHHH,[0,1,3,2])
        _ = tf.einsum('ik,bjkl->bjil', _AT, _) ##!!
        _ = tf.transpose(_,[0,1,3,2])
        # print('++<>', _.shape, _.dtype, _AT.shape, _AT.dtype)
        _ = tf.einsum('ij,bjkl->bikl', _AT, _)
        # print('++<>')
        _ = tf.transpose(_,[0,2,1,3])
        _ = tf.einsum('ij,bjkl->bikl', _AT, _)
        _r3d = tf.transpose(_,[0,2,1,3])
        # _r3d.shape
        return _r3d

    def __splitch_for_idwt(self,x):
        w = x.shape[-1]//8
        __ = [tf.stack([x[:,:,:,:,k], x[:,:,:,:,k+1*w], x[:,:,:,:,k+2*w], x[:,:,:,:,k+3*w],x[:,:,:,:,k+4*w],x[:,:,:,:,k+5*w],x[:,:,:,:,k+6*w],x[:,:,:,:,k+7*w]],  axis=-1) for k in range(w)]
        # print('>>>', len(__), __[0].shape)
        return __
    # __ = __splitch_for_idwt(coeffs)

    def __rejoin(self,subbands):

        ### the correct format for rejoining 3d
        unstackedbands = tf.unstack(subbands, axis=-1)
        _LLL, _LLH, _LHL, _LHH, _HLL, _HLH, _HHL, _HHH = unstackedbands
        # __unst = _LLL, _LLH, _LHL, _LHH, _HLL, _HLH, _HHL, _HHH
        _1 = tf.concat([_LLL, _LLH], axis=-3)
        # print(_1.shape)
        _2 = tf.concat([_LHL, _LHH], axis=-3)
        # print(_2.shape)
        _12 = tf.concat([_1, _2], axis=2)
        # print(_12.shape)

        _3 = tf.concat([_HLL, _HLH], axis=-3)
        # print(_3.shape)
        _4 = tf.concat([_HHL, _HHH], axis=-3)
        # print(_4.shape)
        _34 = tf.concat([_3, _4], axis=2)
        # print(_34.shape)

        _1234 = tf.concat([_12, _34], axis=3)
        # print(_1234.shape)
        return _1234

    # def compute_output_shape(self, input_shape):
    #     batch_size, height, width, channels = input_shape
    #     new_height, new_width, new_channels = height *2, width *2, channels //4
    #     return (batch_size, new_height, new_width, new_channels)
   
    def get_config(self):
        config = super(IDWT3D, self).get_config()
        return config



if __name__=='__main__':
    wavelet = 'db3'
    wavelet = 'bior3.1'

    import numpy as np
    import matplotlib.pyplot as plt
    plt.rcParams['font.size'] = '18'
    from scipy import ndimage
    #from sympy import Matrix


    """
    x = np.random.rand(input_shape[0],input_shape[1])
    plt.figure(figsize=(15,2))
    plt.imshow(x, label='$x$')
    plt.title('$x$')
    """
    import pywt.data
    # Load image
    x1 = pywt.data.camera()

    import cv2
    x = cv2.imread(f'/home/k/brain.png',cv2.IMREAD_GRAYSCALE) #test.jpg
    print('raw x shape:', x.shape)
    # x = cv2.resize(x, (512,512))
    x = cv2.resize(x, (256,256))

    print('x shape:', x.shape)
    #x = x/np.max(x)
    plt.imshow(x,label='$x$')
    plt.title('input $x$')

    x_bak = x

    # global N
    N = x.shape[0]
    input_shape = (N,N,1)


    import tensorflow as tf
    import matplotlib
    import scipy
    import pywt

    x3d = np.concatenate([np.expand_dims(x, axis=-1) for i in range(x.shape[0])], axis=-1)
    print(f'Layer: 3D input x shape is {x3d.shape}')
    _x3d = tf.cast(x3d, dtype=tf.float32)
    _x3dnew = tf.expand_dims(tf.expand_dims(_x3d, axis=-1), axis=0)
    _x3dnew.shape

    # signal 2 with multiple channels
    _x3dnew = tf.concat([_x3dnew,_x3dnew], axis=-1)
    _x3dnew = tf.concat([_x3dnew,_x3dnew], axis=0)
    print(f'Layer: new 3D input x shape is {_x3dnew.shape}')




    


    coeffs = DWT3D(wave=wavelet)(_x3dnew)
    print(coeffs.shape)
    i = np.random.randint(128)
    plt.imshow(coeffs[0,:,:,i,4]), plt.title(f"slice {i}")
    plt.show()

    _r3d = IDWT3D(wave=wavelet)(coeffs)
    _r3d.shape

    i = np.random.randint(256)
    plt.imshow(_r3d[0,:,:,i,0]), plt.title(f"slice {i}")








    # import pywt.data
    # # Load image
    # x1 = pywt.data.camera()

    # import cv2
    # x = cv2.imread(f'/home/k/brain.png',cv2.IMREAD_GRAYSCALE) #test.jpg
    # print('raw x shape:', x.shape)
    # x = cv2.resize(x, (512,512))
    # print('x shape:', x.shape)
    # #x = x/np.max(x)
    # plt.imshow(x,label='$x$')
    # plt.title('input $x$')

    # x.shape
    # xnew = tf.expand_dims(tf.expand_dims(x, axis=-1), axis=0)
    # xnew.shape

    # xnew1 = tf.expand_dims(tf.expand_dims(x1, axis=-1), axis=0)
    # xnew1.shape
    # # del x, x1

    # xnew = tf.cast(xnew, dtype=tf.float32)
    # xnew1 = tf.cast(xnew1, dtype=tf.float32)
    # _1 = tf.concat([xnew,xnew1,xnew],axis=-1)
    # _2 = tf.concat([xnew1,xnew,xnew1],axis=-1)
    # xnew = tf.cast(tf.concat([_1,_2], axis=0), dtype=tf.float32)
    # _1.shape,_2.shape, xnew.shape, _1.dtype,_2.dtype, xnew.dtype
    # del _1, _2




    # coeffs = DWT2D(wave='db6')(xnew)
    # print(coeffs.shape)
    # import matplotlib.pyplot as plt
    # _ = coeffs
    # for b in range(_.shape[0]):
    #     plt.figure(figsize=(16,8))
    #     for c in range(_.shape[-1]):
    #         plt.subplot(1, _.shape[-1], c+1), plt.imshow(_[b,:,:,c]), plt.title(f'bat.{b}, ch.{c}')
    #     plt.show()    

    # out = IDWT2D(wave='db6')(coeffs)
    # # [_.shape for _ in out]
    # print(out.shape, out.dtype)
    # import matplotlib.pyplot as plt
    # _ = out
    # for b in range(_.shape[0]):
    #     plt.figure(figsize=(12,4))
    #     for c in range(_.shape[-1]):
    #         plt.subplot(1, _.shape[-1], c+1), plt.imshow(_[b,:,:,c]), plt.title(f'bat.{b}, ch.{c}')
    #     plt.show()
            







    



