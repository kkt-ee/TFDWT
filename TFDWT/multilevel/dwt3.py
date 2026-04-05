import tensorflow as tf
# from keras import regularizers
from keras import ops
from TFDWT.DWT3DFB import DWT3D, IDWT3D
from tensorflow.keras.layers import Concatenate


def dwt3(x, level=3, Ψ='haar'):
    """ Multilevel 3D DWT
    
        TFDWT: Fast Discrete Wavelet Transform TensorFlow Layers.
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
    """
    subbands = []
    current = x
    channels_in = x.shape[-1]

    for _ in range(level):
        w = DWT3D(wave=Ψ)(current)
        lowpass = w[:, :, :, :, :channels_in]
        highpass = w[:, :, :, :, channels_in:]
        subbands.append(highpass)
        current = lowpass
    subbands.append(current)
    return subbands



def idwt3(subbands, level=3, Ψ='haar'):
    """ Multilevel 3D IDWT
    
        TFDWT: Fast Discrete Wavelet Transform TensorFlow Layers.
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
    """
    *highpasses, lowpass = subbands  # unpack: [H1, H2, ..., Hn, ln]
    
    current = lowpass
    for H in reversed(highpasses):
        current = IDWT3D(wave=Ψ)(Concatenate()([current, H]))
    
    return current


if __name__=='__main__':
    batch_size, N, channels = 1, 32, 2
    x = tf.random.normal((batch_size, N, N, N, channels))
    x.shape
    subbands = dwt3(x, level=level)
    print([_.shape for _ in subbands])
    x_rec = idwt3(subbands, level=level)
    print(np.allclose(x,x_rec, atol=1e-5))
