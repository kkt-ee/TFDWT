# TFDWT: Fast Discrete Wavelet Transform TensorFlow Layers

[![PyPI Version](https://img.shields.io/pypi/v/TFDWT?label=PyPI)](https://pypi.org/project/TFDWT/) 
[![PyPI Version](https://img.shields.io/pypi/pyversions/TFDWT)](https://pypi.org/project/TFDWT/)
[![TensorFlow Version](https://img.shields.io/badge/tensorflow-2.18.0-darkorange)](https://www.tensorflow.org/)
[![Keras Version](https://img.shields.io/badge/keras-3.6.0-darkred)](https://keras.io/)
[![CUDA Version](https://img.shields.io/badge/cuda-12.5.1-green)](https://developer.nvidia.com/cuda-toolkit)
[![NumPy Version](https://img.shields.io/badge/numpy-2.0.2-blueviolet)](https://numpy.org/)
[![MIT](https://img.shields.io/badge/license-GPLv3-deepgreen.svg?style=flat)](https://github.com/kkt-ee/TFDWT/LICENSE)

Fast $1\text{D}$, $2\text{D}$ and $3\text{D}$ Discrete Wavelet Transform ($\text{DWT}$) and Inverse Discrete Wavelet Transform ($\text{IDWT}$) layers for backpropagation networks.

**Available wavelet families ―**

```txt
        Haar (haar)
        Daubechies (db)
        Symlets (sym)
        Coiflets (coif)
        Biorthogonal (bior)
        Reverse biorthogonal (rbio)
```


**Note ―** The permissible input shapes are $2^m$, where $m$ is a Natural number. Additionally, for 2D and 3D transforms, the shape of the input must be square and cube, i.e., equal sides.

  
<br/><br/><br/>

* * *

## Installation guide

*The installation of the TFDWT package is recommended inside a virtual environment with tensorflow[GPU] installed at first.*

<br/>

**Install from PyPI** (Option $1$)

```bash
pip install TFDWT
```

  
<br/><br/>

**Install from Github** (Option $2$)

Download the package

```bash
git clone https://github.com/kkt-ee/TFDWT.git
```

Change directory to the downloaded TFDWT 

```bash
cd TFDWT
```

Run the following command to install the TFDWT package

```bash
pip install .
```



  
<br/><br/><br/>

* * *

## Verify installation

### Compute $\text{DWT}$ $1\text{D}$ and $\text{IDWT}$ $1\text{D}$ of batched, multichannel $x$ of shape $(\text{batch, length, channels})$

```python
"""Perfect Reconstruction 1D DWT level-1 Filter bank"""
from TFDWT.DWTIDWT1Dv1 import DWT1D, IDWT1D

LH = DWT1D(wave='bior3.1')(x)       # Analysis
x_hat = IDWT1D(wave='bior3.1')(LH)  # Synthesis

```

  <br/><br/>

### Compute $\text{DWT}$ $2\text{D}$ and $\text{IDWT}$ $2\text{D}$ of batched, multichannel $x$ of shape $(\text{batch, height, width, channels})$

```python
"""Perfect Reconstruction 2D DWT level-1 Filter bank"""
from TFDWT.DWTIDWT2Dv1 import DWT2D, IDWT2D

LLLHHLHH = DWT2D(wave=wave)(x)      # Analysis
x_hat = IDWT2D(wave=wave)(LLLHHLHH) # Synthesis

```

 <br/><br/>

 ### Compute $\text{DWT}$ $3\text{D}$ and $\text{IDWT}$ $3\text{D}$ of batched, multichannel $x$ of shape $(\text{batch, height, width, depth, channels})$

```python
"""Perfect Reconstruction 3D DWT level-1 Filter bank"""
from TFDWT.DWTIDWT3Dv1 import DWT3D, IDWT3D

LLLLLHLHLLHHHLLHLHHHLHHH = DWT3D(wave=wave)(x)      # Analysis
x_hat = IDWT3D(wave=wave)(LLLLLHLHLLHHHLLHLHHHLHHH) # Synthesis

```

 <br/><br/><br/>

***NOTE ―*** Using the above forward and inverse transforms the above $\text{DWT}$ and $\text{IDWT}$ layers can be used to construct multilevel $\text{DWT}$ filter banks and $\text{Wave Packet Transform}$ filter banks.



  
<br/><br/><br/>



* * *

## Package is tested with dependency versions

```txt
        Python 3.12.7
        TensorFlow 2.18.0
        Keras 3.6.0
        Numpy 2.0.2
        CUDA 12.5.1
```


<br/><br/><br/>

* * *

## Uninstall TFDWT

```bash
pip uninstall TFDWT
```

  
<br/><br/><br/><br/><br/>

* * *

***TFDWT (C) 2025 Kishore Kumar Tarafdar, Prime Minister's Research Fellow, EE, IIT Bombay, भारत*** 🇮🇳