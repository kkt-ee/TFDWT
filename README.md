# TFDWT: Fast Discrete Wavelet Transform TensorFlow Layers [![arXiv](https://img.shields.io/badge/arXiv-2504.04168-b31b1b.svg)](https://doi.org/10.48550/arXiv.2504.04168)

[![PyPI Version](https://img.shields.io/pypi/v/TFDWT?label=PyPI&color=gold)](https://pypi.org/project/TFDWT/) 
[![PyPI Version](https://img.shields.io/pypi/pyversions/TFDWT)](https://pypi.org/project/TFDWT/)
[![TensorFlow Version](https://img.shields.io/badge/tensorflow-2.15--2.19-darkorange)](https://www.tensorflow.org/)
[![Keras Version](https://img.shields.io/badge/keras-2--3-darkred)](https://keras.io/)
[![CUDA Version](https://img.shields.io/badge/cuda-12.5.1-green)](https://developer.nvidia.com/cuda-toolkit)
[![NumPy Version](https://img.shields.io/badge/numpy-2.0.2-blueviolet)](https://numpy.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-deepgreen.svg?style=flat)](LICENSE)




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


**Note ― Shape requirements**

- Single‑level: dimensions must be even. For 1D, `N` is even; for 2D and 3D, each side is even and the input is square/cubic respectively.
- Multilevel (L levels): each side must be divisible by $2^L$. In practice, pad to the nearest multiple of $2^L$ when needed.
- Our examples and tests center‑pad rather than crop, to preserve data.





<br/><br/><br/>

* * *

## Installation guide

*The installation of the ```TFDWT``` package is recommended inside a virtual environment with ```tensorflow[and-cuda]``` installed at first.*

<br/>

**Pre-installation checks**
(Tested in Gentoo and Debian bookworm)
  - Create a new **virtual enironment** with a specific **Python version** (use the Python version supported by TensorFlow)

  ```bash
  conda info --envs
  env_name='tf219'
  conda create -n $env_name python=3.12 ipykernel

  # activate virtual environment
  conda activate tf219
  ```

  - Install TensorFlow using official guide in https://www.tensorflow.org/install/pip  

  ```bash
  # For GPU users
  pip install tensorflow[and-cuda]
  # Verify setup
  python3 -c "import tensorflow as tf; print(f'{tf.config.list_physical_devices('GPU')}, \nTF version {tf.__version__}')"
  ```


<br/><br/>

**Install TFDWT from PyPI** (Option $1$)

```bash
pip install TFDWT
```

  
<br/><br/>

**Install TFDWT from Github** (Option $2$)

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
from TFDWT.DWT1DFB import DWT1D, IDWT1D

LH = DWT1D(wave='bior3.1')(x)       # Analysis
x_hat = IDWT1D(wave='bior3.1')(LH)  # Synthesis

```

<br/><br/>

### Compute $\text{DWT}$ $2\text{D}$ and $\text{IDWT}$ $2\text{D}$ of batched, multichannel $x$ of shape $(\text{batch, height, width, channels})$

```python
"""Perfect Reconstruction 2D DWT level-1 Filter bank"""
from TFDWT.DWT2DFB import DWT2D, IDWT2D

LLLHHLHH = DWT2D(wave='bior1.3')(x)        # Analysis
x_hat = IDWT2D(wave='bior1.3')(LLLHHLHH)   # Synthesis
# `wave` can be any supported orthogonal or biorthogonal wavelet

```

<br/><br/>

### Compute $\text{DWT}$ $3\text{D}$ and $\text{IDWT}$ $3\text{D}$ of batched, multichannel $x$ of shape $(\text{batch, height, width, depth, channels})$

```python
"""Perfect Reconstruction 3D DWT level-1 Filter bank"""
from TFDWT.DWT3DFB import DWT3D, IDWT3D

LLLLLHLHLLHHHLLHLHHHLHHH = DWT3D(wave='bior1.3')(x)        # Analysis
x_hat = IDWT3D(wave='bior1.3')(LLLLLHLHLLHHHLLHLHHHLHHH)   # Synthesis
# `wave` can be any supported orthogonal or biorthogonal wavelet

```

<br/><br/><br/>

***NOTE ―*** Using the above forward and inverse transforms the above $\text{DWT}$ and $\text{IDWT}$ layers can be used to construct multilevel $\text{DWT}$ filter banks and $\text{Wave Packet Transform}$ filter banks.

### Multilevel helpers (convenience API)

The package also provides simple helpers for building multilevel pyramids using the single‑level layers internally.

1D
```python
from TFDWT.multilevel.dwt import dwt, idwt

level = 3
subbands = dwt(x, level=level, Ψ='haar')   # returns [H1, H2, ..., HL, LL]
x_hat    = idwt(subbands, level=level, Ψ='haar')
```

2D
```python
from TFDWT.multilevel.dwt2 import dwt2, idwt2

level = 3
subbands = dwt2(x, level=level, Ψ='haar')  # [H1, H2, ..., HL, LL]
x_hat    = idwt2(subbands, level=level, Ψ='haar')
```

3D
```python
from TFDWT.multilevel.dwt3 import dwt3, idwt3

level = 3
subbands = dwt3(x, level=level, Ψ='haar')  # [H1, H2, ..., HL, LL]
x_hat    = idwt3(subbands, level=level, Ψ='haar')
```

Each `Hi` contains all high‑pass subbands at level `i` (1 for 1D, 3 for 2D, 7 for 3D), concatenated along the channel axis; the last element is the final low‑pass `LL`.



  
<br/><br/><br/>



* * *

## Package is tested with dependency versions

```txt
        Python 3.12.7
        TensorFlow 2.15 to 2.19
        Keras 2 and 3
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

***TFDWT (C) 2026 Kishore Kumar Tarafdar, भारत*** 🇮🇳
