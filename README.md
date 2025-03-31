# TFDWT: Fast Discrete Wavelet Transform TensorFlow Layers

Fast $1\text{D}$ and $2\text{D}$ Discrete Wavelet Transform ($\text{DWT}$) and Inverse Discrete Wavelet Transform ($\text{IDWT}$)

**Available wavelet families ―**

```txt
        Haar (haar)
        Daubechies (db)
        Symlets (sym)
        Coiflets (coif)
        Biorthogonal (bior)
        Reverse biorthogonal (rbio)
```


  
<br/><br/><br/>

* * *

## Installation guide

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

## Test and verify installations

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

**NOTE ―** Using the above forward and inverse transforms the above $\text{DWT}$ and $\text{IDWT}$ layers can be used to construct multilevel $\text{DWT}$ filter banks and $\text{Wave Packet}$ filter banks.



  
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

<br/><br/>

***The installation of the TFDWT package is recommended inside a virtual environment with tensorflow[GPU] installed at first***

<br/><br/><br/>

* * *

## Uninstall TFDWT

```bash
pip uninstall TFDWT
```

  
<br/><br/><br/><br/><br/>

* * *

***TFDWT (C) 2025 Kishore Kumar Tarafdar, Prime Minister's Research Fellow, EE, IIT Bombay, भारत*** 🇮🇳