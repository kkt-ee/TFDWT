---
title: 'TFDWT: Fast Discrete Wavelet Transform TensorFlow Layers'
tags:
  - Python
  - Tensorflow
  - Discrete Wavelet Transform
  - Multiresolution analysis
  - Multirate systems
  - Deep Learning
  - CUDA
  - Signal processing
authors:
  - name: Kishore K. Tarafdar
    orcid: 0009-0001-4548-7639
  - name: Vikram M. Gadre
    orcid: 0000-0001-8439-5625

affiliations:
 - name: Department of Electrical Engineering, Indian Institute of Technology Bombay, India
date: 1 April 2025
bibliography: paper.bib
---


# 1 Summary

TFDWT is an open-source Python library that allows the construction of TensorFlow Layers for Fast Discrete Wavelet Transform (DWT) and Inverse Discrete Wavelet Transform (IDWT) in end-to-end backpropagation learning networks. These layers facilitate the construction of multilevel DWT filter banks and Wavelet Packet Transform (WPT) filter banks for a spatial-frequency representation of the inputs and features in shallow or deep networks. A multiresolution signal representation using a multi-rate discrete wavelet system creates enriched joint natural-frequency representations. The discrete wavelet system partitions the frequency plane into subbands using orthogonal dilated and translated lowpass scaling and highpass wavelet function. A realization of a fast discrete wavelet system is a two-band perfect reconstruction multi-rate filter bank with FIR filters corresponding to the impulse responses of the scaling and wavelet function with downsampling and upsampling operations. A filter bank for a higher dimensional input is a seamless extension by successive separable circular convolutions across each independent axis. The command `pip install TFDWT` installs the latest version of the package. 

# 2 Statement of need

In machine or deep learning, an efficient multiresolution representation of data often helps to build economical and explainable models. The Wavelet toolbox [@misiti1996wavelet] by MathWorks is a proprietary software that has served the requirements for $\text{D}$-dimensional wavelet transforms in the MATLAB environment for a few decades. Several open-source packages are now available for 1D and 2D DWT in Python. Pywavelets [@lee2019pywavelets] is a $\text{D}$-dimensional wavelet transform library in Python that works with Numpy [@harris2020array] arrays. However, it is challenging to directly use Pywavelets with the symbolic tensors in TensorFlow [@developers2022tensorflow] layers and CUDA [@fatica2008cuda]. WaveTF [@versaci2021wavetf] is a solution for constructing 1D and 2D DWT layers in TensorFlow but is limited to only Haar and Db2 wavelets. The package tensorflow-wavelets [@tensorflow-wavelets-1.1.2] supports multiple wavelets, but it has a minor bug in perfect reconstruction due to the padding and boundary effects in processing the finite-length inputs. In Pytorch [@imambi2021pytorch], the pytorch-wavelets [@pytorch-wavelets-1.3.0] package allows the construction of 1D and 2D DWT layers. However, there are limited libraries for 3D and higher dimensional transforms with a wide range of wavelet families for Graphics Processing Unit (GPU) computations. 

For a $\text{{D}}$\-dimensional wavelet $\boldsymbol{\psi} \in L^{2}(\mathbb{R})^{\text{{D}}}$, a discrete wavelet system defined by $\big\{\boldsymbol{\psi}_{m,\boldsymbol{p}}:m\in\mathbb{Z},\boldsymbol{p}\in\mathbb{Z}^{\text{{D}}},\text{D}\in\mathbb{N}\big\}$ forms an orthonormal basis, where $\boldsymbol{\psi}_{m,\boldsymbol{p}}(\boldsymbol{x}):=2^{m}\psi(2^{m}\boldsymbol{x}-\boldsymbol{p})$. Then, by definition the DWT of $\boldsymbol{x}\in\mathbb{Z}^{\text{{D}}}$ is $\boldsymbol{x}\mapsto(\langle\boldsymbol{x},\psi_{m,\boldsymbol{p}}\rangle)_{m,\boldsymbol{p}}$, where $m$ is the dilation parameter and $\boldsymbol{p}$ is the shift or translation parameter. The TFDWT Python package is a simple standalone DWT and IDWT library with minimal dependencies that allow computation with symbolic tensors and CUDA. This release supports up to 3D forward and inverse transforms with various orthogonal and biorthogonal wavelet families. A seamless package upgrade for higher dimensional DWT is possible by separable transforms of the independent axes. The boundary effects are taken care of with cyclic convolutions instead of padding. The package supports orthogonal and biorthogonal wavelets of different families having impulse responses of diverse lengths. In this paper, we defined the platform-independent, underlying mathematics for realizing fast $\text{D}$-dimensional DWT and IDWT layers with filter bank structures having FIR filters, downsamplers and upsamplers. Although our realization of the discrete wavelet system is in TensorFlow 2, a seamless reproduction of the computations is possible in other deep learning frameworks.


# 3 Discrete wavelet system for sequences

A discrete wavelet system contains a pair of quadrature mirror filters with a lowpass scaling function and a highpass mother wavelet. A discrete wavelet system with a continuous one-dimensional ($\text{D}=1$) mother wavelet $\psi\in L^{2}(\mathbb{R})$ is realized by using the impulse responses of the scaling function and the wavelet as Finite Impulse Response (FIR) filters $g\big[n\big]$ and $h\big[n\big]$. \autoref{fig:bior31} shows wavelets and scaling functions of the analysis and synthesis bank of bior3.1 wavelet and their corresponding impulse responses are shown in \autoref{fig:bior31impulse}. These FIR filters are the building blocks of a two-band perfect reconstruction filter bank for realizing fast discrete wavelet systems. \autoref{fig:2BPRFB} shows a two-band perfect reconstruction filter bank that operates on one-dimensional inputs, i.e., sequences in $l^{2}(\mathbb{\mathbb{Z}})$. The analysis and synthesis bank in orthogonal wavelet systems have identical lowpass and highpass FIR filters but differ in biorthogonal wavelet systems. In biorthogonal wavelet filter banks, $\tilde{g}\big[n\big]$ and $\tilde{h}\big[n\big]$ are the lowpass and highpass filters of the synthesis bank. The only difference in the biorthogonal families like bior and rbio is the interchange of the analysis and synthesis scaling and wavelets functions, for example, in bior3.1 and rbio3.1.

![Wavelets and scaling functions of bior3.1 analysis (left) and synthesis (right).\label{fig:bior31}](figs/waveBior31.png){ width=60% }

![Impulse responses of different bior3.1 analysis (left two) and synthesis (right two) lowpass and highpass FIR filters.\label{fig:bior31impulse}](figs/impulsebior31.png){ width=70% }

## 3.1 Circular convolution operators

The four matrices in the two band perfect reconstruction filter bank in \autoref{fig:2BPRFB} are — (i) $\boldsymbol{G}$ is lowpass analysis matrix, (ii) $\boldsymbol{H}$ is highpass analysis matrix, (iii) $\tilde{\boldsymbol{G}}$ is lowpass synthesis matrix and (iv) $\tilde{\boldsymbol{H}}$ is highpass synthesis matrix. These matrices are operators for circular convolution, constructed by circular shifts of the corresponding FIR filters $g\big[n-k\big]$, $h\big[n-k\big]$, $\tilde{g}\big[n-k\big]$ and $\tilde{h}\big[n-k\big]$, where $g=\tilde{g}$ and $h=\tilde{h}$ for orthogonal wavelets.

![Two band perfect reconstruction filter bank.\label{fig:2BPRFB}](figs/DWTFB.jpg){ width=50% }

### 3.1.1 Analysis matrix

For a lowpass analysis FIR filter of length $L$ and a input sequence of length $N$, the circular convolution operator is a matrix $\boldsymbol{G}$ of shape $N\times N$. A downsampling by two $f_{(\downarrow2)}$ of the output the convolution is equivalent to getting rid of the even rows of $\boldsymbol{G}$ to give $\frac{N}{2}\times{\small {N}}$ operator $f_{(\downarrow2)}\boldsymbol{G}$. Similarly, for the highpass analysis FIR filter of same wavelet with convolution and downsampling is a $\frac{N}{2}\times{\small {N}}$ operator $f_{(\downarrow2)}\boldsymbol{H}$. The analysis matrix is,  
$$
\boldsymbol{A}=\left[\begin{array}{c}
f_{(\downarrow2)}\boldsymbol{G}\\
f_{(\downarrow2)}\boldsymbol{H}
\end{array}\right]_{N\times N}
$$
where $\boldsymbol{A}$ is a $N\times N$ decimated circular convolution operator formed by combining the lowpass and highpass decimated operators].


### 3.1.2 Synthesis matrix

The synthesis matrix $\boldsymbol{S}$ is another $N\times N$ decimated circular convolution operator as given by equation \[eq:SynthesisMatrix\],

$$
\boldsymbol{S}=\left[\begin{array}{c}
f_{(\downarrow2)}\tilde{\boldsymbol{G}}\\
f_{(\downarrow2)}\tilde{\boldsymbol{H}}
\end{array}\right]_{N\times N}^{T}
$$

where $\tilde{\boldsymbol{G}}$ and $\tilde{\boldsymbol{H}}$ are matrices formed by the lowpass and highpass synthesis FIR filters. The above is a general representation for both orthogonal and biorthogonal wavelets families where for orthogonal wavelets, $\tilde{\boldsymbol{G}}=\boldsymbol{G}$, $\tilde{\boldsymbol{H}}=\boldsymbol{H}$ and thus $\boldsymbol{S}=\boldsymbol{A}^{T}$.

A two-band perfect reconstruction discrete wavelet system for one-dimensional inputs is given by the analysis equation and the synthesis equation,

$$
\boldsymbol{q}	=\text{{DWT}}\big(\boldsymbol{x}\big)\text{{ or, }}\boldsymbol{q}=\big(\boldsymbol{A}\boldsymbol{x}^{T}\big)^{T}
\text{     —Analysis}$$

$$
\boldsymbol{x}	=\text{{IDWT}}\big(\boldsymbol{q}\big)\text{{ or, }}\boldsymbol{x}=\big(\boldsymbol{S}\boldsymbol{q}^{T}\big)^{T}\text{—Synthesis} 
$$

where $\boldsymbol{A}$ and $\boldsymbol{S}$ are analysis and synthesis matrices, $\boldsymbol{x}$ is a input sequence and $\boldsymbol{q}$ has a distinct lowpass and a highpass subband.

**Example 1.** Given, a sequence $\boldsymbol{x}\in\mathbb{R}^{8}$ or $N=8$ and FIR filters length $L=6$.

$$
\text{{LPF and downsampling }}f_{(\downarrow2)}\boldsymbol{G}	={\small {\left[\begin{array}{cccccccc}
g_{1} & g_{0} & 0 & 0 & g_{5} & g_{4} & g_{3} & g_{2}\\
g_{3} & g_{2} & g_{1} & g_{0} & 0 & 0 & g_{5} & g_{4}\\
g_{5} & g_{4} & g_{3} & g_{2} & g_{1} & g_{0} & 0 & 0\\
0 & 0 & g_{5} & g_{4} & g_{3} & g_{2} & g_{1} & g_{0}
\end{array}\right]_{\frac{N}{2}\times{N}}}}
$$

$$
\text{{HPF and downsampling }}f_{(\downarrow2)}\boldsymbol{H}	={\small {\left[\begin{array}{cccccccc}
h_{1} & h_{0} & 0 & 0 & h_{5} & h_{4} & h_{3} & h_{2}\\
h_{3} & h_{2} & h_{1} & h_{0} & 0 & 0 & h_{5} & h_{4}\\
h_{5} & h_{4} & h_{3} & h_{2} & h_{1} & h_{0} & 0 & 0\\
0 & 0 & h_{5} & h_{4} & h_{3} & h_{2} & h_{1} & h_{0}
\end{array}\right]_{\frac{N}{2}\times{\small {N}}}}}
$$

$$
\text{{Analysis matrix is }}\boldsymbol{A}	={\small {\left[\begin{array}{cccccccc}
g_{1} & g_{0} & 0 & 0 & g_{5} & g_{4} & g_{3} & g_{2}\\
g_{3} & g_{2} & g_{1} & g_{0} & 0 & 0 & g_{5} & g_{4}\\
g_{5} & g_{4} & g_{3} & g_{2} & g_{1} & g_{0} & 0 & 0\\
0 & 0 & g_{5} & g_{4} & g_{3} & g_{2} & g_{1} & g_{0}\\
h_{1} & h_{0} & 0 & 0 & h_{5} & h_{4} & h_{3} & h_{2}\\
h_{3} & h_{2} & h_{1} & h_{0} & 0 & 0 & h_{5} & h_{4}\\
h_{5} & h_{4} & h_{3} & h_{2} & h_{1} & h_{0} & 0 & 0\\
0 & 0 & h_{5} & h_{4} & h_{3} & h_{2} & h_{1} & h_{0}
\end{array}\right]_{{N}\times{N}}}}
$$

Similarly,

$$
\text{{Synthesis matrix is }}\boldsymbol{S}	={\small {\left[\begin{array}{cccccccc}
\tilde{g}_{1} & \tilde{g}_{0} & 0 & 0 & \tilde{g}_{5} & \tilde{g}_{4} & \tilde{g}_{3} & \tilde{g}_{2}\\
\tilde{g}_{3} & \tilde{g}_{2} & \tilde{g}_{1} & \tilde{g}_{0} & 0 & 0 & \tilde{g}_{5} & \tilde{g}_{4}\\
\tilde{g}_{5} & \tilde{g}_{4} & \tilde{g}_{3} & \tilde{g}_{2} & \tilde{g}_{1} & \tilde{g}_{0} & 0 & 0\\
0 & 0 & \tilde{g}_{5} & \tilde{g}_{4} & \tilde{g}_{3} & \tilde{g}_{2} & \tilde{g}_{1} & \tilde{g}_{0}\\
\tilde{h}_{1} & \tilde{h}_{0} & 0 & 0 & \tilde{h}_{5} & \tilde{h}_{4} & \tilde{h}_{3} & \tilde{h}_{2}\\
\tilde{h}_{3} & \tilde{h}_{2} & \tilde{h}_{1} & \tilde{h}_{0} & 0 & 0 & \tilde{h}_{5} & \tilde{h}_{4}\\
\tilde{h}_{5} & \tilde{h}_{4} & \tilde{h}_{3} & \tilde{h}_{2} & \tilde{h}_{1} & \tilde{h}_{0} & 0 & 0\\
0 & 0 & \tilde{h}_{5} & \tilde{h}_{4} & \tilde{h}_{3} & \tilde{h}_{2} & \tilde{h}_{1} & \tilde{h}_{0}
\end{array}\right]_{{N}\times{N}}}}
$$

The DWT of $\boldsymbol{x}$ produces subbbands,

$$
\boldsymbol{q}=\text{{DWT}}\big(\boldsymbol{x}\big)\text{{ or, }}\boldsymbol{q=A}\boldsymbol{x}
$$

Perfect reconstruction,

$$
\boldsymbol{x}=\text{{IDWT}}\big(\boldsymbol{q}\big)\text{{ or, }}\boldsymbol{x}=\boldsymbol{S}\boldsymbol{q}
$$

## 3.2 DWT 1D layer

A DWT 1Dayer operates on input tensors of shape  $(\text{{batch, length, channels}})$ and produces an output of shape  $(\text{batch}, \text{length}/2, 2 \times \text{channels})$. as described in Algorithm 1a.

***Algorithm 1a —***

1.  Input $\boldsymbol{X}$ of shape  $(\text{{batch, length, channels}})$.
2.  Generate analysis matrix $\boldsymbol{A}$ using $\text{{length}}$ of input.
3.  For each batched channel $\boldsymbol{x}\in\boldsymbol{X}$ of shape $(\text{{batch, length}})$: $\boldsymbol{q}_{c}=\boldsymbol{A}\boldsymbol{x}^{T}$    
4.  Stacking for all $c$ channels: $\boldsymbol{Q}:=\big(\boldsymbol{q}_{c}\big)_{\forall c}$ to a shape $(\text{{batch, length, channels}})$.
5.  Group subbands and return an output $\boldsymbol{Q}^{(\text{{grouped}})}$ of shape $(\text{batch}, \text{length}/2, , 2\times\text{channels})$

	```python
		# Grouping two subbands in DWT 1D
		mid = int(Q.shape[1]/2)
		L = Q[:,:mid,:]
		H = Q[:,mid:,:]
		out = Concatenate([L, H], axis=-1)
	```
	
## 3.3 IDWT 1D layer

An IDWT 1D layer operates on input tensors of shape $(\text{batch}, \text{length}/2, 2 \times \text{channels})$ and produces an output of shape $(\text{{batch, length, channels}})$ as described in Algorithm 1b.

***Algorithm 1b —***

1.  Input $\boldsymbol{Q}^{(\text{{grouped}})}$ of shape  $(\text{batch}, \text{length}/2, 2\times\text{channels})$
2.  Ungroup the subbands to get $\boldsymbol{Q}$ of shape $(\text{{batch, length, channels}})$
3.  Generate synthesis matrix $\boldsymbol{S}$ using $\text{{length}}$ of $\boldsymbol{Q}$.
4.  For each batched channel $\boldsymbol{q}\in\boldsymbol{Q}$ of shape $(\text{{batch, length}})$:  $\boldsymbol{x}=\boldsymbol{S}\boldsymbol{q}^{T}$, i.e., a perfect reconstruction, where, $\boldsymbol{S}=\boldsymbol{A}^{T}$ for orthogonal   wavelets
5.  Layer output (perfect reconstruction): $\boldsymbol{X}:=\big(\boldsymbol{x}_{c}\big)_{\forall c}$ is of  shape $(\text{{batch, length, channels}})$


# 4 Higher dimensional discrete wavelet systems

In sequences (1D), the DWT 1D applies along the only independent variable. To achieve higher dimensional DWT, the same DWT 1D is successively applied separably to all the independent variables. For example, in an image (2D) the DWT 2D is a row-wise DWT 1D followed by a column-wise DWT 1D. Similarly, the reconstruction is column-wise IDWT 1D followed by a row-wise IDWT 1D.

## 4.1 Two-dimensional discrete wavelet system

The pixel values in an image are a function of two independent spatial axes. A DWT 2D filter bank is a separable transform with row-wise filtering followed by column-wise filtering that yields four subbands - LL, LH, HL and HH. A two-dimensional discrete wavelet system is,

$$
\boldsymbol{q}	=\text{{DWT}}\big(\boldsymbol{x}\big):=\boldsymbol{A}(\boldsymbol{A}\boldsymbol{x}_{021})_{021}^{T} \text{\hspace{2.1cm}—Analysis} 
$$

$$
\boldsymbol{x}	=\text{{IDWT}}\big(\boldsymbol{q}\big):=\boldsymbol{S}(\boldsymbol{S}\left[\boldsymbol{A}(\boldsymbol{A}\boldsymbol{x}_{021}^{T})_{021}^{T}\right]_{021}^{T})_{021}^{T}\text{\hspace{0.2cm}—Synthesis} 
$$

where, $\boldsymbol{A}$ and $\boldsymbol{S}$ are the same analysis and synthesis matrices defined for one-dimensional wavelet system. \autoref{fig:subdband2d} shows an $N\times N$ image and its four localized spatial-frequency subbands after DWT. Here, the low-frequency band is LL, and the other three are high-frequency subbands representing horizontal, vertical and diagonal features. \autoref{fig:2BPRFB-2} illustrates a separable 2D DWT perfect reconstruction filter bank. The 2D layers operate on batched, multichannel tensors of shape $(\text{{batch, height, width, channels}})$, where each image is of shape height and width. \autoref{fig:2DDWTlayer} illustrates input, output and perfect reconstruction by DWT 2D and IDWT 2D layers. The Multiresolution Encoder-Decoder Convolutional Neural Network in [@tarafdar2025multiresolution] uses these forward and inverse transform layers.

![Natural domain (left) and spatial-frequency tiling (right) after a DWT 2D.\label{fig:subdband2d}](figs/SpatialFrequencyTilingLv1.jpg){ width=70% }

![Separable DWT 2D perfect reconstruction filter bank.\label{fig:2BPRFB-2}](figs/DWTFB2D.jpg){ width=70% }

### 4.1.1 DWT 2D layer

A DWT 2D layer operates on input tensors of shape $(\text{{batch, height, width, channels}})$ and produces an output of shape $(\text{batch}, \text{height}/2, \text{width}/2, 4\times \text{channels})$ as described in Algorithm 2a.

***Algorithm 2a —***

1.  Input $\boldsymbol{X}$ of shape  $(\text{{batch, height, width, channels}})$.
2.  Generate analysis matrix $\boldsymbol{A}$ using $\text{{height}}$   and $\text{{width}}$ of input.
3.  For each batched channel $\boldsymbol{x}_{c}\in\boldsymbol{X}$ of shape $(\text{{batch, height, width}})$:  
    (omitting suffix $c$ in $\boldsymbol{x}$ below for simplicity of  notation)
    a.  Row-wise batch DWT 1D: $\boldsymbol{A}\boldsymbol{x}_{021}^{T}:=\text{Einsum} \big(ij,bjk \rightarrow bik \big)$
    b.  Column-wise batch DWT 1D: $\boldsymbol{A}(\boldsymbol{A}\boldsymbol{x}_{021}^{T})_{021}^{T} :=\text{Einsum} \big(ij,bjk\rightarrow bik\big)$
        Or, equivalently, DWT of a batched channel $\boldsymbol{x}$ is,
        $$\boldsymbol{q}_{c}=\text{{DWT}}\big(\boldsymbol{x}\big)\text{{ or, }}\boldsymbol{q}_{c}=\boldsymbol{A}(\boldsymbol{A}\boldsymbol{x}_{021})_{021}^{T}$$ 
		where, the suffix $021$ in $\boldsymbol{x}_{021}^{T}$ denotes permutation of axis, i.e., transpose.
4.  Stacking for all $c$ channels: $\boldsymbol{Q}:=\big(\boldsymbol{q}_{c}\big)_{\forall c}$ to a shape $(\text{{batch, height, width, channels}})$.
5.  Group subbands and return an output $\boldsymbol{Q}^{(\text{{grouped}})}$ of shape $(\text{batch}, \text{height}/2, \text{width}/2, 4\times\text{channels})$

	```python
		# Grouping four subbands in DWT 2D
		mid = int(Q.shape[1]/2)
		LL = Q[:,:mid,:mid,:]
		LH = Q[:,mid:,:mid,:]
		HL = Q[:,:mid,mid:,:]
		HH = Q[:,mid:,mid:,:]
		output = Concatenate([LL, LH, HL, HH], axis=-1)
	```

### 4.1.2 IDWT 2D layer

An IDWT 2D layer operates on input tensors of shape $(\text{batch}, \text{height}/2, \text{width}/2, 4\times \text{channels})$ and produces an output of shape $(\text{{batch, height, width, channels}})$ as described in Algorithm 2b.

***Algorithm 2b —***

1.  Input $\boldsymbol{Q}^{(\text{{grouped}})}$ of shape  $(\text{batch}, \text{height}/2, \text{width}/2, 4\times\text{channels})$
2.  Ungroup the subbands to get $\boldsymbol{Q}$ of shape $(\text{{batch, height, width, channels}})$
3.  Generate synthesis matrix $\boldsymbol{S}$ using $\text{{height}}$ and $\text{{width}}$ of $\boldsymbol{Q}$.
4.  For each batched channel $\boldsymbol{q}_{c}\in\boldsymbol{Q}$ of shape $(\text{{batch, height, width}})$:  
    (omitting suffix $c$ in $\boldsymbol{q}$ below for simplicity of  notation)
	a.   Row-wise batch IDWT 1D: $\boldsymbol{S}\boldsymbol{q}_{021}^{T}:=\text{Einsum}\big(ij,bjk\rightarrow bik\big)$
	b.   Column wise batch IDWT 1D: $\boldsymbol{S}(\boldsymbol{S}\boldsymbol{q}_{021}^{T})_{021}^{T}:=\text{Einsum}\big(ij,bjk\rightarrow bik\big)$
        or equivalently, a perfect reconstruction,
        $$\boldsymbol{x}=\text{{IDWT}}\big(\boldsymbol{q}\big)\text{{ or, }}\boldsymbol{x}=\boldsymbol{S}(\boldsymbol{S}\boldsymbol{q}_{021}^{T})_{021}^{T}$$ 
		where, the suffix $021$ in $\boldsymbol{x}_{021}^{T}$ denotes permutation of axis, i.e., transpose and $\boldsymbol{S}=\boldsymbol{A}^{T}$ for orthogonal   wavelets
5.  Layer output : $\boldsymbol{X}:=\big(\boldsymbol{x}_{c}\big)_{\forall c}$ is of  shape $(\text{{batch, height, width, channels}})$ \
(Perfect reconstruction)

![DWT decomposition and perfect reconstruction of a multichannel image tensor.\label{fig:2DDWTlayer}](figs/DWT2DIDWT2D_PRlayer.jpg){ width=60% }

## 4.2 Three-dimensional discrete wavelet system

A three-dimensional (3D) discrete wavelet system for a 3D input $\boldsymbol{x}$ is given by,

$$
\boldsymbol{q}	=\text{{DWT}}\big(\boldsymbol{x}\big):=\left[\boldsymbol{A}(\boldsymbol{A}(\boldsymbol{A}\boldsymbol{x}_{0213}^{T})_{0213}^{T})_{0132}^{T}\right]{}_{0132}^{T}
\text{\hspace{3.9cm} —Analysis}
$$

$$
\boldsymbol{x}	=\text{{IDWT}}\big(\boldsymbol{q}\big):=(\boldsymbol{S}(\boldsymbol{S}(\boldsymbol{S}\left[\boldsymbol{A}(\boldsymbol{A}(\boldsymbol{A}\boldsymbol{x}_{0213}^{T})_{0213}^{T})_{0132}^{T}\right]{}_{0132}^{T}{}_{0132}^{T})_{0132}^{T})_{0213}^{T})_{0213}
\text{ \hspace{.1cm}    —Synthesis}
$$

where, $\boldsymbol{A}$ and $\boldsymbol{S}$ are the same analysis and synthesis matrices as defined for one-dimensional wavelet system. The DWT 3D and IDWT 3D layers operate on batched, multichannel tensors of shape $(\text{{batch, height, width, depth, channels}})$.

### 4.2.1 DWT 3D layer

A DWT 3D layer operates on input tensors of shape $(\text{{batch, height, width, depth, channels}})$ and produces an output of shape $(\text{batch}, \text{height}/2, \text{width}/2, \text{depth}/2, 8\times \text{channels})$ as described in Algorithm 3a.

***Algorithm 3a —***

1.  Input $\boldsymbol{X}$ of shape  (batch, height, width, depth, channels).
2.  Generate analysis matrix $\boldsymbol{A}$ using $\text{{height}}$,  $\text{{width}}$ and $\text{{depth}}$of input.
3.  For each batched channel $\boldsymbol{x}_{c}\in\boldsymbol{X}$ of shape $(\text{{batch, height, width, depth}})$:  
    (omitting suffix $c$ in $\boldsymbol{x}$ below for simplicity of notation)
	a.  Row-wise batch DWT 1D:   $\boldsymbol{A}\boldsymbol{x}_{0213}^{T}:=\text{{Einsum}}\big(ij,bjkl\rightarrow bikl\big)$
	b.  Column-wise batch DWT 1D: $\boldsymbol{A}\left(\boldsymbol{A}\boldsymbol{x}_{0213}^{T}\right)_{0213}^{T}:=\text{{Einsum}}\big(ij,bjkl\rightarrow bikl\big)$
	c.  Depth-wise batch IDWT 1D: $\boldsymbol{A}(\boldsymbol{A}(\boldsymbol{A}\boldsymbol{x}_{0213}^{T})_{0213}^{T})_{0132}^{T}:=\text{{Einsum}}\big(ik,bjkl\rightarrow bjil\big)$
        Therefore, DWT of $\boldsymbol{x}$ yield coefficients: $$\boldsymbol{q}_{c}:=\text{{DWT}}\left(\boldsymbol{x}\right)=\left[\boldsymbol{A}(\boldsymbol{A}(\boldsymbol{A}\boldsymbol{x}_{0213}^{T})_{0213}^{T})_{0132}^{T}\right]{}_{0132}^{T}$$
4.  Stacking for all $c$ channels as $\boldsymbol{Q}:=\big(\boldsymbol{q}_{c}\big)_{\forall c}$ to a shape (batch, height, width, depth, channels).
5.  Group subbands and return an output $\boldsymbol{Q}^{(\text{{grouped}})}$ of shape (batch, height$/2$, width$/2$, depth$/2$, $8\times$channels).  
  
	```python
		# Grouping Eight subbands in 3D DWT
        mid = int(Q.shape[2]/2)
		LLL = Q[:,:mid,:mid,:mid,:]
		LLH = Q[:,mid:,:mid,:mid,:]
		LHL = Q[:,:mid,mid:,:mid,:]
		LHH = Q[:,mid:,mid:,:mid,:]
		HLL = Q[:,:mid,:mid,mid:,:]
		HLH = Q[:,mid:,mid:,mid:,:]
		HHL = Q[:,:mid,mid:,mid:,:]
		HHH = Q[:,_mid:,mid:,mid:,:]
		output = Concatenate([LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH], axis=-1)
	```

### 4.2.2 IDWT 3D layer

An IDWT 3D layer operates on input tensors of shape $(\text{batch}, \text{height}/2, \text{width}/2, \text{depth}/2, 8\times \text{channels})$ and produces an output of shape $(\text{{batch, height, width, depth, channels}})$ as described in Algorithm 3b.

***Algorithm 3b —***

1.  Input $\boldsymbol{Q}^{(\text{{grouped}})}$ of shape $(\text{batch}, \text{height}/2, \text{width}/2, \text{depth}/2, 8\times\text{channels})$
2.  Ungroup to get $\boldsymbol{Q}$ of shape $(\text{{batch, height, width, depth, channels}})$
3.  Generate synthesis matrix $\boldsymbol{S}$ using $\text{{height}}$,  $\text{{width}}$ and $\text{{depth}}$ of $\boldsymbol{Q}$.
4.  For each batched channel $\boldsymbol{q}_{c}\in\boldsymbol{Q}$ of shape $(\text{{batch, height, width, depth}})$:  
    (omitting suffix $c$ in $\boldsymbol{q}$ below for simplicity of  notation)
	a.  Row-wise batch IDWT 1D: $\boldsymbol{S}\boldsymbol{q}_{0132}^{T}:=\text{{Einsum}}\big(ik,bjkl\rightarrow bjil\big)$
	b.  Column-wise batch IDWT 1D: $\boldsymbol{S} (\boldsymbol{S}\boldsymbol{q}_{0132}^{T})_{0132}:=\text{{Einsum}}\big(ij,bjkl\rightarrow bikl\big)$
	c.  Depth-wise batch IDWT 1D: $\boldsymbol{S}(\boldsymbol{S}(\boldsymbol{S}\boldsymbol{q}_{0132}^{T})_{0132}^{T})_{0213}^{T}:=\text{{Einsum}}\big(ij,bjkl\rightarrow bikl\big)$
        or equivalently, a perfect reconstruction,        
        $$\boldsymbol{x}_{c}:=\text{{IDWT}}\left(\boldsymbol{q}\right)=\left[\boldsymbol{S}(\boldsymbol{S}(\boldsymbol{S}\boldsymbol{q}_{0132}^{T})_{0132}^{T})_{0213}^{T}\right]_{0213}^{T}$$
        where, $\boldsymbol{S}=\boldsymbol{A}^{T}$ for orthogonal wavelets.
5.  Layer output : $\boldsymbol{X}:=\big(\boldsymbol{x}_{c}\big)_{\forall c}$ is of  shape $(\text{{batch, height, width, depth, channels}})$ \
(Perfect reconstruction)
    



<br><br>

In general, a seamless realization of fast $\text{D}$\-dimensional DWT and IDWT is possible by extending the above separable method to all the independent $N$ axes one after the other. The number of subbands will be equal to $2^{\text{{D}}}$ for a $\text{{D}}$ dimensional DWT. For example, sequences $(\text{{D}}=1)$ yield two subbands, images $(\text{{D}}=2)$ yield four subbands, three-dimensional inputs $(\text{{D}}=3)$ with voxels yield eight subbands etc.

# 5 Multilevel wavelet filter banks

The above-discussed DWT and IDWT layers are building blocks in constructing multilevel DWT filter banks. \autoref{fig:MultilevelDWTtiling} shows the partitioning of the 1D frequency axis and tiling of the 2D frequency plane using a level-4 1D and 2D DWT. The multilevel DWT successively decomposes the low-frequency feature. If the high-frequency features are also decomposed successively, then we get a Wavelet Packet Transform (WPT) filter bank.

![Spatial-frequency tiling by DWT level 4 decomposition of a sequence of length N (top) and image of shape Nimes N (bottom).\label{fig:MultilevelDWTtiling}](figs/SpatialFrequencyTilingLv4.jpg){ width=80% }


# References


