import os
import sys
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if PKG_ROOT not in sys.path:
    sys.path.insert(0, PKG_ROOT)

import tensorflow as tf

from TFDWT.multilevel.dwt2 import dwt2, idwt2
from TFDWT.DWT2DFB import IDWT2D


DEFAULT_NIFTI_PATH = \
    "/home/kkt/DATASETS.port/IBSR_nifti_stripped/IBSR_01/IBSR_01_ana.nii.gz"


def _load_nifti(path: str) -> np.ndarray:
    import nibabel as nib
    img = nib.load(path)
    data = img.get_fdata(dtype=np.float32)
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    if data.ndim == 4 and data.shape[-1] == 1:
        data = data[..., 0]
    return data


def _normalize01(x: np.ndarray) -> np.ndarray:
    x_min, x_max = float(np.min(x)), float(np.max(x))
    if x_max > x_min:
        x = (x - x_min) / (x_max - x_min)
    else:
        x = np.zeros_like(x)
    return x.astype(np.float32, copy=False)


def _extract_slice(vol: np.ndarray, plane: str = "axial") -> np.ndarray:
    X, Y, Z = vol.shape
    if plane == "axial":
        return vol[:, :, Z // 2]
    if plane == "coronal":
        return vol[:, Y // 2, :]
    if plane == "sagittal":
        return vol[X // 2, :, :]
    raise ValueError("plane must be one of: axial, coronal, sagittal")


def _pad_square_pow2(img: np.ndarray, level: int, pad_value: float = 0.0) -> np.ndarray:
    H, W = img.shape
    base = max(H, W)
    m = 2 ** level
    N = ((base + m - 1) // m) * m
    if N % 2 == 1:
        N += 1
    out = np.full((N, N), pad_value, dtype=np.float32)
    top = (N - H) // 2
    left = (N - W) // 2
    out[top:top + H, left:left + W] = img
    return out


def _split_level2_subbands(Hlvl: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    # Hlvl has shape (1, n2, n2, 3*channels)
    return tf.split(Hlvl, num_or_size_splits=3, axis=-1)


def _compute_ll_chain_2d(subbands: list[tf.Tensor], wavelet: str) -> list[tf.Tensor]:
    """Compute [L1, L2, ..., Ln] from [H1,...,Hn,Ln] using IDWT2D."""
    *highpasses, low = subbands
    Ls = [None] * len(subbands)
    current = low
    Ls[-1] = current
    for i in range(len(highpasses) - 1, -1, -1):
        current = IDWT2D(wave=wavelet)(tf.keras.layers.Concatenate()([current, highpasses[i]]))
        Ls[i] = current
    return Ls[1:]  # return [L1, L2, ..., Ln]


def _plot_level_2d(orig: np.ndarray, recon: np.ndarray, LL: tf.Tensor, Hlvl: tf.Tensor, out_path: str, pct: float = 99.5):
    names = ["LL", "LH", "HL", "HH"]
    LH, HL, HH = _split_level2_subbands(Hlvl)
    bands = [LL, LH, HL, HH]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    vmin_io = float(min(orig.min(), recon.min()))
    vmax_io = float(max(orig.max(), recon.max()))
    axes[0,0].imshow(orig, cmap="gray", vmin=vmin_io, vmax=vmax_io); axes[0,0].set_title("Input"); axes[0,0].axis("off")
    axes[1,0].imshow(recon, cmap="gray", vmin=vmin_io, vmax=vmax_io); axes[1,0].set_title("Reconstruction"); axes[1,0].axis("off")

    # Subbands arranged [[LL, HL],[LH, HH]] in the remaining cols
    layout = { (0,1):0, (0,2):2, (1,1):1, (1,2):3 }
    for (r,c), idx in layout.items():
        b = bands[idx].numpy()[0, :, :, 0]
        if idx == 0:
            bmin, bmax = float(b.min()), float(b.max())
            axes[r,c].imshow(b, cmap="gray", vmin=bmin, vmax=bmax)
        else:
            vmax = float(np.percentile(np.abs(b), pct)); vmin = -vmax
            axes[r,c].imshow(b, cmap="seismic", vmin=vmin, vmax=vmax)
        axes[r,c].set_title(names[idx]); axes[r,c].axis("off")

    fig.suptitle("Multilevel DWT2D: Input/Output and Level Subbands")
    fig.tight_layout(); os.makedirs(os.path.dirname(out_path), exist_ok=True); fig.savefig(out_path, dpi=160); plt.close(fig)


def main(level: int = 3, plane: str = "axial", wavelet: str = "haar"):
    vol = _normalize01(_load_nifti(DEFAULT_NIFTI_PATH))
    img = _extract_slice(vol, plane=plane)
    img = _pad_square_pow2(img, level)
    x = tf.convert_to_tensor(img[None, ..., None], dtype=tf.float32)
    subbands = dwt2(x, level=level, Ψ=wavelet)
    xhat = idwt2(subbands, level=level, Ψ=wavelet).numpy()[0, :, :, 0]
    max_err = float(np.max(np.abs(img - xhat)))
    print("Multilevel 2D max_err:", max_err)

    # Compute lowpass chain and plot per level
    LLs = _compute_ll_chain_2d(subbands, wavelet)
    for i in range(level):
        LL_i = LLs[i]
        H_i = subbands[i]
        out_img = os.path.join(THIS_DIR, f"dwt2d_multilevel_L{i+1}_{plane}.png")
        _plot_level_2d(img, xhat, LL_i, H_i, out_img)
        print(f"Saved: {out_img}")


def test_multilevel_2d_pr():
    try:
        import nibabel  # noqa: F401
    except Exception:
        import pytest
        pytest.skip("nibabel not installed")
    if not os.path.exists(DEFAULT_NIFTI_PATH):
        import pytest
        pytest.skip("Sample NIfTI not found")

    level = 3
    vol = _normalize01(_load_nifti(DEFAULT_NIFTI_PATH))
    img = _extract_slice(vol, plane="axial")
    img = _pad_square_pow2(img, level)
    x = tf.convert_to_tensor(img[None, ..., None], dtype=tf.float32)
    subbands = dwt2(x, level=level, Ψ="haar")
    xhat = idwt2(subbands, level=level, Ψ="haar").numpy()[0, :, :, 0]
    LLs = _compute_ll_chain_2d(subbands, "haar")
    _plot_level_2d(img, xhat, LLs[0], subbands[0], os.path.join(THIS_DIR, "dwt2d_multilevel_L1_axial.png"))
    assert float(np.max(np.abs(img - xhat))) < 1e-5


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Multilevel DWT2D on NIfTI slice")
    p.add_argument("--level", type=int, default=3)
    p.add_argument("--plane", default="axial")
    p.add_argument("--wavelet", default="haar")
    args = p.parse_args()
    main(level=args.level, plane=args.plane, wavelet=args.wavelet)

