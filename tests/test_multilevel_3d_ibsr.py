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

from TFDWT.multilevel.dwt3 import dwt3, idwt3
from TFDWT.DWT3DFB import IDWT3D


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


def _pad_cube_pow2(vol: np.ndarray, level: int, pad_value: float = 0.0) -> np.ndarray:
    if vol.ndim == 4 and vol.shape[-1] == 1:
        vol = vol[..., 0]
    X, Y, Z = vol.shape
    base = max(X, Y, Z)
    m = 2 ** level
    N = ((base + m - 1) // m) * m
    if N % 2 == 1:
        N += 1
    out = np.full((N, N, N), pad_value, dtype=np.float32)
    sx, sy, sz = (N - X) // 2, (N - Y) // 2, (N - Z) // 2
    out[sx:sx+X, sy:sy+Y, sz:sz+Z] = vol
    return out


def _split_level3_subbands(Hlvl: tf.Tensor) -> list[tf.Tensor]:
    # Hlvl shape: (1, n2, n2, n2, 7*channels)
    return tf.split(Hlvl, num_or_size_splits=7, axis=-1)


def _compute_ll_chain_3d(subbands: list[tf.Tensor], wavelet: str) -> list[tf.Tensor]:
    *highpasses, low = subbands
    Ls = [None] * len(subbands)
    current = low
    Ls[-1] = current
    for i in range(len(highpasses) - 1, -1, -1):
        current = IDWT3D(wave=wavelet)(tf.keras.layers.Concatenate()([current, highpasses[i]]))
        Ls[i] = current
    return Ls[1:]


def _central_slice(arr: np.ndarray, plane: str = "axial") -> np.ndarray:
    if plane == "axial":
        k = arr.shape[2] // 2
        return arr[:, :, k]
    if plane == "coronal":
        k = arr.shape[1] // 2
        return arr[:, k, :]
    if plane == "sagittal":
        k = arr.shape[0] // 2
        return arr[k, :, :]
    raise ValueError("plane must be one of: axial, coronal, sagittal")


def _plot_level_3d(orig: np.ndarray, recon: np.ndarray, LL: tf.Tensor, Hlvl: tf.Tensor, out_path: str, plane: str = "axial", pct: float = 99.5):
    names = ["LLL", "LLH", "LHL", "LHH", "HLL", "HLH", "HHL", "HHH"]
    bands = [LL] + _split_level3_subbands(Hlvl)

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    vmin_io = float(min(orig.min(), recon.min())); vmax_io = float(max(orig.max(), recon.max()))
    axes[0,0].imshow(orig, cmap="gray", vmin=vmin_io, vmax=vmax_io); axes[0,0].set_title("Input"); axes[0,0].axis("off")
    axes[1,0].imshow(recon, cmap="gray", vmin=vmin_io, vmax=vmax_io); axes[1,0].set_title("Reconstruction"); axes[1,0].axis("off")

    for i in range(8):
        vol = bands[i].numpy()[0, :, :, :, 0]
        sl = _central_slice(vol, plane)
        if i == 0:
            bmin, bmax = float(sl.min()), float(sl.max())
            cmap = "gray"; vmin, vmax = bmin, bmax
        else:
            vmax = float(np.percentile(np.abs(vol), pct)); vmin = -vmax; cmap = "seismic"
        r = 0 if i < 4 else 1
        c = 1 + (i % 4)
        axes[r,c].imshow(sl, cmap=cmap, vmin=vmin, vmax=vmax)
        axes[r,c].set_title(names[i]); axes[r,c].axis("off")

    fig.suptitle("Multilevel DWT3D: Input/Output and Level Subbands")
    fig.tight_layout(); os.makedirs(os.path.dirname(out_path), exist_ok=True); fig.savefig(out_path, dpi=160); plt.close(fig)


def main(level: int = 3, wavelet: str = "haar", plane: str = "axial"):
    vol = _normalize01(_load_nifti(DEFAULT_NIFTI_PATH))
    vol = _pad_cube_pow2(vol, level)
    x = tf.convert_to_tensor(vol[None, ..., None], dtype=tf.float32)
    subbands = dwt3(x, level=level, Ψ=wavelet)
    xhat = idwt3(subbands, level=level, Ψ=wavelet).numpy()[0, :, :, :, 0]
    max_err = float(np.max(np.abs(vol - xhat)))
    print("Multilevel 3D max_err:", max_err)

    LLs = _compute_ll_chain_3d(subbands, wavelet)
    for i in range(level):
        LL_i = LLs[i]
        H_i = subbands[i]
        out_img = os.path.join(THIS_DIR, f"dwt3d_multilevel_L{i+1}_{plane}.png")
        _plot_level_3d(_central_slice(vol, plane), _central_slice(xhat, plane), LL_i, H_i, out_img, plane=plane)
        print(f"Saved: {out_img}")


def test_multilevel_3d_pr():
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
    vol = _pad_cube_pow2(vol, level)
    x = tf.convert_to_tensor(vol[None, ..., None], dtype=tf.float32)
    subbands = dwt3(x, level=level, Ψ="haar")
    xhat = idwt3(subbands, level=level, Ψ="haar").numpy()[0, :, :, :, 0]
    assert float(np.max(np.abs(vol - xhat))) < 1e-5


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Multilevel DWT3D on NIfTI volume")
    p.add_argument("--level", type=int, default=3)
    p.add_argument("--wavelet", default="haar")
    p.add_argument("--plane", default="axial")
    args = p.parse_args()
    main(level=args.level, wavelet=args.wavelet, plane=args.plane)

