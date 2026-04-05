import os
import sys
import numpy as np
import matplotlib

# Non-interactive backend
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

# Ensure the package root is on sys.path when run directly
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if PKG_ROOT not in sys.path:
    sys.path.insert(0, PKG_ROOT)

import tensorflow as tf

from TFDWT.DWT2DFB import DWT2D, IDWT2D


DEFAULT_NIFTI_PATH = \
    "/home/kkt/DATASETS.port/IBSR_nifti_stripped/IBSR_01/IBSR_01_ana.nii.gz"


def _load_nifti(path: str) -> np.ndarray:
    try:
        import nibabel as nib  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "nibabel is required to load NIfTI files. Please install it: pip install nibabel"
        ) from e

    if not os.path.exists(path):
        raise FileNotFoundError(f"NIfTI path not found: {path}")

    img = nib.load(path)
    data = img.get_fdata(dtype=np.float32)
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    # Squeeze trailing singleton if present (e.g., (X,Y,Z,1))
    if data.ndim == 4 and data.shape[-1] == 1:
        data = data[..., 0]
    return data


def _normalize01(x: np.ndarray) -> np.ndarray:
    x_min = float(np.min(x))
    x_max = float(np.max(x))
    if x_max > x_min:
        x = (x - x_min) / (x_max - x_min)
    else:
        x = np.zeros_like(x)
    return x.astype(np.float32, copy=False)


def _extract_slice(vol: np.ndarray, plane: str = "axial", index: int | None = None) -> np.ndarray:
    """Extract a 2D slice from a 3D volume (no cropping)."""
    if vol.ndim != 3:
        raise AssertionError(f"Expected 3D volume for slicing, got {vol.shape}")
    X, Y, Z = vol.shape
    if index is None:
        if plane == "axial":
            index = Z // 2
        elif plane == "coronal":
            index = Y // 2
        elif plane == "sagittal":
            index = X // 2
        else:
            raise ValueError("plane must be one of: axial, coronal, sagittal")

    if plane == "axial":
        img = vol[:, :, index]
    elif plane == "coronal":
        img = vol[:, index, :]
    elif plane == "sagittal":
        img = vol[index, :, :]
    else:
        raise ValueError("plane must be one of: axial, coronal, sagittal")
    return img.astype(np.float32, copy=False)


def _to_even_square(img: np.ndarray, target_N: int | None = None, pad_value: float = 0.0) -> np.ndarray:
    """Pad a 2D image to an even-sized square without cropping (centered)."""
    if img.ndim != 2:
        raise AssertionError(f"Expected 2D image, got {img.shape}")
    H, W = img.shape
    base = max(H, W)
    if target_N is None:
        N = base
    else:
        N = max(int(target_N), base)
    if N % 2 == 1:
        N += 1

    out = np.full((N, N), pad_value, dtype=np.float32)
    # centered placement
    top = (N - H) // 2
    left = (N - W) // 2
    out[top:top + H, left:left + W] = img
    return out


def run_dwt2d_and_recon(image2d: np.ndarray, wavelet: str = "haar", clean: bool = True):
    assert image2d.ndim == 2
    N = image2d.shape[0]
    assert image2d.shape == (N, N), "Input must be a square image"

    x = tf.convert_to_tensor(image2d[None, ..., None], dtype=tf.float32)  # (1,N,N,1)
    dwt = DWT2D(wave=wavelet, clean=clean)
    idwt = IDWT2D(wave=wavelet, clean=clean)

    subbands = dwt(x)   # (1, N/2, N/2, 4)
    xhat = idwt(subbands)

    abs_err = tf.math.abs(x - xhat)
    max_err = float(tf.reduce_max(abs_err))
    mae = float(tf.reduce_mean(abs_err))
    num = float(tf.norm(x - xhat))
    den = float(tf.norm(x) + 1e-12)
    rel_l2 = num / den
    return subbands, x, xhat, {"max_err": max_err, "mae": mae, "rel_l2": rel_l2}


def plot_subbands_grid_2d(subbands: tf.Tensor, x: tf.Tensor, xhat: tf.Tensor, out_path: str, pct: float = 99.5):
    """Plot input, reconstruction, and 4 subbands in one 2x3 grid.

    Layout:
      - Col0: Input (row0), Reconstruction (row1)
      - Col1..2: Subbands arranged to match spatial quadrants:
          row0: LL (col1), HL (col2)
          row1: LH (col1), HH (col2)
    """
    sub = subbands.numpy()
    x_np = x.numpy()
    xhat_np = xhat.numpy()
    assert sub.ndim == 4 and sub.shape[0] == 1 and sub.shape[-1] == 4, f"Unexpected shape {sub.shape}"
    assert x_np.ndim == 4 and x_np.shape == xhat_np.shape and x_np.shape[0] == 1 and x_np.shape[-1] == 1
    sub = sub[0]  # (n2, n2, 4)

    names = ["LL", "LH", "HL", "HH"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Column 0: input vs reconstruction
    img_in = x_np[0, :, :, 0]
    img_out = xhat_np[0, :, :, 0]
    vmin_io = float(min(img_in.min(), img_out.min()))
    vmax_io = float(max(img_in.max(), img_out.max()))
    axes[0, 0].imshow(img_in, cmap="gray", vmin=vmin_io, vmax=vmax_io)
    axes[0, 0].set_title("Input")
    axes[0, 0].axis("off")
    axes[1, 0].imshow(img_out, cmap="gray", vmin=vmin_io, vmax=vmax_io)
    axes[1, 0].set_title("Reconstruction")
    axes[1, 0].axis("off")

    # Subbands: [[LL, HL], [LH, HH]] -> indices [0,2] on row0; [1,3] on row1
    layout = { (0,1): 0, (0,2): 2, (1,1): 1, (1,2): 3 }
    for (r, c), idx in layout.items():
        band = sub[:, :, idx]
        vmax = float(np.percentile(np.abs(band), pct))
        if vmax == 0.0 or not np.isfinite(vmax):
            vmax = float(np.max(np.abs(band)) + 1e-8)
        vmin = -vmax
        rms = float(np.sqrt(np.mean(band**2)) + 1e-12)
        if idx == 0:
            bmin = float(np.min(band))
            bmax = float(np.max(band))
            axes[r, c].imshow(band, cmap="gray", vmin=bmin, vmax=bmax)
        else:
            axes[r, c].imshow(band, cmap="seismic", vmin=vmin, vmax=vmax)
        axes[r, c].set_title(f"{names[idx]}  rms={rms:.2e}")
        axes[r, c].axis("off")

    fig.suptitle(f"DWT2D Input/Output and Subbands (pct={pct})")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_in_out_2d(x: tf.Tensor, xhat: tf.Tensor, out_path: str):
    """Plot input and reconstruction images side-by-side for 2D case."""
    x_np = x.numpy()
    xhat_np = xhat.numpy()
    assert x_np.ndim == 4 and x_np.shape[0] == 1 and x_np.shape[-1] == 1
    assert xhat_np.shape == x_np.shape
    img_in = x_np[0, :, :, 0]
    img_out = xhat_np[0, :, :, 0]
    vmin = float(min(img_in.min(), img_out.min()))
    vmax = float(max(img_in.max(), img_out.max()))

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img_in, cmap="gray", vmin=vmin, vmax=vmax)
    axes[0].set_title("Input")
    axes[0].axis("off")
    axes[1].imshow(img_out, cmap="gray", vmin=vmin, vmax=vmax)
    axes[1].set_title("Reconstruction")
    axes[1].axis("off")
    fig.suptitle("DWT2D Input vs Reconstruction")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main(nifti_path: str = DEFAULT_NIFTI_PATH, plane: str = "axial", wavelet: str = "haar", target_N: int | None = None):
    print(f"Loading volume: {nifti_path}")
    vol = _normalize01(_load_nifti(nifti_path))
    img = _extract_slice(vol, plane=plane, index=None)
    sq = _to_even_square(img, target_N)
    N = sq.shape[0]
    print(f"Slice shape: {img.shape} -> square {sq.shape} (N={N})")

    subbands, x, xhat, metrics = run_dwt2d_and_recon(sq, wavelet=wavelet, clean=True)
    print("Reconstruction metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.6e}")
    tol = 1e-5
    print(f"Perfect reconstruction within tol={tol}: {metrics['max_err'] < tol}")

    out_img = os.path.join(THIS_DIR, f"dwt2d_ibsr_subbands_{plane}.png")
    plot_subbands_grid_2d(subbands, x, xhat, out_img)
    print(f"Saved combined figure: {out_img}")


def test_dwt2d_ibsr_pr():
    """Pytest test: DWT2D/IDWT2D PR on a zero-padded central slice (no cropping)."""
    try:
        import nibabel  # noqa: F401
    except Exception:
        import pytest  # type: ignore
        pytest.skip("nibabel not installed; skipping NIfTI-based test")

    if not os.path.exists(DEFAULT_NIFTI_PATH):
        import pytest  # type: ignore
        pytest.skip("Sample NIfTI not found; skipping")

    vol = _normalize01(_load_nifti(DEFAULT_NIFTI_PATH))
    img = _extract_slice(vol, plane="axial", index=None)
    sq = _to_even_square(img)
    subbands, x, xhat, metrics = run_dwt2d_and_recon(sq, wavelet="haar", clean=True)

    out_img = os.path.join(THIS_DIR, "dwt2d_ibsr_subbands_axial.png")
    plot_subbands_grid_2d(subbands, x, xhat, out_img)

    assert metrics["max_err"] < 1e-5, f"Max error too high: {metrics['max_err']:.3e}"


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Test DWT2D on a NIfTI slice and plot subbands")
    p.add_argument("--nifti", default=DEFAULT_NIFTI_PATH, help="Path to .nii or .nii.gz volume")
    p.add_argument("--plane", default="axial", choices=["axial", "coronal", "sagittal"], help="Slice plane")
    p.add_argument("--wavelet", default="haar", help="Wavelet name (e.g., haar, db2, bior1.5)")
    p.add_argument("--N", type=int, default=None, help="Target square size (even). Default: pad to max dimension")
    args = p.parse_args()

    main(nifti_path=args.nifti, plane=args.plane, wavelet=args.wavelet, target_N=args.N)
