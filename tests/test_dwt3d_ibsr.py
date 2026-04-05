import os
import sys
import math
import numpy as np
import matplotlib

# Use a non-interactive backend for headless environments
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

# Ensure the package root is on sys.path when run directly
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if PKG_ROOT not in sys.path:
    sys.path.insert(0, PKG_ROOT)

import tensorflow as tf

from TFDWT.DWT3DFB import DWT3D, IDWT3D


DEFAULT_NIFTI_PATH = \
    "/home/kkt/DATASETS.port/IBSR_nifti_stripped/IBSR_01/IBSR_01_ana.nii.gz"


def _to_even_cubic(vol: np.ndarray, target_N: int | None = None, pad_value: float = 0.0) -> np.ndarray:
    """Return an even-sized cubic volume by centered zero-padding only (no cropping).

    - Accepts 3D volumes or 4D with trailing singleton channel/time dim and squeezes it.
    - Chooses N so that N >= each input dimension; i.e., it never crops.
      * If target_N is provided and target_N >= max(vol.shape), N = target_N.
      * Otherwise N = max(vol.shape).
    - Ensures the final N is even (rounds up by 1 if odd) via padding only.
    """
    # Squeeze 4D inputs with last dim == 1
    if vol.ndim == 4 and vol.shape[-1] == 1:
        vol = np.squeeze(vol, axis=-1)
    if vol.ndim != 3:
        raise AssertionError(f"Expected 3D or 4D (with last dim=1), got shape {vol.shape}")

    X, Y, Z = vol.shape
    base = max(X, Y, Z)
    if target_N is None:
        N = base
    else:
        N = max(int(target_N), base)
    if N % 2 == 1:
        N += 1

    # Prepare destination cube
    cube = np.full((N, N, N), pad_value, dtype=np.float32)

    # Determine source and destination slices per axis (centered) — pad only
    src_slices = []
    dst_slices = []
    for s in (X, Y, Z):
        # With N >= s ensured, we never crop.
        start_dst = (N - s) // 2
        dst_slices.append(slice(start_dst, start_dst + s))
        src_slices.append(slice(0, s))

    cube[dst_slices[0], dst_slices[1], dst_slices[2]] = vol[src_slices[0], src_slices[1], src_slices[2]].astype(np.float32, copy=False)
    return cube


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
    # Replace NaNs/Infs if present
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    # Squeeze trailing singleton if present (e.g., shape (X,Y,Z,1))
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


def run_dwt3d_and_recon(volume3d: np.ndarray, wavelet: str = "haar", clean: bool = True):
    """Run 3D DWT and inverse DWT, return subbands and reconstruction error metrics."""
    assert volume3d.ndim == 3

    # Prepare input tensor: (1, N, N, N, 1)
    N = volume3d.shape[0]
    assert volume3d.shape == (N, N, N), "Input must be a cubic volume"
    x = tf.convert_to_tensor(volume3d[None, ..., None], dtype=tf.float32)

    dwt = DWT3D(wave=wavelet, clean=clean)
    idwt = IDWT3D(wave=wavelet, clean=clean)

    subbands = dwt(x)  # (1, N/2, N/2, N/2, 8) if clean else (1,N,N,N,1)
    xhat = idwt(subbands)

    # Errors
    abs_err = tf.math.abs(x - xhat)
    max_err = float(tf.reduce_max(abs_err))
    mae = float(tf.reduce_mean(abs_err))
    # Relative L2 error
    num = float(tf.norm(x - xhat))
    den = float(tf.norm(x) + 1e-12)
    rel_l2 = num / den

    return subbands, x, xhat, {"max_err": max_err, "mae": mae, "rel_l2": rel_l2}


def plot_subbands_grid(subbands: tf.Tensor, x: tf.Tensor, xhat: tf.Tensor, out_path: str, plane: str = "axial", pct: float = 99.5):
    """Plot input and reconstruction plus 8 subbands in one figure.

    Layout: 2 rows x 5 columns
        col0: Input (row0), Reconstruction (row1)
        col1-4: Subbands [LLL, LLH, LHL, LHH] on row0 and [HLL, HLH, HHL, HHH] on row1
    """
    sub = subbands.numpy()
    x_np = x.numpy()
    xhat_np = xhat.numpy()
    assert sub.ndim == 5 and sub.shape[0] == 1, f"Unexpected subbands shape {sub.shape}"
    assert x_np.ndim == 5 and x_np.shape == xhat_np.shape and x_np.shape[0] == 1 and x_np.shape[-1] == 1

    sub = sub[0]                # (n2, n2, n2, 8)
    vol_in = x_np[0, :, :, :, 0]
    vol_out = xhat_np[0, :, :, :, 0]

    def central_slice(arr: np.ndarray, plane: str):
        if plane == "axial":
            k = arr.shape[2] // 2
            return arr[:, :, k]
        elif plane == "coronal":
            k = arr.shape[1] // 2
            return arr[:, k, :]
        elif plane == "sagittal":
            k = arr.shape[0] // 2
            return arr[k, :, :]
        else:
            raise ValueError("plane must be one of: axial, coronal, sagittal")

    names = ["LLL", "LLH", "LHL", "LHH", "HLL", "HLH", "HHL", "HHH"]

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))

    # Column 0: Input vs Reconstruction with matched scaling
    sl_in = central_slice(vol_in, plane)
    sl_out = central_slice(vol_out, plane)
    vmin_io = float(min(sl_in.min(), sl_out.min()))
    vmax_io = float(max(sl_in.max(), sl_out.max()))
    axes[0, 0].imshow(sl_in, cmap="gray", vmin=vmin_io, vmax=vmax_io)
    axes[0, 0].set_title("Input")
    axes[0, 0].axis("off")
    axes[1, 0].imshow(sl_out, cmap="gray", vmin=vmin_io, vmax=vmax_io)
    axes[1, 0].set_title("Reconstruction")
    axes[1, 0].axis("off")

    # Subbands
    for i in range(8):
        vol_i = sub[:, :, :, i]
        sl = central_slice(vol_i, plane)
        vmax = float(np.percentile(np.abs(vol_i), pct))
        if vmax == 0.0 or not np.isfinite(vmax):
            vmax = float(np.max(np.abs(vol_i)) + 1e-8)
        vmin = -vmax
        rms = float(np.sqrt(np.mean(vol_i**2)) + 1e-12)
        # Map index to row/col: first four to row0 col1..4, next four to row1 col1..4
        row = 0 if i < 4 else 1
        col = 1 + (i % 4)
        ax = axes[row, col]
        ax.imshow(sl, cmap="seismic", vmin=vmin, vmax=vmax)
        ax.set_title(f"{names[i]}  rms={rms:.2e}")
        ax.axis("off")

    fig.suptitle(f"DWT3D Input/Output and Subbands (central {plane} slice, pct={pct})")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_in_out_3d(x: tf.Tensor, xhat: tf.Tensor, out_path: str, plane: str = "axial"):
    """Plot central slice of input and reconstruction side-by-side for comparison."""
    x_np = x.numpy()
    xhat_np = xhat.numpy()
    assert x_np.ndim == 5 and x_np.shape[0] == 1 and x_np.shape[-1] == 1
    assert xhat_np.shape == x_np.shape

    vol = x_np[0, :, :, :, 0]
    vol_hat = xhat_np[0, :, :, :, 0]
    N = vol.shape[0]
    mid = N // 2

    if plane == "axial":
        sl_in, sl_out = vol[:, :, mid], vol_hat[:, :, mid]
    elif plane == "coronal":
        sl_in, sl_out = vol[:, mid, :], vol_hat[:, mid, :]
    elif plane == "sagittal":
        sl_in, sl_out = vol[mid, :, :], vol_hat[mid, :, :]
    else:
        raise ValueError("plane must be one of: axial, coronal, sagittal")

    vmin = float(min(sl_in.min(), sl_out.min()))
    vmax = float(max(sl_in.max(), sl_out.max()))

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(sl_in, cmap="gray", vmin=vmin, vmax=vmax)
    axes[0].set_title("Input (central slice)")
    axes[0].axis("off")
    axes[1].imshow(sl_out, cmap="gray", vmin=vmin, vmax=vmax)
    axes[1].set_title("Reconstruction (central slice)")
    axes[1].axis("off")
    fig.suptitle(f"DWT3D Input vs Reconstruction ({plane})")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main(nifti_path: str = DEFAULT_NIFTI_PATH, wavelet: str = "haar", target_N: int | None = None):
    print(f"Loading volume: {nifti_path}")
    vol = _load_nifti(nifti_path)
    vol = _normalize01(vol)
    # Make an even cubic volume; if target_N is provided (e.g., 256), pad/crop to that size
    cube = _to_even_cubic(vol, target_N)
    N = cube.shape[0]
    print(f"Cube shape: {cube.shape} (N={N})")

    subbands, x, xhat, metrics = run_dwt3d_and_recon(cube, wavelet=wavelet, clean=True)
    print("Reconstruction metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.6e}")

    # Basic PR check: allow tiny numerical error
    tol = 1e-5
    ok = metrics["max_err"] < tol
    print(f"Perfect reconstruction within tol={tol}: {ok}")

    out_img = os.path.join(THIS_DIR, "dwt3d_ibsr_subbands.png")
    plot_subbands_grid(subbands, x, xhat, out_img)
    print(f"Saved combined figure: {out_img}")


def test_dwt3d_ibsr_pr():
    """Pytest-style test: checks PR on a zero-padded cubic version of the sample NIfTI.

    Skips if nibabel is not installed or the file is missing.
    """
    try:
        import nibabel  # noqa: F401
    except Exception:
        import pytest  # type: ignore
        pytest.skip("nibabel not installed; skipping NIfTI-based test")

    if not os.path.exists(DEFAULT_NIFTI_PATH):
        import pytest  # type: ignore
        pytest.skip("Sample NIfTI not found; skipping")

    vol = _normalize01(_load_nifti(DEFAULT_NIFTI_PATH))
    # Create an even cubic volume by padding/cropping to the max dimension
    cube = _to_even_cubic(vol)
    subbands, x, xhat, metrics = run_dwt3d_and_recon(cube, wavelet="haar", clean=True)

    # Save a plot alongside the test for manual inspection
    out_img = os.path.join(THIS_DIR, "dwt3d_ibsr_subbands.png")
    plot_subbands_grid(subbands, x, xhat, out_img)

    # Assert near-perfect reconstruction
    assert metrics["max_err"] < 1e-5, f"Max error too high: {metrics['max_err']:.3e}"


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Test DWT3D on a NIfTI volume and plot subbands")
    p.add_argument("--nifti", default=DEFAULT_NIFTI_PATH, help="Path to .nii or .nii.gz volume")
    p.add_argument("--wavelet", default="haar", help="Wavelet name (e.g., haar, db2, bior1.5)")
    p.add_argument("--N", type=int, default=None, help="Target cube size (even). Default: largest even cube")
    args = p.parse_args()

    main(nifti_path=args.nifti, wavelet=args.wavelet, target_N=args.N)
