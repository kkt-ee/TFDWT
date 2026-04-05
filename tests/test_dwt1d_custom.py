import os
import sys
import numpy as np
import matplotlib

# Non-interactive backend for headless envs
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

# Ensure the package root is on sys.path when run directly
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if PKG_ROOT not in sys.path:
    sys.path.insert(0, PKG_ROOT)

import tensorflow as tf

from TFDWT.DWT1DFB import DWT1D, IDWT1D


def _to_even_length(x: np.ndarray, target_N: int | None = None, pad_value: float = 0.0) -> np.ndarray:
    """Pad a 1D signal to an even length without cropping (centered).

    - If target_N is provided, ensures final length is max(target_N, len(x)).
    - Guarantees even length by padding one extra sample if needed.
    """
    assert x.ndim == 1, f"Expected 1D array, got shape {x.shape}"
    L = x.shape[0]
    base = L
    if target_N is None:
        N = base
    else:
        N = max(int(target_N), base)
    if N % 2 == 1:
        N += 1
    if N == L:
        return x.astype(np.float32, copy=False)

    out = np.full((N,), pad_value, dtype=np.float32)
    start = (N - L) // 2
    out[start:start + L] = x.astype(np.float32, copy=False)
    return out


def _make_custom_waveform(L: int = 1024, fs: float = 1.0) -> np.ndarray:
    """Create a rich 1D waveform: sinusoids + chirp + step + Gaussian bump."""
    t = np.arange(L, dtype=np.float32) / fs
    t_norm = t / max(t[-1], 1.0)
    # Components
    s1 = 0.6 * np.sin(2 * np.pi * (5.0) * t_norm)
    s2 = 0.3 * np.sin(2 * np.pi * (20.0) * t_norm)
    # Quadratic chirp
    chirp = 0.2 * np.sin(2 * np.pi * (0.5 * (t_norm ** 2) * 40.0))
    # Step at middle
    step = 0.4 * (t_norm > 0.5).astype(np.float32)
    # Localized Gaussian bump
    bump = 0.5 * np.exp(-0.5 * ((t_norm - 0.3) / 0.05) ** 2)
    x = s1 + s2 + chirp + step + bump
    # Normalize to [0,1]
    x_min = float(np.min(x))
    x_max = float(np.max(x))
    if x_max > x_min:
        x = (x - x_min) / (x_max - x_min)
    else:
        x = np.zeros_like(x)
    return x.astype(np.float32, copy=False)


def run_dwt1d_and_recon(x1d: np.ndarray, wavelet: str = "haar", clean: bool = True):
    assert x1d.ndim == 1
    N = x1d.shape[0]
    x = tf.convert_to_tensor(x1d[None, :, None], dtype=tf.float32)  # (1, N, 1)

    dwt = DWT1D(wave=wavelet, clean=clean)
    idwt = IDWT1D(wave=wavelet, clean=clean)

    subbands = dwt(x)    # (1, N/2, 2) for single channel
    xhat = idwt(subbands) # (1, N, 1)

    abs_err = tf.math.abs(x - xhat)
    max_err = float(tf.reduce_max(abs_err))
    mae = float(tf.reduce_mean(abs_err))
    num = float(tf.norm(x - xhat))
    den = float(tf.norm(x) + 1e-12)
    rel_l2 = num / den
    return subbands, x, xhat, {"max_err": max_err, "mae": mae, "rel_l2": rel_l2}


def plot_subbands_1d(subbands: tf.Tensor, x: tf.Tensor, xhat: tf.Tensor, out_path: str):
    """Plot input and reconstruction along with L/H subbands in one figure (no residual)."""
    sub = subbands.numpy()
    x_np = x.numpy(); xhat_np = xhat.numpy()
    assert sub.ndim == 3 and sub.shape[0] == 1 and sub.shape[-1] == 2, f"Unexpected shape {sub.shape}"
    assert x_np.ndim == 3 and x_np.shape == xhat_np.shape and x_np.shape[0] == 1 and x_np.shape[-1] == 1
    sub = sub[0]  # (N2, 2)
    L = sub[:, 0]
    H = sub[:, 1]
    xin = x_np[0, :, 0]
    xout = xhat_np[0, :, 0]
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    # Input
    axes[0,0].plot(np.arange(xin.size), xin, color="black", linewidth=1.0)
    axes[0,0].set_title("Input")
    axes[0,0].grid(True, alpha=0.3)
    # Reconstruction
    axes[0,1].plot(np.arange(xout.size), xout, color="tab:blue", linewidth=1.0)
    axes[0,1].set_title("Reconstruction")
    axes[0,1].grid(True, alpha=0.3)
    # Subbands
    axes[1,0].plot(np.arange(L.size), L, color="black", linewidth=1.0)
    axes[1,0].set_title("L (Low-pass)")
    axes[1,0].grid(True, alpha=0.3)
    axes[1,1].plot(np.arange(H.size), H, color="tab:red", linewidth=1.0)
    axes[1,1].set_title("H (High-pass)")
    axes[1,1].grid(True, alpha=0.3)
    fig.suptitle("DWT1D Input and Subbands")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_in_out_1d(x: tf.Tensor, xhat: tf.Tensor, out_path: str):
    """Overlay input and reconstruction waveforms, plus residual in a subplot."""
    x_np = x.numpy(); xhat_np = xhat.numpy()
    assert x_np.ndim == 3 and x_np.shape[0] == 1 and x_np.shape[-1] == 1
    assert xhat_np.shape == x_np.shape
    xin = x_np[0, :, 0]
    xout = xhat_np[0, :, 0]
    N = xin.shape[0]
    t = np.arange(N)

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axes[0].plot(t, xin, label="Input", color="black", linewidth=1.0)
    axes[0].plot(t, xout, label="Reconstruction", color="tab:blue", linewidth=1.0, alpha=0.8)
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)
    resid = xin - xout
    axes[1].plot(t, resid, label="Residual", color="tab:red", linewidth=1.0)
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)
    fig.suptitle("DWT1D Input vs Reconstruction (+ residual)")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main(wavelet: str = "haar", target_N: int | None = None):
    x = _make_custom_waveform(L=1024)
    x_pad = _to_even_length(x, target_N)
    print(f"Signal length: {len(x)} -> padded: {len(x_pad)}")

    subbands, x_tf, xhat_tf, metrics = run_dwt1d_and_recon(x_pad, wavelet=wavelet, clean=True)
    print("Reconstruction metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.6e}")
    tol = 1e-6
    print(f"Perfect reconstruction within tol={tol}: {metrics['max_err'] < tol}")

    out_img = os.path.join(THIS_DIR, "dwt1d_custom_subbands.png")
    plot_subbands_1d(subbands, x_tf, xhat_tf, out_img)
    print(f"Saved combined figure: {out_img}")


def test_dwt1d_custom_pr():
    """Pytest test: DWT1D/IDWT1D PR on a custom zero-padded 1D waveform (no cropping)."""
    x = _make_custom_waveform(L=1024)
    x_pad = _to_even_length(x)
    subbands, x_tf, xhat_tf, metrics = run_dwt1d_and_recon(x_pad, wavelet="haar", clean=True)

    out_img = os.path.join(THIS_DIR, "dwt1d_custom_subbands.png")
    plot_subbands_1d(subbands, x_tf, xhat_tf, out_img)

    assert metrics["max_err"] < 1e-6, f"Max error too high: {metrics['max_err']:.3e}"


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Test DWT1D on a custom waveform and plot subbands")
    p.add_argument("--wavelet", default="haar", help="Wavelet name (e.g., haar, db2, bior1.5)")
    p.add_argument("--N", type=int, default=None, help="Target even length (pad only, no crop)")
    args = p.parse_args()

    main(wavelet=args.wavelet, target_N=args.N)
