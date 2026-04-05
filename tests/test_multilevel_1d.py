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

from TFDWT.multilevel.dwt import dwt as dwt1d, idwt as idwt1d


def _make_custom_waveform(L: int = 1024) -> np.ndarray:
    t = np.arange(L, dtype=np.float32)
    tn = t / max(t[-1], 1.0)
    s1 = 0.6 * np.sin(2 * np.pi * 5.0 * tn)
    s2 = 0.3 * np.sin(2 * np.pi * 20.0 * tn)
    chirp = 0.2 * np.sin(2 * np.pi * (0.5 * (tn ** 2) * 40.0))
    step = 0.4 * (tn > 0.5).astype(np.float32)
    bump = 0.5 * np.exp(-0.5 * ((tn - 0.3) / 0.05) ** 2)
    x = s1 + s2 + chirp + step + bump
    # Normalize to [0,1]
    x_min, x_max = float(x.min()), float(x.max())
    if x_max > x_min:
        x = (x - x_min) / (x_max - x_min)
    else:
        x = np.zeros_like(x)
    return x.astype(np.float32, copy=False)


def _pad_to_multiple_pow2_1d(x: np.ndarray, level: int, pad_value: float = 0.0) -> np.ndarray:
    m = 2 ** level
    L = x.shape[0]
    N = ((L + m - 1) // m) * m
    if N == L:
        return x.astype(np.float32, copy=False)
    out = np.full((N,), pad_value, dtype=np.float32)
    start = (N - L) // 2
    out[start:start + L] = x
    return out


def plot_multilevel_1d(x: np.ndarray, xhat: np.ndarray, subbands: list[tf.Tensor], out_path: str):
    """Combined plot: input, reconstruction, H1..Hn, and final LL (no residual)."""
    n = len(subbands) - 1
    fig_rows = 1 + ((n + 2) // 2)  # 1 row for input/recon, then ceil((n+1)/2) rows for H.. and L
    fig, axes = plt.subplots(fig_rows, 2, figsize=(14, 4 * fig_rows), squeeze=False)

    # Row 0: Input and Reconstruction
    t = np.arange(x.size)
    axes[0, 0].plot(t, x, color="black", linewidth=1.0)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_title("Input")
    axes[0, 1].plot(t, xhat, color="tab:blue", linewidth=1.0)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_title("Reconstruction")

    # Highpasses H1..Hn
    row = 1
    col = 0
    for i, H in enumerate(subbands[:-1], start=1):
        h = H.numpy()[0, :, 0]
        axes[row, col].plot(np.arange(h.size), h, linewidth=1.0)
        axes[row, col].grid(True, alpha=0.3)
        axes[row, col].set_title(f"H{i}")
        col ^= 1
        if col == 0:
            row += 1

    # Final lowpass
    Ln = subbands[-1].numpy()[0, :, 0]
    axes[row, col].plot(np.arange(Ln.size), Ln, color="black", linewidth=1.0)
    axes[row, col].grid(True, alpha=0.3)
    axes[row, col].set_title(f"L{n}")

    fig.suptitle("Multilevel DWT1D: Input and Subbands")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main(level: int = 3, wavelet: str = "haar"):
    x = _make_custom_waveform(1024)
    x = _pad_to_multiple_pow2_1d(x, level)
    xt = tf.convert_to_tensor(x[None, :, None], dtype=tf.float32)
    subbands = dwt1d(xt, level=level, Ψ=wavelet)
    xhat = idwt1d(subbands, level=level, Ψ=wavelet).numpy()[0, :, 0]
    max_err = float(np.max(np.abs(x - xhat)))
    print("Multilevel 1D max_err:", max_err)
    out_img = os.path.join(THIS_DIR, f"dwt1d_multilevel_L{level}.png")
    plot_multilevel_1d(x, xhat, subbands, out_img)
    print(f"Saved: {out_img}")


def test_multilevel_1d_pr():
    level = 3
    x = _make_custom_waveform(1024)
    x = _pad_to_multiple_pow2_1d(x, level)
    xt = tf.convert_to_tensor(x[None, :, None], dtype=tf.float32)
    subbands = dwt1d(xt, level=level, Ψ="haar")
    xhat = idwt1d(subbands, level=level, Ψ="haar").numpy()[0, :, 0]
    out_img = os.path.join(THIS_DIR, f"dwt1d_multilevel_L{level}.png")
    plot_multilevel_1d(x, xhat, subbands, out_img)
    assert float(np.max(np.abs(x - xhat))) < 1e-6


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Multilevel DWT1D test and plot")
    p.add_argument("--level", type=int, default=3)
    p.add_argument("--wavelet", default="haar")
    args = p.parse_args()
    main(level=args.level, wavelet=args.wavelet)
