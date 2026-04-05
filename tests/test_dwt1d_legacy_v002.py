import os
import sys
import types
import importlib
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))


def _make_custom_waveform(L: int = 1024) -> np.ndarray:
    t = np.arange(L, dtype=np.float32)
    tn = t / max(t[-1], 1.0)
    s1 = 0.6 * np.sin(2 * np.pi * 5.0 * tn)
    s2 = 0.3 * np.sin(2 * np.pi * 20.0 * tn)
    chirp = 0.2 * np.sin(2 * np.pi * (0.5 * (tn ** 2) * 40.0))
    step = 0.4 * (tn > 0.5).astype(np.float32)
    bump = 0.5 * np.exp(-0.5 * ((tn - 0.3) / 0.05) ** 2)
    x = s1 + s2 + chirp + step + bump
    x_min, x_max = float(x.min()), float(x.max())
    if x_max > x_min:
        x = (x - x_min) / (x_max - x_min)
    else:
        x = np.zeros_like(x)
    return x.astype(np.float32, copy=False)


def _to_even_length(x: np.ndarray, target_N: int | None = None, pad_value: float = 0.0) -> np.ndarray:
    L = x.shape[0]
    base = L
    N = base if target_N is None else max(int(target_N), base)
    if N % 2 == 1:
        N += 1
    if N == L:
        return x.astype(np.float32, copy=False)
    out = np.full((N,), pad_value, dtype=np.float32)
    start = (N - L) // 2
    out[start:start + L] = x.astype(np.float32, copy=False)
    return out


def _import_legacy_v002():
    """Load legacy TFDWT_v002 as a temporary top-level package 'TFDWT'."""
    legacy_root = os.path.join(PKG_ROOT, "older_versions", "TFDWT_v002")
    if not os.path.isdir(legacy_root):
        raise FileNotFoundError(f"Legacy folder not found: {legacy_root}")

    saved = {name: sys.modules[name] for name in list(sys.modules) if name == "TFDWT" or name.startswith("TFDWT.")}
    for name in list(sys.modules):
        if name == "TFDWT" or name.startswith("TFDWT."):
            del sys.modules[name]

    pkg = types.ModuleType("TFDWT")
    pkg.__path__ = [legacy_root]
    sys.modules["TFDWT"] = pkg

    try:
        mod = importlib.import_module("TFDWT.DWTIDWT1Dv1")
        DWT1D_legacy = getattr(mod, "DWT1D")
        IDWT1D_legacy = getattr(mod, "IDWT1D")
        return DWT1D_legacy, IDWT1D_legacy
    finally:
        # restore original TFDWT modules to avoid interference
        for name in list(sys.modules):
            if name == "TFDWT" or name.startswith("TFDWT."):
                del sys.modules[name]
        for k, v in saved.items():
            sys.modules[k] = v


def plot_combined_legacy(x: np.ndarray, xhat: np.ndarray, subbands: np.ndarray, out_path: str):
    """Plot input and reconstruction (legacy) and subbands in one figure (no residual)."""
    assert subbands.ndim == 3 and subbands.shape[-1] == 2  # (1, N/2, 2)
    L = subbands[0, :, 0]
    H = subbands[0, :, 1]

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes[0,0].plot(np.arange(x.size), x, color="black", linewidth=1.0)
    axes[0,0].set_title("Input")
    axes[0,0].grid(True, alpha=0.3)
    axes[0,1].plot(np.arange(xhat.size), xhat, color="tab:green", linewidth=1.0)
    axes[0,1].set_title("Reconstruction (legacy)")
    axes[0,1].grid(True, alpha=0.3)
    axes[1,0].plot(np.arange(L.size), L, color="black", linewidth=1.0)
    axes[1,0].set_title("L (Low-pass, legacy)")
    axes[1,0].grid(True, alpha=0.3)
    axes[1,1].plot(np.arange(H.size), H, color="tab:green", linewidth=1.0)
    axes[1,1].set_title("H (High-pass, legacy)")
    axes[1,1].grid(True, alpha=0.3)
    fig.suptitle("Legacy v0.0.2 DWT1D Input and Subbands")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main(wavelet: str = "haar", target_N: int | None = None):
    import tensorflow as tf

    DWT1D_legacy, IDWT1D_legacy = _import_legacy_v002()
    x = _make_custom_waveform(1024)
    x_pad = _to_even_length(x, target_N)

    xt = tf.convert_to_tensor(x_pad[None, :, None], dtype=tf.float32)
    dwt = DWT1D_legacy(wave=wavelet)
    idwt = IDWT1D_legacy(wave=wavelet)
    subbands = dwt(xt)             # (1, N/2, 2)
    xhat = idwt(subbands).numpy()  # (1, N, 1)

    abs_err = np.abs(xt.numpy() - xhat)
    print("Legacy reconstruction max_err:", float(abs_err.max()))

    out_img = os.path.join(THIS_DIR, "dwt1d_custom_subbands_legacy_v002.png")
    plot_combined_legacy(x_pad, xhat[0, :, 0], subbands.numpy(), out_img)
    print(f"Saved legacy combined figure: {out_img}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Plot 1D DWT using legacy TFDWT_v002")
    p.add_argument("--wavelet", default="haar")
    p.add_argument("--N", type=int, default=None)
    args = p.parse_args()
    main(wavelet=args.wavelet, target_N=args.N)
