"""Numpy port of the NCA update rule in nca.py, so the Space can run in the browser (Pyodide has no torch).

The state is one grid, shaped (C, H, W) float32. Each model's settings (channels, wrap, alive mask, render mode)
come from the `config` entry of its .npz. Keep this in sync with `CAModel.forward` in nca.py.
"""

import json

import numpy as np

ALIVE_THRESHOLD = 0.1
SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32) / 8.0
SOBEL_Y = SOBEL_X.T.copy()


def load_weights(path_or_file):
    """{"w1", "b1", "w2", "cfg"}, where cfg holds channels, hidden, fire_rate, alive_mask, wrap, render, grid, name."""
    z = np.load(path_or_file)
    return {k: z[k] for k in ("w1", "b1", "w2")} | {"cfg": json.loads(str(z["config"]))}


def make_seed(cfg):
    c, g = cfg["channels"], cfg["grid"]
    x = np.zeros((c, g, g), dtype=np.float32)
    if cfg["alive_mask"]:
        x[3:, g // 2, g // 2] = 1.0
    return x


def _pad(a, wrap):
    """Pad the last two axes by 1: wrap around (toroidal grid) or with zeros."""
    width = [(0, 0)] * (a.ndim - 2) + [(1, 1), (1, 1)]
    return np.pad(a, width, mode="wrap" if wrap else "constant")


def _filter3x3(x, k, wrap):
    """Depthwise 3x3 cross-correlation, like the torch depthwise conv."""
    _, h, w = x.shape
    p = _pad(x, wrap)
    out = np.zeros_like(x)
    for i in range(3):
        for j in range(3):
            if k[i, j]:
                out += k[i, j] * p[:, i:i + h, j:j + w]
    return out


def _max3x3(a):
    """3x3 neighborhood max of a 2D array, zero padded (only used with the alive mask, which never wraps)."""
    h, w = a.shape
    p = _pad(a, False)
    return np.max([p[i:i + h, j:j + w] for i in range(3) for j in range(3)], axis=0)


def _alive(x):
    """3x3 max of alpha > threshold. Zero padding is fine: zero never exceeds the threshold."""
    return _max3x3(x[3]) > ALIVE_THRESHOLD


def step(x, wts, rng, fire_rate=None):
    """One CA step. Returns a new (C, H, W) array.

    Same result as CAModel.forward. With the alive mask, the MLP only runs on cells that fire and lie within one
    cell of a live cell: cells farther out end up zeroed by the mask and can't change any kept cell's
    post-update life test, so their update never matters. Under WebAssembly this is about 7x faster.
    """
    cfg = wts["cfg"]
    fire_rate = cfg["fire_rate"] if fire_rate is None else fire_rate
    c, h, w = x.shape
    pre_life = _alive(x) if cfg["alive_mask"] else None
    fire = rng.random((h, w)) <= fire_rate
    idx = np.flatnonzero(fire & _max3x3(pre_life) if pre_life is not None else fire)
    # (3C, H, W), interleaved per channel [c0, c0_dx, c0_dy, c1, ...] to match the torch depthwise conv
    sx, sy = _filter3x3(x, SOBEL_X, cfg["wrap"]), _filter3x3(x, SOBEL_Y, cfg["wrap"])
    y = np.stack([x, sx, sy], axis=1).reshape(3 * c, h * w)
    hid = np.maximum(wts["w1"] @ y[:, idx] + wts["b1"][:, None], 0)
    x = x.reshape(c, h * w).copy()
    x[:, idx] += wts["w2"] @ hid
    x = x.reshape(c, h, w)
    if pre_life is not None:
        x = x * (pre_life & _alive(x))
    return x.astype(np.float32)


def erase(x, cy, cx, radius):
    """Zero a circle of cells (for damage). Coordinates in pixels."""
    _, h, w = x.shape
    yy, xx = np.ogrid[:h, :w]
    x = x.copy()
    x[:, (yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2] = 0
    return x


def to_rgb(x, cfg):
    """Displayable RGB, (H, W, 3) uint8."""
    if cfg["render"] == "network":
        import network_numpy

        return network_numpy.render(x)
    if cfg["render"] == "rgb":
        rgb = np.clip(x[:3] + 0.5, 0, 1)
    else:
        a = np.clip(x[3], 0, 1)
        rgb = np.clip(1.0 - a[None] + x[:3], 0, 1)
    return (rgb.transpose(1, 2, 0) * 255).astype(np.uint8)


def to_heatmap(x, channel):
    """One channel scaled to grayscale, (H, W) uint8. Hidden channels are unbounded, so scale by the max |value|."""
    v = x[channel]
    m = float(np.abs(v).max()) or 1.0
    return ((v / m * 0.5 + 0.5) * 255).astype(np.uint8)
