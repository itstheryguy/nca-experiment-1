"""Numpy port of the NCA update rule in nca.py, so the Space can run in the browser (Pyodide has no torch).

The state is one grid, shaped (C, H, W) float32. Keep this in sync with `CAModel.forward` in nca.py.
"""

import numpy as np

ALIVE_THRESHOLD = 0.1
SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32) / 8.0
SOBEL_Y = SOBEL_X.T.copy()


def load_weights(path_or_file):
    z = np.load(path_or_file)
    return {k: z[k] for k in ("w1", "b1", "w2")} | {"fire_rate": float(z["fire_rate"])}


def make_seed(channels, grid):
    x = np.zeros((channels, grid, grid), dtype=np.float32)
    x[3:, grid // 2, grid // 2] = 1.0
    return x


def _filter3x3(x, k):
    """Depthwise 3x3 cross-correlation with zero padding, like F.conv2d(padding=1, groups=C)."""
    _, h, w = x.shape
    p = np.pad(x, ((0, 0), (1, 1), (1, 1)))
    out = np.zeros_like(x)
    for i in range(3):
        for j in range(3):
            if k[i, j]:
                out += k[i, j] * p[:, i:i + h, j:j + w]
    return out


def _alive(x):
    """3x3 max of alpha > threshold. Zero padding is fine: zero never exceeds the threshold."""
    _, h, w = x.shape
    p = np.pad(x[3], 1)
    m = np.max([p[i:i + h, j:j + w] for i in range(3) for j in range(3)], axis=0)
    return m > ALIVE_THRESHOLD


def step(x, wts, rng, fire_rate=None):
    """One CA step. Returns a new (C, H, W) array."""
    fire_rate = wts["fire_rate"] if fire_rate is None else fire_rate
    c, h, w = x.shape
    pre_life = _alive(x)
    # (3C, H, W), interleaved per channel [c0, c0_dx, c0_dy, c1, ...] to match the torch depthwise conv
    y = np.stack([x, _filter3x3(x, SOBEL_X), _filter3x3(x, SOBEL_Y)], axis=1).reshape(3 * c, h, w)
    hid = np.maximum(wts["w1"] @ y.reshape(3 * c, -1) + wts["b1"][:, None], 0)
    dx = (wts["w2"] @ hid).reshape(c, h, w)
    fire = rng.random((h, w)) <= fire_rate
    x = x + dx * fire
    life = pre_life & _alive(x)
    return (x * life).astype(np.float32)


def erase(x, cy, cx, radius):
    """Zero a circle of cells (for click-to-damage). Coordinates in pixels."""
    _, h, w = x.shape
    yy, xx = np.ogrid[:h, :w]
    x = x.copy()
    x[:, (yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2] = 0
    return x


def to_rgb(x):
    """RGB on a white background, (H, W, 3) uint8."""
    a = np.clip(x[3], 0, 1)
    rgb = np.clip(1.0 - a[None] + x[:3], 0, 1)
    return (rgb.transpose(1, 2, 0) * 255).astype(np.uint8)


def to_heatmap(x, channel):
    """One channel scaled to grayscale, (H, W) uint8. Hidden channels are unbounded, so scale by the max |value|."""
    v = x[channel]
    m = float(np.abs(v).max()) or 1.0
    return ((v / m * 0.5 + 0.5) * 255).astype(np.uint8)
