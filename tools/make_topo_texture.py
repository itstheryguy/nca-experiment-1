"""Generate the topographic-map texture used as the texture NCA's target: data/topo_texture.png.

Random terrain from spectral synthesis (1/f^beta noise via FFT, so it tiles seamlessly), colored in
elevation bands (hypsometric tints) with dark contour lines between bands. Run: python tools/make_topo_texture.py
"""
import sys
from pathlib import Path

import matplotlib
import numpy as np

SIZE = 128          # px; the NCA grid is 64 cells and is upsampled 2x for the style loss, so 1 cell = 2 px
BETA = 5.0          # spectral slope: higher = smoother terrain
BANDS = 10          # elevation bands between contour lines
SEED = 7


def terrain(size, beta, seed):
    rng = np.random.default_rng(seed)
    f = np.fft.fftfreq(size)
    k = np.sqrt(f[:, None] ** 2 + f[None, :] ** 2)
    k[0, 0] = 1.0
    spectrum = (rng.normal(size=(size, size)) + 1j * rng.normal(size=(size, size))) / k ** (beta / 2)
    spectrum[0, 0] = 0
    h = np.fft.ifft2(spectrum).real
    return (h - h.min()) / (h.max() - h.min())


def render(h, bands):
    level = np.minimum((h * bands).astype(int), bands - 1)
    cmap = matplotlib.colormaps["gist_earth"]
    colors = cmap(np.linspace(0.25, 0.95, bands))[:, :3]
    img = colors[level]
    # Contour line wherever a pixel's band differs from its right or lower neighbor (wrapping, to stay tileable).
    edge = (level != np.roll(level, -1, 0)) | (level != np.roll(level, -1, 1))
    edge |= np.roll(edge, 1, 0) | np.roll(edge, 1, 1)   # 2 px wide = 1 NCA cell
    img[edge] = img[edge] * 0.35
    return (img * 255).astype(np.uint8)


if __name__ == "__main__":
    from PIL import Image

    out = Path(sys.argv[1] if len(sys.argv) > 1 else "data/topo_texture.png")
    Image.fromarray(render(terrain(SIZE, BETA, SEED), BANDS)).save(out)
    print("wrote", out)
