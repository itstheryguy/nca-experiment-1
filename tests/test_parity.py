"""The numpy runtime (space/nca_numpy.py) must match CAModel. Run: python tests/test_parity.py"""
import io
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "space")]
import nca  # noqa: E402
import nca_numpy  # noqa: E402


def test_parity(steps=20):
    torch.manual_seed(0)
    model = nca.CAModel()
    torch.nn.init.normal_(model.w2.weight, std=0.05)  # non-zero, or every step is a no-op
    buf = io.BytesIO()
    nca.export_npz(model, buf)
    buf.seek(0)
    wts = nca_numpy.load_weights(buf)

    # Start from a random live patch, not a single seed, so most of the grid is exercised.
    x_t = torch.zeros(1, nca.CHANNELS, nca.GRID, nca.GRID)
    x_t[:, :, 24:40, 24:40] = torch.rand(1, nca.CHANNELS, 16, 16)
    x_n = x_t[0].numpy().copy()
    rng = np.random.default_rng(0)
    with torch.no_grad():
        for _ in range(steps):  # fire_rate=1 removes the randomness, so the two paths are comparable
            x_t = model(x_t, fire_rate=1.0)
            x_n = nca_numpy.step(x_n, wts, rng, fire_rate=1.0)
    err = np.abs(x_t[0].numpy() - x_n).max()
    alive = int((x_n[3] > 0.1).sum())
    assert alive > 100, "most of the patch should still be alive"
    assert err < 1e-4, err
    print(f"ok: max abs diff {err:.2e} after {steps} steps, {alive} live cells")


if __name__ == "__main__":
    test_parity()
