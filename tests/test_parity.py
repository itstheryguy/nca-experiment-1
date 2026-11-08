"""The numpy runtime (space/nca_numpy.py) must match CAModel, for both model types. Run: python tests/test_parity.py"""
import io
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "space")]
import nca  # noqa: E402
import nca_numpy  # noqa: E402

CONFIGS = {
    "mountain-style": nca.CAConfig(channels=16, hidden=128, alive_mask=True, wrap=False, render="rgba"),
    "texture-style": nca.CAConfig(channels=12, hidden=96, alive_mask=False, wrap=True, render="rgb"),
}


def check(name, cfg, steps=20, grid=64):
    torch.manual_seed(0)
    model = nca.CAModel(cfg)
    torch.nn.init.normal_(model.w2.weight, std=0.05)  # non-zero, or every step is a no-op
    buf = io.BytesIO()
    nca.export_npz(model, buf, grid, name)
    buf.seek(0)
    wts = nca_numpy.load_weights(buf)
    assert wts["cfg"]["channels"] == cfg.channels and wts["cfg"]["wrap"] == cfg.wrap

    # A random live patch touching the top edge, so padding (zero vs wrap) matters.
    x_t = torch.zeros(1, cfg.channels, grid, grid)
    x_t[:, :, 0:16, 24:40] = torch.rand(1, cfg.channels, 16, 16)
    x_n = x_t[0].numpy().copy()
    rng = np.random.default_rng(0)
    with torch.no_grad():
        for _ in range(steps):  # fire_rate=1 removes the randomness, so the two paths are comparable
            x_t = model(x_t, fire_rate=1.0)
            x_n = nca_numpy.step(x_n, wts, rng, fire_rate=1.0)
    err = np.abs(x_t[0].numpy() - x_n).max()
    assert err < 1e-4, (name, err)
    rgb_err = np.abs(nca.to_rgb(x_t[0], cfg) * 255 - nca_numpy.to_rgb(x_n, wts["cfg"])).max()
    assert rgb_err <= 1.0, (name, rgb_err)
    print(f"ok {name}: max abs diff {err:.2e} after {steps} steps")


def test_parity():
    for name, cfg in CONFIGS.items():
        check(name, cfg)


if __name__ == "__main__":
    test_parity()
