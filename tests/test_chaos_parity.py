"""chaos.py (PyTorch, batched) must match space/chaos_numpy.py draw for draw. Run: python tests/test_chaos_parity.py"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "space")]
import chaos  # noqa: E402
import chaos_numpy as cn  # noqa: E402

CFGS = {
    "mountain-style": {"channels": 16, "alive_mask": True, "wrap": False, "render": "rgba"},
    "texture-style": {"channels": 12, "alive_mask": False, "wrap": True, "render": "rgb"},
}


def run(cfg, preset, ticks=60, grid=64, seed=0):
    rng = np.random.default_rng(seed)
    x0 = rng.normal(0, 0.5, (cfg["channels"], grid, grid)).astype(np.float32)
    if cfg["alive_mask"]:  # a live blob in the middle, empty background around it
        yy, xx = np.ogrid[:grid, :grid]
        x0[3] = np.where((yy - 32) ** 2 + (xx - 32) ** 2 < 15 ** 2, 1.0, 0.0)
    draws = []

    def rec(shape):
        a = rng.random(shape)
        draws.append(a)
        return a

    c_n = np.zeros((grid, grid), np.int8)
    for s in (5, 40, 300):  # sparks inside and outside the blob
        c_n.flat[s * 13] = cn.RED
    x_n, hist_n = x0.copy(), []
    for _ in range(ticks):
        c_n, x_n = cn.tick(c_n, x_n, cfg, preset, rec)
        hist_n.append(c_n.copy())

    it = iter(draws)

    def replay(shape):
        a = next(it)
        assert a.size == int(np.prod(shape)), (a.shape, shape)
        return torch.as_tensor(a, dtype=torch.float64).reshape(shape)

    ch = chaos.Chaos(preset, cfg, "cpu")
    c_t = torch.zeros(1, grid, grid, dtype=torch.long)
    for s in (5, 40, 300):
        c_t.view(-1)[s * 13] = cn.RED
    x_t = torch.as_tensor(x0)[None]
    for i in range(ticks):
        c_t, x_t = ch.tick(c_t, x_t, replay)
        assert (c_t[0].numpy() == hist_n[i]).all(), f"chaos differs at tick {i}"
    assert next(it, None) is None, "torch used fewer draws"
    err = np.abs(x_t[0].numpy() - x_n).max()
    assert err < 1e-6, err
    return cn.counts(c_n)


def test_chaos_parity():
    presets = chaos.load_presets()
    for cname, cfg in CFGS.items():
        for pname, preset in presets.items():
            frac = run(cfg, preset)
            print(f"ok {cname:15s} {pname:10s} after 60 ticks: " + ", ".join(f"{k} {v:.2f}" for k, v in frac.items()))


if __name__ == "__main__":
    test_chaos_parity()
