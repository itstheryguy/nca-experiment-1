"""Render the README GIF for the network model with the Space's numpy runtime.

Grows a network on a random layout, damages it at `--damage-at`, and lets it repair. Tries seeds in order and keeps
the first layout with at least `--min-targets` targets that is fully connected before the damage and again at the
end, so the GIF shows the behavior (69% of layouts connect fully; see README).

    python tools/make_network_gif.py --seed 0
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "space"))
import nca_numpy as ca  # noqa: E402
import network_numpy as nw  # noqa: E402


def run(wts, seed, steps, damage_at, every):
    rng = np.random.default_rng(seed)
    cfg = wts["cfg"]
    layout = nw.make_layout(rng, cfg["grid"])
    inp = nw.inputs(layout, cfg["grid"])
    x = nw.seed(cfg, layout)
    frames, ok_before = [], False
    for t in range(steps):
        if t == damage_at:
            ok_before = nw.connected(x) == 1.0
            # Erase a circle on the network's midpoint, so the damage cuts it.
            ys, xs = np.nonzero(nw.density(x) > 0.5)
            cy, cx = (int(ys.mean()), int(xs.mean())) if len(ys) else (32, 32)
            x = nw.clamp(ca.erase(x, cy, cx, radius=9), inp)
        x = nw.clamp(ca.step(x, wts, rng), inp)
        if t % every == 0:
            frames.append(nw.render(x))
    return layout, frames, ok_before and nw.connected(x) == 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-targets", type=int, default=5)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--damage-at", type=int, default=250)
    ap.add_argument("--every", type=int, default=2)
    ap.add_argument("--out", default=str(ROOT / "assets" / "network.gif"))
    a = ap.parse_args()
    wts = ca.load_weights(ROOT / "space" / "models" / "network.npz")
    for seed in range(a.seed, a.seed + 200):
        layout, frames, ok = run(wts, seed, a.steps, a.damage_at, a.every)
        if ok and len(layout[1]) >= a.min_targets:
            break
    else:
        sys.exit("no seed found")
    imgs = [Image.fromarray(f).resize((256, 256), Image.NEAREST) for f in frames]
    imgs += [imgs[-1]] * 30  # hold the last frame
    imgs[0].save(a.out, save_all=True, append_images=imgs[1:], duration=40, loop=0, optimize=True)
    print(f"seed {seed}: {len(layout[1])} targets, {len(imgs)} frames -> {a.out}")


if __name__ == "__main__":
    main()
