"""Monte Carlo runs of an NCA against the chaos agent: many independent games in one batch.

Each game grows the NCA from its seed, sparks one agent at a random cell at `start`, and runs until chaos has
burned out everywhere (no red cells) plus `settle` more steps, or `max_steps`. Per game it records when chaos burned
out, the final share of NCA / blue cells, and (for the mountain) image integrity.

    python tools/montecarlo.py mountain --presets brushfire smolder wildfire --n 128
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import chaos  # noqa: E402
import nca  # noqa: E402
import train  # noqa: E402


def integrity(x, c, ch, target_rgb):
    """1 = the target image, 0 = no better than a blank grid, as displayed (chaos cells in their colors)."""
    img = nca.rgb(x, ch.cfg)
    colors = torch.as_tensor(chaos.cn.COLORS / 255.0, dtype=img.dtype, device=img.device)
    img = torch.where((c > 0)[:, None], colors[c].permute(0, 3, 1, 2), img)
    mse = ((img - target_rgb) ** 2).mean(dim=(1, 2, 3))
    blank = ((1.0 - target_rgb) ** 2).mean()
    return (1 - mse / blank).clamp(min=0)


@torch.no_grad()
def simulate(model, exp, preset, n=128, start=150, every=1, max_steps=5000, settle=200, device="cpu", seed=0):
    gen = torch.Generator(device=device).manual_seed(seed)
    torch.manual_seed(seed)
    rand = chaos.torch_rand(device, gen)
    ch = chaos.Chaos(preset, exp.ca, device)
    x = model.grow(nca.make_seed(exp.ca, n, exp.grid, device), start)
    c = ch.spark(torch.zeros(n, exp.grid, exp.grid, dtype=torch.long, device=device), rand)
    burnout = torch.full((n,), -1, device=device)
    peak_red = torch.zeros(n, device=device)
    step = start
    while step < max_steps:
        x = ch.clamp(model(x), c)
        if (step - start) % every == 0:
            c, x = ch.tick(c, x, rand)
        step += 1
        red = (c == chaos.RED).flatten(1).float().mean(1)
        peak_red = torch.maximum(peak_red, red)
        burnout = torch.where((burnout < 0) & (red == 0), step, burnout)
        if (burnout >= 0).all() and step >= int(burnout.max()) + settle:
            break
    frac = chaos.counts(c)
    out = {"burnout_step": burnout.cpu().numpy(), "final_nca": frac[:, 0].cpu().numpy(),
           "final_blue": frac[:, 3].cpu().numpy(), "peak_red": peak_red.cpu().numpy(), "steps_run": step}
    if exp.loss == "pixel":
        target = nca.rgb(nca.load_target(exp.target, exp.target_size, exp.grid).to(device), exp.ca)
        out["integrity"] = integrity(x, c, ch, target).cpu().numpy()
    return out


def variant(preset_raw, original=False, **overrides):
    """A preset, optionally reverted to the original rules (no sabotage, dead agents always freeze blue, 1:1 ticks)."""
    p = json.loads(json.dumps(preset_raw))
    if original:
        p["table"] = [row for row in p["table"] if not row.get("sabotage")]
        p.update(black_to_nca=0.0, sabotage_kill=0.0, every=1)
    p.update({k: v for k, v in overrides.items() if v is not None})
    return chaos.cn.compile_preset(p)


def summarize(name, r, start=150, every=1):
    b = r["burnout_step"]
    done = b >= 0
    fizzled = done & (b - start <= 20)
    ticks = (b[done] - start) / every
    line = (f"{name:10s} fizzled {fizzled.mean():4.0%}, finished {done.mean():4.0%}"
            f" in {ticks.mean() if done.any() else float('nan'):4.0f} ticks avg ({np.median(ticks) if done.any() else float('nan'):4.0f} median)"
            f" | final NCA {np.median(r['final_nca']):4.0%}  blue {np.median(r['final_blue']):4.0%}"
            f" | peak red {np.median(r['peak_red']):4.0%}")
    if "integrity" in r:
        i = r["integrity"]
        line += f" | integrity median {np.median(i):.2f}, mountain survives (>0.8) {np.mean(i > 0.8):4.0%}"
    return line


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment", choices=["mountain", "texture"])
    ap.add_argument("--presets", nargs="+", default=["brushfire", "smolder", "wildfire"])
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--start", type=int, default=150)
    ap.add_argument("--every", type=int, default=None, help="NCA steps per chaos tick (default: the preset's)")
    ap.add_argument("--thaw", type=float, default=None)
    ap.add_argument("--empty-defends", action="store_true", help="every NCA cell can retake chaos, not just living ones")
    ap.add_argument("--original", action="store_true", help="the original rules: no sabotage, dead agents always freeze")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    a = ap.parse_args()

    exp = train.EXPERIMENTS[a.experiment]
    model = nca.CAModel(exp.ca).to(a.device).eval()
    model.load_state_dict(torch.load(ROOT / "models" / f"{exp.name}.pth", map_location=a.device))
    raw = json.loads(chaos.PRESETS_PATH.read_text())
    for p in a.presets:
        preset = variant(raw[p], a.original, thaw=a.thaw, every=a.every, empty_defends=a.empty_defends or None)
        t = time.time()
        r = simulate(model, exp, preset, n=a.n, start=a.start, every=preset["every"], device=a.device)
        print(summarize(p, r, a.start, preset["every"]), f"| {r['steps_run']} steps, {time.time() - t:.0f}s", flush=True)
