"""Check on a local training run: progress, recent loss, and a preview grown from the latest checkpoint.

    python peek.py texture            # writes checkpoints/texture/peek.png
    python peek.py mountain --watch   # refresh the progress line every 10 s (Ctrl-C to stop)
"""
import csv
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

import nca
import train

EXP = train.EXPERIMENTS[sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else "mountain"]
CKPT = Path("checkpoints") / EXP.name
TOTAL = EXP.iters


def progress():
    rows = list(csv.reader(open(CKPT / "train_log.csv")))
    it, loss, lr, secs = int(rows[-1][0]), float(rows[-1][1]), float(rows[-1][2]), float(rows[-1][3])
    eta = secs / it * (TOTAL - it) / 60
    bar = "█" * (30 * it // TOTAL) + "░" * (30 - 30 * it // TOTAL)
    return f"{bar} {it}/{TOTAL}  loss {loss:.5f}  lr {lr:.0e}  {secs / 60:.1f} min elapsed, ~{eta:.0f} min left", rows


if "--watch" in sys.argv:
    while True:
        print("\r" + progress()[0], end="", flush=True)
        time.sleep(10)

line, rows = progress()
print(line)
ckpt = sorted(CKPT.glob(f"{EXP.name}_*.pth"))[-1]
model = nca.CAModel(EXP.ca)
model.load_state_dict(torch.load(ckpt, map_location="cpu"))
with torch.no_grad():
    x, frames = nca.make_seed(EXP.ca, 1, EXP.grid), []
    for step in range(1, 401):
        if step == 201:
            x[..., EXP.grid // 2:] = 0  # cut in half to check regrowth
        x = model(x)
        if step in (25, 50, 100, 200, 225, 300, 400):
            frames.append((step, nca.to_rgb(x, EXP.ca)[0]))

fig, ax = plt.subplots(1, len(frames) + 1, figsize=(2.2 * (len(frames) + 1), 2.4))
for a, (s, f) in zip(ax, frames):
    a.imshow(f); a.axis("off"); a.set_title(f"step {s}", fontsize=9)
ax[-1].plot([int(r[0]) for r in rows], [float(r[1]) for r in rows]); ax[-1].set_yscale("log"); ax[-1].set_title("loss", fontsize=9)
fig.suptitle(f"{ckpt.name} (cut in half at step 200)", fontsize=10)
plt.tight_layout(); plt.savefig(CKPT / "peek.png", dpi=110)
print(f"preview: {CKPT / 'peek.png'}")
