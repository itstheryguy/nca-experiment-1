"""The chaos agent: a stochastic cellular process that fights the NCA for the grid. Numpy, for the Space.

Cells are NCA (0), RED (1, an active agent), BLACK (2, a dead agent), or BLUE (3, frozen). One chaos tick:

1. Thaw: each BLUE becomes an empty NCA cell with probability `thaw`.
2. Freeze: each BLACK becomes BLUE with probability `freeze`.
3. Retake: each RED or BLACK cell touching a living NCA cell (8 neighbors) is reclaimed by the NCA with
   probability `retake_red` / `retake_black`, and becomes an empty NCA cell. Living: alpha > 0.1 for models with
   an alive mask, any NCA cell otherwise.
4. Agents act: each RED rolls 0-99 and looks up the preset's table. `spread` claims the nearest NCA or BLACK
   cells, ring by ring (random order within a ring). NCA cells are always taken; BLACK cells resist with
   probability `black_resist`. `cod` is the chance the agent dies (becomes BLACK) after acting, `die` kills it
   outright, and `spawn` places new agents at random cells anywhere.

Chaos cells are written into the NCA state as fixed colors (alpha 0, hidden channels 0), so the NCA perceives them
through its normal 3x3 view but can't change them, and they never count as alive.

All randomness comes from `rand(shape) -> uniform [0, 1) array`, called in a fixed order, so chaos.py (the batched
PyTorch version used for training and Monte Carlo runs) can be checked against this file draw for draw.
"""

import json
from pathlib import Path

import numpy as np

NCA, RED, BLACK, BLUE = 0, 1, 2, 3
COLORS = np.array([[0, 0, 0], [220, 38, 38], [17, 17, 17], [37, 99, 235]], dtype=np.uint8)  # display, by state
SIGNATURE = np.array([[0, 0, 0], [1, 0, 0], [0, 0, 0], [0, 0, 1]], dtype=np.float32)       # RGB the NCA sees
MAX_RADIUS = 6

# Neighborhood offsets within MAX_RADIUS, nearest first.
_oy, _ox = np.mgrid[-MAX_RADIUS:MAX_RADIUS + 1, -MAX_RADIUS:MAX_RADIUS + 1]
_d2 = (_oy ** 2 + _ox ** 2).ravel()
_keep = (_d2 > 0) & (_d2 <= MAX_RADIUS ** 2)
_order = np.argsort(_d2[_keep], kind="stable")
OFF_Y, OFF_X, OFF_D2 = _oy.ravel()[_keep][_order], _ox.ravel()[_keep][_order], _d2[_keep][_order]


def candidates(n):
    """How many nearest offsets to consider for a spread of n: whole rings, at least 8 and about 3n cells."""
    k = min(max(8, 3 * n), len(OFF_D2)) - 1
    return int(np.searchsorted(OFF_D2, OFF_D2[k], side="right"))


def load_presets(src):
    """Presets from a JSON path (or an already-parsed dict), compiled."""
    raw = src if isinstance(src, dict) else json.loads(Path(src).read_text())
    return {k: compile_preset(v) for k, v in raw.items() if not k.startswith("_")}


def compile_preset(p):
    """Preset dict -> lookup arrays indexed by roll (0-99), plus the scalar rates."""
    spread, cod, spawn, die = np.zeros(100, np.int64), np.zeros(100), np.zeros(100, np.int64), np.zeros(100, bool)
    for row in p["table"]:
        r = slice(row["lo"], row["hi"] + 1)
        spread[r], cod[r], spawn[r], die[r] = row.get("spread", 0), row.get("cod", 0.0), row.get("spawn", 0), row.get("die", False)
    out = {k: p[k] for k in ("freeze", "thaw", "retake_red", "retake_black", "black_resist")}
    return out | {"label": p.get("label", ""), "spread": spread, "cod": cod, "spawn": spawn, "die": die}


def signature(cfg):
    """(4, C) state written into RED/BLACK/BLUE cells (row 0, NCA, unused). RGB in the model's render space."""
    sig = np.zeros((4, cfg["channels"]), np.float32)
    sig[:, :3] = SIGNATURE - (0.5 if cfg["render"] == "rgb" else 0.0)
    return sig


def clamp(x, chaos, cfg):
    """Overwrite chaos cells in the NCA state (C, H, W) with their fixed signature."""
    m = chaos > 0
    if m.any():
        x = x.copy()
        x[:, m] = signature(cfg)[chaos[m]].T
    return x


def _touching(mask, wrap):
    """Cells with at least one True among their 8 neighbors (or themselves)."""
    h, w = mask.shape
    p = np.pad(mask, 1, mode="wrap" if wrap else "constant")
    return np.max([p[i:i + h, j:j + w] for i in range(3) for j in range(3)], axis=0)


def tick(chaos, x, cfg, P, rand):
    """One chaos tick. chaos (H, W) int8, x (C, H, W) NCA state. Returns new (chaos, x)."""
    h, w = chaos.shape
    wrap = cfg["wrap"]
    c = chaos.copy()
    cleared = np.zeros((h, w), bool)

    thaw = (c == BLUE) & (rand((h, w)) < P["thaw"])
    c[thaw] = NCA
    cleared |= thaw
    c[(c == BLACK) & (rand((h, w)) < P["freeze"])] = BLUE

    living = (c == NCA) & (x[3] > 0.1 if cfg["alive_mask"] else True)
    p_retake = np.where(c == RED, P["retake_red"], np.where(c == BLACK, P["retake_black"], 0.0))
    retake = _touching(living, wrap) & (rand((h, w)) < p_retake)
    c[retake] = NCA
    cleared |= retake

    agents = np.flatnonzero(c == RED)
    roll = np.minimum((rand((agents.size,)) * 100).astype(np.int64), 99)
    spread, spawn = P["spread"][roll], P["spawn"][roll]
    dies = P["die"][roll] | (rand((agents.size,)) < P["cod"][roll])
    snap = c.ravel().copy()
    new_red = []

    for n in np.unique(spread[spread > 0]):  # agents grouped by spread size, nearest-cell search per group
        a = agents[spread == n]
        m = candidates(int(n))
        ty, tx = a[:, None] // w + OFF_Y[None, :m], a[:, None] % w + OFF_X[None, :m]
        if wrap:
            ty, tx, valid = ty % h, tx % w, np.ones(ty.shape, bool)
        else:
            valid = (ty >= 0) & (ty < h) & (tx >= 0) & (tx < w)
            ty, tx = np.clip(ty, 0, h - 1), np.clip(tx, 0, w - 1)
        cell = ty * w + tx
        target_ok = valid & ((snap[cell] == NCA) | (snap[cell] == BLACK))
        key = np.where(target_ok, OFF_D2[None, :m] + rand((a.size, m)), np.inf)
        order = np.argsort(key, axis=1, kind="stable")[:, :n]
        chosen = np.take_along_axis(cell, order, 1)[np.isfinite(np.take_along_axis(key, order, 1))]
        new_red.append(chosen)

    k = int(spawn.sum())
    if k:
        new_red.append(np.minimum((rand((k,)) * h * w).astype(np.int64), h * w - 1))

    flat = c.ravel()
    flat[agents[dies]] = BLACK
    if new_red:
        t = np.concatenate(new_red)
        t = t[(snap[t] == NCA) | ((snap[t] == BLACK) & (rand((t.size,)) >= P["black_resist"]))]
        flat[t] = RED
    c = flat.reshape(h, w)

    x = x.copy()
    x[:, cleared & (c == NCA)] = 0
    return c, clamp(x, c, cfg)


def spark(chaos, rand):
    """Place one agent at a random cell (if it isn't already chaos)."""
    h, w = chaos.shape
    i = min(int(rand((1,))[0] * h * w), h * w - 1)
    c = chaos.copy()
    if c.flat[i] == NCA:
        c.flat[i] = RED
    return c


def counts(chaos):
    """Fraction of cells in each state: {"nca", "red", "black", "blue"}."""
    n = np.bincount(chaos.ravel(), minlength=4) / chaos.size
    return dict(zip(("nca", "red", "black", "blue"), n.tolist()))


def overlay(rgb, chaos):
    """Paint chaos cells over an (H, W, 3) uint8 NCA image."""
    out = rgb.copy()
    m = chaos > 0
    out[m] = COLORS[chaos[m]]
    return out
