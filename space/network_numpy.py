"""The network task: a hub and scattered targets the NCA must connect with as little network as possible.

State channels for network models: 0 is the network density the NCA grows (shown as mycelium; read as clip(x, 0,
1)), 1 marks the hub, 2 marks targets. Channels 1-2 are inputs: they're written back after every step, so the NCA
sees them but can't change them. The rest are hidden. Inspired by Physarum polycephalum's Tokyo rail network
(Tero et al., Science 2010).

Shared by the Space (numpy) and training (layouts are generated here, then moved to torch).
"""

import numpy as np

NET, HUB, TARGET = 0, 1, 2
BACKGROUND = np.array([247, 244, 236], np.float32)
MYCELIUM = np.array([92, 64, 51], np.float32)
HUB_COLOR = np.array([232, 163, 23], np.uint8)
TARGET_COLOR = np.array([20, 140, 150], np.uint8)


def make_layout(rng, grid=64, k_range=(3, 6), margin=4, min_dist=12, tries=1000):
    """Random hub and k targets, at least min_dist apart (Chebyshev) and margin from the edges.
    Returns (hub, targets) as (y, x) tuples."""
    k = int(rng.integers(k_range[0], k_range[1] + 1))
    pts = []
    for _ in range(tries):
        p = tuple(int(v) for v in rng.integers(margin, grid - margin, size=2))
        if all(max(abs(p[0] - q[0]), abs(p[1] - q[1])) >= min_dist for q in pts):
            pts.append(p)
            if len(pts) == k + 1:
                break
    return pts[0], pts[1:]


def inputs(layout, grid=64):
    """(2, grid, grid) float32: hub mask, target mask."""
    hub, targets = layout
    m = np.zeros((2, grid, grid), np.float32)
    m[0][hub] = 1.0
    for t in targets:
        m[1][t] = 1.0
    return m


def clamp(x, inp):
    """Write the input channels back into the state (C, H, W)."""
    x = x.copy()
    x[HUB:TARGET + 1] = inp
    return x


def seed(cfg, layout):
    x = np.zeros((cfg["channels"], cfg["grid"], cfg["grid"]), np.float32)
    return clamp(x, inputs(layout, cfg["grid"]))


def density(x):
    return np.clip(x[NET], 0, 1)


def render(x):
    """(H, W, 3) uint8: mycelium density on a light background, hub and targets on top."""
    n = density(x)[..., None]
    img = BACKGROUND * (1 - n) + MYCELIUM * n
    img = img.astype(np.uint8)
    img[x[TARGET] > 0.5] = TARGET_COLOR
    img[x[HUB] > 0.5] = HUB_COLOR
    return img


def connected(x, thresh=0.5):
    """Fraction of targets reachable from the hub through network cells (8-connected, density > thresh)."""
    passable = (density(x) > thresh) | (x[HUB] > 0.5) | (x[TARGET] > 0.5)
    reach = x[HUB] > 0.5
    while True:
        p = np.pad(reach, 1)
        h, w = reach.shape
        grown = np.max([p[i:i + h, j:j + w] for i in range(3) for j in range(3)], axis=0) & passable
        if (grown == reach).all():
            break
        reach = grown
    targets = x[TARGET] > 0.5
    return float((reach & targets).sum() / max(targets.sum(), 1))


def mst_cells(layout):
    """Cells in a minimum spanning tree of hub and targets with 8-connected moves (Chebyshev lengths): the
    baseline network size. A Steiner tree can be shorter by sharing trunks."""
    pts = [layout[0], *layout[1]]
    d = lambda a, b: max(abs(a[0] - b[0]), abs(a[1] - b[1]))  # noqa: E731
    in_tree, total = {0}, 0
    while len(in_tree) < len(pts):
        best = min((d(pts[i], pts[j]), j) for i in in_tree for j in range(len(pts)) if j not in in_tree)
        total += best[0]
        in_tree.add(best[1])
    return total
