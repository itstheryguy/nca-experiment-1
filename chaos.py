"""The chaos agent, batched in PyTorch for training and Monte Carlo runs.

Same rules as space/chaos_numpy.py (read its docstring for them), over a batch: chaos is (B, H, W), the NCA state
(B, C, H, W). Random numbers come from `rand(shape)` in the same order as the numpy version, so with B = 1 the two
agree draw for draw (tests/test_chaos_parity.py).
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent / "space"))
import chaos_numpy as cn  # noqa: E402  (shared constants and preset compiler)

NCA, RED, BLACK, BLUE = cn.NCA, cn.RED, cn.BLACK, cn.BLUE
PRESETS_PATH = Path(__file__).parent / "space" / "chaos_presets.json"


def load_presets(src=PRESETS_PATH):
    return cn.load_presets(src)


def torch_rand(device, generator=None):
    return lambda shape: torch.rand(shape, device=device, generator=generator)


class Chaos:
    """Preset tables as tensors on one device."""

    def __init__(self, preset, cfg, device):
        self.P = preset
        self.cfg = cfg
        self.device = device
        t = lambda a, dt=None: torch.as_tensor(a, device=device, dtype=dt)  # noqa: E731
        self.spread, self.cod = t(preset["spread"], torch.long), t(preset["cod"], torch.float32)
        self.spawn, self.die = t(preset["spawn"], torch.long), t(preset["die"], torch.bool)
        self.off_y, self.off_x = t(cn.OFF_Y, torch.long), t(cn.OFF_X, torch.long)
        self.off_d2 = t(cn.OFF_D2, torch.float32)
        self.sig = t(cn.signature(cfg if isinstance(cfg, dict) else vars(cfg)), torch.float32)

    def clamp(self, x, chaos):
        """Overwrite chaos cells with their fixed signature. Differentiable w.r.t. the other cells."""
        sig = self.sig[chaos].permute(0, 3, 1, 2)  # (B, C, H, W)
        return torch.where((chaos > 0)[:, None], sig, x)

    def _touching(self, mask):
        m = mask[:, None].float()
        m = F.pad(m, (1, 1, 1, 1), mode="circular") if self._wrap else F.pad(m, (1, 1, 1, 1))
        return F.max_pool2d(m, 3, stride=1)[:, 0] > 0

    @property
    def _wrap(self):
        return self.cfg["wrap"] if isinstance(self.cfg, dict) else self.cfg.wrap

    @property
    def _alive_mask(self):
        return self.cfg["alive_mask"] if isinstance(self.cfg, dict) else self.cfg.alive_mask

    @torch.no_grad()
    def tick(self, chaos, x, rand):
        """One chaos tick. Returns new (chaos, x). x keeps its autograd graph; only chaos changes are applied."""
        P = self.P
        b, h, w = chaos.shape
        hw = h * w
        c = chaos.clone()
        cleared = torch.zeros_like(c, dtype=torch.bool)

        thaw = (c == BLUE) & (rand((b, h, w)) < P["thaw"])
        c[thaw] = NCA
        cleared |= thaw
        c[(c == BLACK) & (rand((b, h, w)) < P["freeze"])] = BLUE

        living = c == NCA
        if self._alive_mask:
            living &= x[:, 3] > 0.1
        p_retake = torch.where(c == RED, P["retake_red"], torch.where(c == BLACK, P["retake_black"], 0.0))
        retake = self._touching(living) & (rand((b, h, w)) < p_retake)
        c[retake] = NCA
        cleared |= retake

        agents = torch.nonzero(c.flatten() == RED)[:, 0]
        roll = (rand((agents.numel(),)) * 100).long().clamp(max=99)
        spread, spawn = self.spread[roll], self.spawn[roll]
        dies = self.die[roll] | (rand((agents.numel(),)) < self.cod[roll])
        snap = c.flatten().clone()
        new_red = []

        for n in torch.unique(spread[spread > 0]).tolist():  # sorted ascending, like np.unique
            a = agents[spread == n]
            m = cn.candidates(n)
            bi, r = a // hw, a % hw
            ty = (r // w)[:, None] + self.off_y[None, :m]
            tx = (r % w)[:, None] + self.off_x[None, :m]
            if self._wrap:
                ty, tx, valid = ty % h, tx % w, torch.ones_like(ty, dtype=torch.bool)
            else:
                valid = (ty >= 0) & (ty < h) & (tx >= 0) & (tx < w)
                ty, tx = ty.clamp(0, h - 1), tx.clamp(0, w - 1)
            cell = bi[:, None] * hw + ty * w + tx
            target_ok = valid & ((snap[cell] == NCA) | (snap[cell] == BLACK))
            key = torch.where(target_ok, self.off_d2[None, :m] + rand((a.numel(), m)), torch.inf)
            order = torch.argsort(key, dim=1, stable=True)[:, :n]
            chosen = torch.gather(cell, 1, order)[torch.isfinite(torch.gather(key, 1, order))]
            new_red.append(chosen)

        k = int(spawn.sum())
        if k:
            owner = torch.repeat_interleave(agents // hw, spawn)  # spawns land in the agent's own grid
            new_red.append(owner * hw + (rand((k,)) * hw).long().clamp(max=hw - 1))

        flat = c.flatten()
        flat[agents[dies]] = BLACK
        if new_red:
            t = torch.cat(new_red)
            st = snap[t]
            t = t[(st == NCA) | ((st == BLACK) & (rand((t.numel(),)) >= P["black_resist"]))]
            flat[t] = RED
        c = flat.view(b, h, w)

        x = torch.where((cleared & (c == NCA))[:, None], 0.0, x)
        return c, self.clamp(x, c)

    @torch.no_grad()
    def spark(self, chaos, rand, mask=None):
        """One agent at a random cell in each grid (or each grid where mask is True), if that cell is NCA."""
        b, h, w = chaos.shape
        i = (rand((b,)) * h * w).long().clamp(max=h * w - 1)
        flat = chaos.clone().view(b, -1)
        ok = flat[torch.arange(b, device=chaos.device), i] == NCA
        if mask is not None:
            ok &= mask
        flat[torch.arange(b, device=chaos.device)[ok], i[ok]] = RED
        return flat.view(b, h, w)


def counts(chaos):
    """(B, 4) fraction of cells in each state, ordered NCA, RED, BLACK, BLUE."""
    b = chaos.shape[0]
    return F.one_hot(chaos.view(b, -1).long(), 4).float().mean(1)
