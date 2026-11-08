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
        self.sabotage = t(preset["sabotage"], torch.bool)
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

    def tick(self, chaos, x, rand):
        """One chaos tick. Returns new (chaos, x). The chaos logic runs without gradients; its effects on x are
        applied with torch.where, so gradients still flow through the cells the NCA keeps."""
        with torch.no_grad():
            c, cleared, seed_mask = self._advance(chaos, x.detach(), rand)
        x = torch.where(cleared[:, None], 0.0, x)
        if seed_mask is not None:
            seed_state = torch.zeros(x.shape[1], device=x.device)
            seed_state[3:] = 1.0
            x = torch.where(seed_mask[:, None], seed_state[None, :, None, None], x)
        return c, self.clamp(x, c)

    def _advance(self, chaos, x, rand):
        """Chaos logic: (new chaos, cells returned to the NCA empty, counter-seed cells or None)."""
        P = self.P
        b, h, w = chaos.shape
        hw = h * w
        c = chaos.clone()
        cleared = torch.zeros_like(c, dtype=torch.bool)

        thaw = (c == BLUE) & (rand((b, h, w)) < P["thaw"])
        c[thaw] = NCA
        cleared |= thaw
        resolve = (c == BLACK) & (rand((b, h, w)) < P["freeze"])
        to_nca = resolve & (rand((b, h, w)) < P["black_to_nca"])
        c[resolve] = BLUE
        c[to_nca] = NCA
        cleared |= to_nca

        living = c == NCA
        if self._alive_mask and not P["empty_defends"]:
            living &= x[:, 3] > 0.1
        p_retake = torch.where(c == RED, P["retake_red"], torch.where(c == BLACK, P["retake_black"], 0.0))
        retake = self._touching(living) & (rand((b, h, w)) < p_retake)
        c[retake] = NCA
        cleared |= retake

        agents = torch.nonzero(c.flatten() == RED)[:, 0]
        roll = (rand((agents.numel(),)) * 100).long().clamp(max=99)
        spread, spawn = self.spread[roll], self.spawn[roll]
        sab = self.sabotage[roll]
        dies = self.die[roll] | sab | (rand((agents.numel(),)) < self.cod[roll])
        saboteurs = torch.zeros(b * hw, dtype=torch.bool, device=c.device)
        saboteurs[agents[sab]] = True
        victims = (self._touching(saboteurs.view(b, h, w)) & (c == RED)
                   & (rand((b, h, w)) < P["sabotage_kill"])).flatten()
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
        flat[victims] = BLACK
        seeds = torch.zeros(0, dtype=torch.long, device=c.device)
        if new_red:
            t = torch.cat(new_red)
            st, u, u_seed = snap[t], rand((t.numel(),)), rand((t.numel(),))
            hit_nca = (st == NCA) & (u >= P["nca_resist"])
            hit_black = (st == BLACK) & (u >= P["black_resist"])
            seeds = t[(st == NCA) & ~hit_nca & (u_seed < P["counter_seed"])]
            flat[t[hit_nca | hit_black]] = RED
        c = flat.view(b, h, w)

        cleared &= c == NCA
        seed_mask = None
        if self._alive_mask and seeds.numel():  # counter-seeds only take root in empty cells
            seed_mask = torch.zeros(b * hw, dtype=torch.bool, device=c.device)
            seed_mask[seeds] = True
            alpha = torch.where(cleared, 0.0, x[:, 3])  # as it will be after clearing
            seed_mask = seed_mask.view(b, h, w) & (c == NCA) & (alpha <= 0.1)
        return c, cleared, seed_mask

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
