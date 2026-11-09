"""Training for every experiment: the pool loop, losses, and experiment configs.

train.ipynb is a thin driver around `Trainer`, and runs the same locally (MPS) or on Colab (CUDA).
"""

import csv
import json
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import nca
from nca import CAConfig


@dataclass
class Experiment:
    name: str
    ca: CAConfig
    target: str                         # image path
    loss: str                           # "pixel" (MSE to a fixed RGBA image), "style" (VGG Gram matrices), or
                                        # "network" (connect hub and targets cheaply, see network.py)
    grid: int = 64
    target_size: int = 44               # pixel loss: emoji width in px within the grid
    pool_size: int = 1024
    batch: int = 8
    steps: tuple = (64, 96)             # CA steps per iteration, sampled uniformly
    iters: int = 8000
    lr: float = 2e-3
    lr_drop_at: tuple = (2000,)         # lr x0.1 at each
    n_damaged: int = 3                  # pool samples per batch hit with an erased circle
    reseed_every: int = 1               # replace one pool sample with a fresh seed every N iterations
    overflow_weight: float = 0.0        # penalty for state values outside [-1, 1]
    chaos: dict | None = None           # chaos agent settings, see CHAOS below
    init_from: str | None = None        # start from these weights (.pth) instead of from scratch
    network: dict | None = None         # network task: targets (min, max), empty_cost, material_weight
    notes: str = field(default="", repr=False)


MOUNTAIN = Experiment(
    name="mountain",
    ca=CAConfig(channels=16, hidden=128, alive_mask=True, wrap=False, render="rgba"),
    target="data/noto_1f3d4_snowcapped.png",
    loss="pixel",
)

TEXTURE = Experiment(
    name="texture",
    # Smaller than the mountain model: every cell is alive, so the browser runs the MLP on the whole grid.
    ca=CAConfig(channels=12, hidden=96, alive_mask=False, wrap=True, render="rgb"),
    target="data/topo_texture.png",
    loss="style",
    batch=4,
    steps=(32, 64),
    iters=6000,
    n_damaged=1,
    reseed_every=8,
    overflow_weight=1e-2,
)

# Chaos training: each iteration picks one preset at random, so the model sees light to heavy chaos. Pool samples
# without active agents get a spark with probability spark_prob. Samples where chaos holds more than clear_above of
# the grid, or burned out (with probability clear_burned_out), have their chaos cleared back to empty NCA cells, so
# the pool keeps a mix of fresh fights, ongoing fights, and aftermaths to regrow from.
#
# v1.1 (after v1 came out worse than the base mountain):
# - counter_seed is turned off in training. Counter-seeds grow extra mountains, which the one-mountain target
#   punishes, so v1 learned to resist growing from seeds: slower, hazier growth and ~1% of seeds dying out.
#   The Space keeps counter-seeds on, so the model meets them there as something new.
# - chaos_free: this share of iterations runs with no chaos ticks or sparks, so plain growth keeps being practiced.
# - Fine-tuning learning rate 1e-3 -> 5e-4 (in the experiment below).
CHAOS = {"presets": ["smolder", "brushfire", "wildfire"], "spark_prob": 0.3, "clear_above": 0.5,
         "clear_burned_out": 0.3, "overrides": {"counter_seed": 0.0}, "chaos_free": 0.5}

# Connect a random hub to 3-6 random targets with as little network as possible. Every cell is active (no alive
# mask, no wrap). Steps reach 128 because signals spread at most one cell per step on a 64-cell grid.
# Material cost ramps in (explore, then prune, like Physarum): with it on from the start, the cost on all 4,096
# cells outweighed the route signal on the ~100 cells of each path, and the network channel went dead.
NETWORK = Experiment(
    name="network",
    ca=CAConfig(channels=16, hidden=128, alive_mask=False, wrap=False, render="network"),
    target="",
    loss="network",
    steps=(64, 128),
    iters=8000,
    n_damaged=2,
    overflow_weight=1.0,
    network={"targets": (3, 6), "empty_cost": 10.0, "material_weight": 2.0, "material_ramp": (1500, 4000)},
)

EXPERIMENTS = {
    "mountain": MOUNTAIN,
    "texture": TEXTURE,
    "network": NETWORK,
    # Fine-tuned from the base models: they already grow; here they learn to live with chaos.
    "mountain_chaos": replace(MOUNTAIN, name="mountain_chaos", chaos=CHAOS, init_from="models/mountain.pth",
                              iters=4000, lr=5e-4, lr_drop_at=(2500,), n_damaged=1, notes="v1.1"),
    "texture_chaos": replace(TEXTURE, name="texture_chaos", chaos=CHAOS, init_from="models/texture.pth",
                             iters=3000, lr=1e-3, lr_drop_at=(2000,), n_damaged=0),
}


class StyleLoss(torch.nn.Module):
    """Per-sample Gram-matrix style loss on VGG16 features (Gatys et al.), as in Self-Organising Textures."""

    LAYERS = (1, 6, 11, 18, 25)  # relu1_1, relu2_1, relu3_1, relu4_1, relu5_1

    def __init__(self, target_rgb, device):
        super().__init__()
        from torchvision.models import VGG16_Weights, vgg16

        self.vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[: max(self.LAYERS) + 1].to(device).eval()
        for p in self.vgg.parameters():
            p.requires_grad_(False)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406], device=device)[:, None, None])
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225], device=device)[:, None, None])
        with torch.no_grad():
            self.target_grams = [g.detach() for g in self.grams(target_rgb[None].to(device))]

    def grams(self, img):
        x, out = (img - self.mean) / self.std, []
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            if i in self.LAYERS:
                f = x.flatten(2)
                out.append(f @ f.transpose(1, 2) / f.shape[-1])
        return out

    def forward(self, img):
        """(B, 3, H, W) in [0, 1] -> (B,) loss."""
        # Each layer normalized by the target's Gram magnitude, so layers weigh equally and the total is ~1 at start.
        return sum(((g - t) ** 2).mean(dim=(1, 2)) / (t ** 2).mean() for g, t in zip(self.grams(img), self.target_grams))


class Trainer:
    """Holds model, optimizer, and pool, so `run` can be called again to continue training."""

    def __init__(self, exp: Experiment, device, out_dir="checkpoints", seed=0):
        torch.manual_seed(seed)
        self.exp, self.device = exp, device
        self.model = nca.CAModel(exp.ca).to(device)
        if exp.init_from:
            self.model.load_state_dict(torch.load(exp.init_from, map_location=device))
        self.opt = torch.optim.Adam(self.model.parameters(), lr=exp.lr)
        self.sched = torch.optim.lr_scheduler.MultiStepLR(self.opt, milestones=list(exp.lr_drop_at), gamma=0.1)
        self.pool = nca.make_seed(exp.ca, exp.pool_size, exp.grid, device)
        if exp.network:
            import network

            self.net = network
            self.np_rng = np.random.default_rng(seed)
            self.pool_inputs = self.new_layouts(exp.pool_size)
            self.pool = network.seed(exp.ca.channels, self.pool_inputs)
        self.pool_chaos = torch.zeros(exp.pool_size, exp.grid, exp.grid, dtype=torch.long, device=device)
        if exp.chaos:
            import chaos

            raw = json.loads(chaos.PRESETS_PATH.read_text())
            overrides = exp.chaos.get("overrides", {})
            self.chaos = [chaos.Chaos(chaos.cn.compile_preset(raw[name] | overrides), exp.ca, device)
                          for name in exp.chaos["presets"]]
            self.rand = chaos.torch_rand(device)
            if exp.init_from:  # start the pool from grown states, so chaos meets a formed pattern
                with torch.no_grad():
                    for i in range(0, exp.pool_size, 256):
                        self.pool[i:i + 256] = self.model.grow(self.pool[i:i + 256], exp.steps[1])
        self.losses, self.it = [], 0
        self.out_dir = Path(out_dir) / exp.name
        self.out_dir.mkdir(parents=True, exist_ok=True)
        if exp.loss == "pixel":
            self.target = nca.load_target(exp.target, exp.target_size, exp.grid).to(device)
        elif exp.loss == "style":
            self.target = nca.load_texture(exp.target, size=2 * exp.grid).to(device)
            self.style = StyleLoss(self.target, device)

    def new_layouts(self, n):
        return self.net.layouts(n, self.exp.grid, self.np_rng, self.device, k_range=self.exp.network["targets"])

    def per_sample_loss(self, x, c=None):
        if self.exp.loss == "pixel":
            err = (x[:, :4] - self.target) ** 2
            if c is None:
                loss = err.mean(dim=(1, 2, 3))
            else:  # chaos cells are out of the NCA's control: score only the cells it owns
                m = (c == 0)[:, None].float()
                loss = (err * m).sum(dim=(1, 2, 3)) / (4 * m.sum(dim=(1, 2, 3))).clamp(min=1)
        elif self.exp.loss == "network":
            cfg = self.exp.network
            lo, hi = cfg.get("material_ramp", (0, 0))
            ramp = min(max((self.it - lo) / max(hi - lo, 1), 0.0), 1.0) if hi else 1.0
            loss = self.net.loss(x, cfg["empty_cost"], cfg["material_weight"] * ramp)
        else:
            img = F.interpolate(nca.rgb(x, self.exp.ca), scale_factor=2, mode="bilinear", align_corners=False)
            loss = self.style(img)
        if self.exp.overflow_weight:
            loss = loss + self.exp.overflow_weight * (x - x.clamp(-1, 1)).abs().mean(dim=(1, 2, 3))
        return loss

    def batch(self):
        exp = self.exp
        self.chaos_free = bool(exp.chaos) and torch.rand(()).item() < exp.chaos.get("chaos_free", 0.0)
        idx = torch.randperm(exp.pool_size, device=self.device)[: exp.batch]
        x, c = self.pool[idx].clone(), self.pool_chaos[idx].clone()
        with torch.no_grad():
            if exp.loss in ("pixel", "network"):  # worst first, so the worst sample is the one reseeded
                order = self.per_sample_loss(x, c if exp.chaos else None).argsort(descending=True)
                x, c, idx = x[order], c[order], idx[order]
            reseeded = self.it % exp.reseed_every == 0
            if exp.network:
                inp = self.pool_inputs[idx].clone()
                if reseeded:  # a fresh layout, grown from nothing
                    inp[:1] = self.new_layouts(1)
                    x[:1] = self.net.seed(exp.ca.channels, inp[:1])
                    self.pool_inputs[idx[0]] = inp[0]
                if exp.n_damaged:
                    self.net.damage(x[-exp.n_damaged:], inp[-exp.n_damaged:])
                self.inputs = inp
                return x, c, idx
            if reseeded:
                x[:1] = nca.make_seed(exp.ca, 1, exp.grid, self.device)
                c[:1] = 0
            if exp.n_damaged:
                nca.damage(x[-exp.n_damaged:])
            if exp.chaos:
                x, c = self.manage_chaos(x, c, reseeded)
        return x, c, idx

    def manage_chaos(self, x, c, reseeded):
        """Clear saturated or burned-out chaos, and spark samples that have no active agents."""
        cfg, b = self.exp.chaos, x.shape[0]
        frac_chaos = (c > 0).flatten(1).float().mean(1)
        burned_out = ((c > 0).flatten(1).any(1)) & ~((c == 1).flatten(1).any(1))
        clear = (frac_chaos > cfg["clear_above"]) | (burned_out & (torch.rand(b, device=c.device) < cfg["clear_burned_out"]))
        if clear.any():
            x = torch.where((clear[:, None, None] & (c > 0))[:, None], 0.0, x)
            c = torch.where(clear[:, None, None], 0, c)
        no_agents = ~(c == 1).flatten(1).any(1)
        spark = no_agents & (torch.rand(b, device=c.device) < cfg["spark_prob"])
        if self.chaos_free:
            spark[:] = False
        if reseeded:
            spark[0] = False  # let a fresh seed grow before chaos arrives
        c = self.chaos[0].spark(c, self.rand, spark)
        return self.chaos[0].clamp(x, c), c

    def grow(self, x, c, steps):
        if self.exp.network:
            for _ in range(steps):
                x = self.net.clamp(self.model(x), self.inputs)
            return x, c
        if not self.exp.chaos:
            return self.model.grow(x, steps), c
        if self.chaos_free:  # no ticks: existing chaos cells stay as they are, the NCA just grows
            for _ in range(steps):
                x = self.chaos[0].clamp(self.model(x), c)
            return x, c
        ch = self.chaos[int(torch.randint(len(self.chaos), ()))]
        every = int(ch.P["every"])
        for i in range(steps):
            x = ch.clamp(self.model(x), c)
            if i % every == 0:
                c, x = ch.tick(c, x, self.rand)
        return x, c

    def run(self, iters=None, on_progress=None, every=100):
        """Train until iteration `iters` (default: the experiment's). Calls on_progress(trainer, before, after)
        every `every` iterations. Stops cleanly on KeyboardInterrupt."""
        iters = iters or self.exp.iters
        log_path = self.out_dir / "train_log.csv"
        t0 = time.time()
        try:
            while self.it < iters:
                x, c, idx = self.batch()
                before = x.detach()
                lo, hi = self.exp.steps
                x, c = self.grow(x, c, int(torch.randint(lo, hi + 1, ())))
                loss = self.per_sample_loss(x, c if self.exp.chaos else None).mean()
                self.opt.zero_grad()
                loss.backward()
                for p in self.model.parameters():  # per-parameter gradient normalization
                    p.grad /= p.grad.norm() + 1e-8
                self.opt.step()
                self.sched.step()
                self.pool[idx] = x.detach()
                self.pool_chaos[idx] = c
                self.last_chaos = c
                self.losses.append(loss.item())
                self.it += 1
                if self.it % every == 0:
                    elapsed = time.time() - t0
                    with open(log_path, "a", newline="") as f:
                        csv.writer(f).writerow([self.it, np.mean(self.losses[-every:]),
                                                self.sched.get_last_lr()[0], round(elapsed)])
                    if on_progress:
                        on_progress(self, before, x.detach())
                if self.it % 1000 == 0:
                    torch.save(self.model.state_dict(), self.out_dir / f"{self.exp.name}_{self.it:05d}.pth")
        except KeyboardInterrupt:
            print(f"stopped at iteration {self.it}")
        return self
