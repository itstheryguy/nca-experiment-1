"""Neural Cellular Automata in PyTorch: the model, seeds, targets, and weight export.

Two kinds of model share this code, chosen by `CAConfig`:
- "mountain" style (Mordvintsev et al., "Growing Neural Cellular Automata", Distill 2020): grows one image from a
  seed cell. Uses an alive mask, zero padding, and premultiplied RGBA in channels 0-3.
- "texture" style (Niklasson et al., "Self-Organising Textures", Distill 2021): every cell is alive, edges wrap
  around, and channels 0-2 are RGB offset by 0.5, trained with a VGG style loss.

The in-browser Space runs a numpy port of `CAModel.forward` (space/nca_numpy.py). Any change to the update rule
must be made in both places; tests/test_parity.py checks that they agree.
"""

import json
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ALIVE_THRESHOLD = 0.1

# 3x3 perception filters. F.conv2d is cross-correlation, and the numpy port matches that.
IDENTITY = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float32)
SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32) / 8.0
SOBEL_Y = SOBEL_X.T.copy()


@dataclass
class CAConfig:
    channels: int = 16
    hidden: int = 128
    fire_rate: float = 0.5
    alive_mask: bool = True   # cells with no neighbor above ALIVE_THRESHOLD alpha are cleared
    wrap: bool = False        # toroidal grid (circular padding) instead of zero padding
    render: str = "rgba"      # "rgba": premultiplied RGBA in 0-3, shown on white. "rgb": RGB in 0-2, offset by 0.5


def alive_mask(x):
    """True where the cell or one of its 8 neighbors has alpha > ALIVE_THRESHOLD. Shape (B, 1, H, W)."""
    return F.max_pool2d(x[:, 3:4], kernel_size=3, stride=1, padding=1) > ALIVE_THRESHOLD


class CAModel(nn.Module):
    def __init__(self, cfg: CAConfig = CAConfig()):
        super().__init__()
        self.cfg = cfg
        c = cfg.channels
        # Depthwise kernels, (3C, 1, 3, 3) ordered [c0_id, c0_sx, c0_sy, c1_id, ...]. Fixed, not trained.
        kernels = torch.from_numpy(np.stack([IDENTITY, SOBEL_X, SOBEL_Y]))[:, None]
        self.register_buffer("kernels", kernels.repeat(c, 1, 1, 1))
        # Per-pixel MLP as 1x1 convs.
        self.w1 = nn.Conv2d(c * 3, cfg.hidden, kernel_size=1)
        # No bias on the output layer: a bias would push every live cell the same way each step.
        self.w2 = nn.Conv2d(cfg.hidden, c, kernel_size=1, bias=False)
        nn.init.zeros_(self.w2.weight)  # do-nothing update at init, so training starts stable

    def perceive(self, x):
        """(B, C, H, W) -> (B, 3C, H, W), interleaved per channel: [c0, c0_dx, c0_dy, c1, ...]."""
        if self.cfg.wrap:
            return F.conv2d(F.pad(x, (1, 1, 1, 1), mode="circular"), self.kernels, groups=self.cfg.channels)
        return F.conv2d(x, self.kernels, padding=1, groups=self.cfg.channels)

    def forward(self, x, fire_rate=None):
        """One CA step."""
        fire_rate = self.cfg.fire_rate if fire_rate is None else fire_rate
        pre_life = alive_mask(x) if self.cfg.alive_mask else None
        dx = self.w2(F.relu(self.w1(self.perceive(x))))
        # Asynchronous update: each cell fires with probability fire_rate.
        fire = (torch.rand(x.shape[0], 1, x.shape[2], x.shape[3], device=x.device) <= fire_rate).float()
        x = x + dx * fire
        if pre_life is not None:
            # A cell lives only if it was alive before and after the update. Dead cells are zeroed.
            x = x * (pre_life & alive_mask(x)).float()
        return x

    def grow(self, x, steps):
        for _ in range(steps):
            x = self(x)
        return x


def make_seed(cfg: CAConfig, n=1, grid=64, device="cpu"):
    """Starting state. Mountain style: one live cell in the center (alpha and hidden = 1). Texture style: zeros."""
    x = torch.zeros(n, cfg.channels, grid, grid, device=device)
    if cfg.alive_mask:
        x[:, 3:, grid // 2, grid // 2] = 1.0
    return x


def load_target(path, size=44, grid=64):
    """Emoji as a (4, grid, grid) premultiplied RGBA tensor, `size` px wide, centered with margin."""
    from PIL import Image

    img = Image.open(path).convert("RGBA")
    img = img.crop(img.getbbox())  # drop transparent border so `size` is the emoji itself
    img.thumbnail((size, size), Image.LANCZOS)
    a = np.asarray(img, dtype=np.float32) / 255.0
    a[..., :3] *= a[..., 3:4]  # premultiply, so a transparent pixel is all zeros
    out = np.zeros((grid, grid, 4), dtype=np.float32)
    h, w = a.shape[:2]
    top, left = (grid - h) // 2, (grid - w) // 2
    out[top:top + h, left:left + w] = a
    return torch.from_numpy(out).permute(2, 0, 1).contiguous()


def load_texture(path, size=128):
    """Texture image as a (3, size, size) RGB tensor in [0, 1]."""
    from PIL import Image

    img = Image.open(path).convert("RGB").resize((size, size), Image.LANCZOS)
    return torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1).contiguous()


def rgb(x, cfg: CAConfig):
    """Displayable RGB in [0, 1] from a (..., C, H, W) state, as a tensor (..., 3, H, W)."""
    if cfg.render == "rgb":
        return (x[..., :3, :, :] + 0.5).clamp(0, 1)
    a = x[..., 3:4, :, :].clamp(0, 1)
    return (1.0 - a + x[..., :3, :, :]).clamp(0, 1)


def to_rgb(x, cfg: CAConfig):
    """Like `rgb`, as (..., H, W, 3) numpy for plotting."""
    return rgb(x.detach().cpu(), cfg).movedim(-3, -1).numpy()


def damage(x, n_circles=1):
    """Zero out a random circle (radius 0.1-0.4 of the grid) in each sample of x, in place."""
    b, _, h, w = x.shape
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, h, device=x.device),
                            torch.linspace(-1, 1, w, device=x.device), indexing="ij")
    for _ in range(n_circles):
        center = torch.rand(b, 2, 1, 1, device=x.device) - 0.5  # within the middle half
        r = 0.1 + 0.3 * torch.rand(b, 1, 1, device=x.device)
        mask = ((xx - center[:, 0]) ** 2 + (yy - center[:, 1]) ** 2) < r ** 2
        x.mul_((~mask).unsqueeze(1).float())
    return x


def export_npz(model, path, grid, name):
    """Save weights and settings as plain arrays for the numpy runtime (space/nca_numpy.py)."""
    sd = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
    np.savez(
        path,
        w1=sd["w1.weight"][:, :, 0, 0].astype(np.float32),  # (hidden, 3C)
        b1=sd["w1.bias"].astype(np.float32),                # (hidden,)
        w2=sd["w2.weight"][:, :, 0, 0].astype(np.float32),  # (C, hidden)
        config=np.array(json.dumps(asdict(model.cfg) | {"grid": grid, "name": name})),
    )
