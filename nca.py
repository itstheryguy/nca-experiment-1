"""Growing Neural Cellular Automata in PyTorch.

Based on Mordvintsev et al., "Growing Neural Cellular Automata" (Distill, 2020).
The model, seed, target loading, and helpers shared by training and export live here.
The in-browser Space runs a numpy port of `CAModel.forward` (space/nca_numpy.py), so any
change to the update rule must be made in both places. `tests/test_parity.py` checks they agree.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CHANNELS = 16  # 0-3 RGBA (premultiplied), 4-15 hidden
GRID = 64
HIDDEN = 128
ALIVE_THRESHOLD = 0.1
FIRE_RATE = 0.5

# 3x3 perception filters. F.conv2d is cross-correlation, and the numpy port matches that.
IDENTITY = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float32)
SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32) / 8.0
SOBEL_Y = SOBEL_X.T.copy()


def alive_mask(x):
    """True where the cell or one of its 8 neighbors has alpha > ALIVE_THRESHOLD. Shape (B, 1, H, W)."""
    return F.max_pool2d(x[:, 3:4], kernel_size=3, stride=1, padding=1) > ALIVE_THRESHOLD


class CAModel(nn.Module):
    def __init__(self, channels=CHANNELS, hidden=HIDDEN, fire_rate=FIRE_RATE):
        super().__init__()
        self.channels = channels
        self.fire_rate = fire_rate
        # Depthwise kernels, (3C, 1, 3, 3) ordered [c0_id, c0_sx, c0_sy, c1_id, ...]. Fixed, not trained.
        kernels = torch.from_numpy(np.stack([IDENTITY, SOBEL_X, SOBEL_Y]))[:, None]
        self.register_buffer("kernels", kernels.repeat(channels, 1, 1, 1))
        # Per-pixel MLP as 1x1 convs.
        self.w1 = nn.Conv2d(channels * 3, hidden, kernel_size=1)
        # No bias on the output layer: a bias would push every live cell the same way each step.
        self.w2 = nn.Conv2d(hidden, channels, kernel_size=1, bias=False)
        nn.init.zeros_(self.w2.weight)  # do-nothing update at init, so training starts stable

    def perceive(self, x):
        """(B, C, H, W) -> (B, 3C, H, W), interleaved per channel: [c0, c0_dx, c0_dy, c1, ...]."""
        return F.conv2d(x, self.kernels, padding=1, groups=self.channels)

    def forward(self, x, fire_rate=None):
        """One CA step."""
        fire_rate = self.fire_rate if fire_rate is None else fire_rate
        pre_life = alive_mask(x)
        dx = self.w2(F.relu(self.w1(self.perceive(x))))
        # Asynchronous update: each cell fires with probability fire_rate.
        fire = (torch.rand(x.shape[0], 1, x.shape[2], x.shape[3], device=x.device) <= fire_rate).float()
        x = x + dx * fire
        # A cell lives only if it was alive before and after the update. Dead cells are zeroed.
        life = (pre_life & alive_mask(x)).float()
        return x * life

    def grow(self, x, steps):
        for _ in range(steps):
            x = self(x)
        return x


def make_seed(n=1, grid=GRID, channels=CHANNELS, device="cpu"):
    """Empty grid with one live cell in the center: alpha and hidden channels = 1, RGB = 0."""
    x = torch.zeros(n, channels, grid, grid, device=device)
    x[:, 3:, grid // 2, grid // 2] = 1.0
    return x


def load_target(path, size=44, grid=GRID):
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


def to_rgb(x):
    """(B, C, H, W) or (C, H, W) state -> RGB on a white background, as (..., H, W, 3) numpy in [0, 1]."""
    x = x.detach().cpu()
    rgb, a = x[..., :3, :, :], x[..., 3:4, :, :].clamp(0, 1)
    out = (1.0 - a + rgb).clamp(0, 1)
    return out.movedim(-3, -1).numpy()


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


def export_npz(model, path):
    """Save the trained weights as plain arrays for the numpy runtime (space/nca_numpy.py)."""
    sd = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
    np.savez(
        path,
        w1=sd["w1.weight"][:, :, 0, 0].astype(np.float32),  # (HIDDEN, 3C)
        b1=sd["w1.bias"].astype(np.float32),                # (HIDDEN,)
        w2=sd["w2.weight"][:, :, 0, 0].astype(np.float32),  # (C, HIDDEN)
        fire_rate=np.float32(model.fire_rate),
    )
