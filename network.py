"""The network task in PyTorch: batched layouts, input clamping, and the route-cost loss.

See space/network_numpy.py for the channel layout. The loss asks for every target to have a cheap route to the hub
and for the network to be small:

    cost of entering a cell = 1 inside the network, `empty_cost` outside (linear in the density in between)
    route  = sum over targets of the cheapest path cost from the hub (8-connected moves)
    loss   = (route + material_weight * network cells) / (grid * empty_cost)   (about 1-3 for an empty grid)

The cheapest path costs come from repeated relaxation (D = min(D, neighbor D + cost)), unrolled in PyTorch, so the
gradient reaches the cells along each target's current best route. Material cost is what makes shared trunks pay
off, so the best answer is a Steiner-like tree, not a star of separate spokes.
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent / "space"))
import network_numpy as nn_np  # noqa: E402

NET, HUB, TARGET = nn_np.NET, nn_np.HUB, nn_np.TARGET
BIG = 1e4


def layouts(n, grid, rng, device, **kw):
    """(n, 2, grid, grid) input masks (hub, targets) for n random layouts."""
    return torch.as_tensor(np.stack([nn_np.inputs(nn_np.make_layout(rng, grid, **kw), grid) for _ in range(n)]),
                           device=device)


def clamp(x, inp):
    """Write the input channels back. Differentiable w.r.t. the other channels."""
    return torch.cat([x[:, :HUB], inp, x[:, TARGET + 1:]], dim=1)


def seed(n_channels, inp):
    x = torch.zeros(inp.shape[0], n_channels, inp.shape[2], inp.shape[3], device=inp.device)
    return clamp(x, inp)


def density(x, leak=0.0):
    """clip(x0, 0, 1) for display and material; with leak > 0 (route costs), a small slope below 0.

    Two gradient traps to avoid:
    - Past 1: no gradient on purpose. An identity gradient there kept pushing full cells higher (more density
      always looks cheaper to route through), and the state blew up.
    - At and below 0: every cell starts at exactly 0, and torch's clamp passes no gradient at its boundary, so the
      network channel never got a signal. With leak, the gradient is 1 at 0 and `leak` below it.
    """
    n = x[:, NET]
    if not leak:
        return n.clamp(0, 1)
    return torch.where(n < 0, leak * n, n.clamp(max=1))


def route_costs(x, empty_cost=10.0, iters=None):
    """(B, H, W) cheapest path cost from the hub to every cell."""
    b, _, h, w = x.shape
    hub, tgt = x[:, HUB] > 0.5, x[:, TARGET] > 0.5
    cost = 1.0 + (empty_cost - 1.0) * (1.0 - density(x, leak=0.05))
    cost = torch.where(hub | tgt, torch.ones_like(cost), cost)
    d = torch.where(hub, torch.zeros_like(cost), torch.full_like(cost, BIG))
    for _ in range(iters or 2 * max(h, w)):
        nb = -F.max_pool2d(F.pad(-d[:, None], (1, 1, 1, 1), value=-BIG), 3, stride=1)[:, 0]
        d = torch.minimum(d, nb + cost)
    return d


def loss(x, empty_cost=10.0, material_weight=2.0):
    """(B,) loss: route cost to every target plus material, per grid width."""
    grid = x.shape[-1]
    tgt = x[:, TARGET] > 0.5
    route = (route_costs(x, empty_cost) * tgt).sum(dim=(1, 2))
    inputs_mask = (x[:, HUB] > 0.5) | tgt
    material = (density(x) * ~inputs_mask).sum(dim=(1, 2))
    return (route + material_weight * material) / (grid * empty_cost)


@torch.no_grad()
def connected(x, thresh=0.5):
    """(B,) fraction of targets connected to the hub through network cells."""
    passable = (x[:, NET] > thresh) | (x[:, HUB] > 0.5) | (x[:, TARGET] > 0.5)
    reach = x[:, HUB] > 0.5
    for _ in range(x.shape[-1] * x.shape[-2]):
        grown = (F.max_pool2d(reach[:, None].float(), 3, stride=1, padding=1)[:, 0] > 0) & passable
        if torch.equal(grown, reach):
            break
        reach = grown
    tgt = x[:, TARGET] > 0.5
    return (reach & tgt).flatten(1).sum(1) / tgt.flatten(1).sum(1).clamp(min=1)


def damage(x, inp, n_circles=1):
    """Erase random circles (everything but the inputs), in place, then restore the inputs."""
    from nca import damage as erase

    erase(x, n_circles)
    x[:, HUB:TARGET + 1] = inp
    return x
