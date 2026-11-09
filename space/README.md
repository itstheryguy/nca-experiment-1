---
title: Neural Cellular Automata
emoji: 🏔️
colorFrom: blue
colorTo: green
sdk: static
pinned: false
license: mit
short_description: Self-repairing NCAs versus a stochastic chaos agent
---

# Neural Cellular Automata vs. a chaos agent

Every pixel runs the same small neural network and sees only its 3×3 neighborhood. Nothing stores the picture:
it exists only as the cells' agreement with each other, which is why it can repair itself.

Four models, picked from a dropdown:

- **Mountain:** grows the 🏔️ emoji from one live cell and regrows damage
  ([Growing NCA](https://distill.pub/2020/growing-ca/), Mordvintsev et al., 2020).
- **Topographic texture:** every cell is alive and the grid wraps around; trained on the *style* of a contour map
  with a VGG16 feature loss ([Self-Organising Textures](https://distill.pub/selforg/2021/textures/), Niklasson et al., 2021).
- **Mountain / texture, trained against chaos:** the same models fine-tuned while a chaos agent attacks them.
- **Network:** connect a randomly placed hub to 3-6 targets with as little network as possible, after the slime mold
  *Physarum polycephalum* redrawing the Tokyo rail network. Cells signal through their hidden channels where the hub
  and targets are, explore, then prune to a tree about the size of a minimum spanning tree.

The **chaos agent** is a stochastic cellular process: active agents (red) roll a dice table each tick to spread,
die (black), sabotage their neighbors, or spawn elsewhere; dead agents either return to the NCA or freeze (blue).
The NCA reclaims chaos cells next to its living cells. Presets range from smolder (usually fizzles) to wildfire
(usually wins). The "All channels" view shows the NCA's hidden state reacting to the attack.

Training is in PyTorch (locally on an M1 Pro and on a Colab A100). The app runs entirely in your browser with
[stlite](https://github.com/whitphx/stlite): the weights are exported to `.npz` and stepped by a numpy port of the
model, checked against PyTorch by parity tests.

Code: [github.com/itstheryguy/nca-experiment-1](https://github.com/itstheryguy/nca-experiment-1).
Emoji: [Noto Emoji](https://github.com/googlefonts/noto-emoji) by Google, Apache License 2.0.
