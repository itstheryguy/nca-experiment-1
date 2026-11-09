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

# Neural Cellular Automata

Every pixel runs the same small neural network and sees only its 3×3 neighborhood. Nothing stores the picture:
it exists only as the cells' agreement with each other, which is why it can repair itself.

Five models, picked from a dropdown:

- **Network:** connect a random hub to 3-6 targets with as little network as possible, after the slime mold
  *Physarum polycephalum* redrawing the Tokyo rail network. Trained with a differentiable shortest-path loss. On
  unseen layouts it connects every target 69% of the time, with a network about the size of a minimum spanning
  tree, and reconnects after damage.
- **Mountain:** grows the 🏔️ emoji from one live cell and regrows damage. A reimplementation of
  [Growing NCA](https://distill.pub/2020/growing-ca/) (Mordvintsev et al., 2020).
- **Topographic texture:** every cell is alive and the grid wraps around; trained on the *style* of a contour map
  with a VGG16 feature loss. A reimplementation of
  [Self-Organising Textures](https://distill.pub/selforg/2021/textures/) (Niklasson et al., 2021).
- **Mountain and texture, trained against chaos:** the same models fine-tuned while the chaos agent attacks them.

The **chaos agent** is a stochastic cellular process designed for this project: active agents (red) roll a dice
table each tick to spread, die (black), sabotage their neighbors, or spawn elsewhere; dead agents either return to
the NCA or freeze (blue). The NCA reclaims chaos cells next to its living cells. The rules were tuned with Monte Carlo
simulations, and presets range from smolder (usually fizzles) to wildfire (usually wins). The "All channels" view
shows the NCA's hidden state reacting to the attack.

Training is in PyTorch, locally on an M1 Pro and on a Colab A100. The app runs entirely in your browser with
[stlite](https://github.com/whitphx/stlite): the weights are exported to `.npz` and stepped by a numpy port of the
model, checked against PyTorch by parity tests.

Code, results, and write-up: [github.com/itstheryguy/nca-experiment-1](https://github.com/itstheryguy/nca-experiment-1).
Emoji: [Noto Emoji](https://github.com/googlefonts/noto-emoji) by Google, Apache License 2.0.
