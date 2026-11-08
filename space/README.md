---
title: Growing NCA
emoji: 🏔️
colorFrom: blue
colorTo: green
sdk: static
pinned: false
license: mit
short_description: A neural cellular automaton that grows and regrows a mountain.
---

# Growing a mountain with a Neural Cellular Automaton

Every pixel runs the same small neural network (8,320 weights) and sees only its 3×3 neighborhood. Starting from a
single live cell, the cells grow the 🏔️ emoji, hold it stable, and regrow it after damage. A PyTorch
reimplementation of [Growing Neural Cellular Automata](https://distill.pub/2020/growing-ca/) (Mordvintsev et al.,
Distill 2020), trained on an Apple M1 Pro.

The app runs entirely in your browser ([stlite](https://github.com/whitphx/stlite)). Training is in PyTorch; the
weights are exported to `nca_mountain.npz` and the Space steps the automaton with a numpy port of the model,
since Pyodide has no PyTorch. A parity test checks that the two implementations agree.

Emoji: [Noto Emoji](https://github.com/googlefonts/noto-emoji) by Google, Apache License 2.0.
