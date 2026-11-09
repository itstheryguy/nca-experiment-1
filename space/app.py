"""Neural Cellular Automata: the Hugging Face Space.

Runs in the visitor's browser with stlite (Streamlit on WebAssembly), so the Space needs no server. Pyodide has
no PyTorch, so the CA runs on a numpy port of the model (nca_numpy.py) with weights exported from training. Each
model's .npz carries its own settings (channels, wrap, alive mask, render mode).
The same file also runs as a normal Streamlit app: `streamlit run space/app.py`.
"""

import base64
import math
import struct
import time
import zlib
from pathlib import Path

import numpy as np
import streamlit as st

import chaos_numpy as cn
import nca_numpy as ca
import network_numpy as nw

HERE = Path(__file__).parent
FRAME_SECONDS = 0.1  # fragment timer while playing
REPO = "https://github.com/itstheryguy/nca-experiment-1"
GROWING = "https://distill.pub/2020/growing-ca/"
TEXTURES = "https://distill.pub/selforg/2021/textures/"
# Shown in this order when the file exists. index.html must list each file too, so stlite loads it.
MODELS = {
    "Mountain": ("models/mountain.npz",
                 "Grows the 🏔️ emoji from a single live cell, holds it steady, and regrows damage."),
    "Topographic texture": ("models/texture.npz",
                            "Every cell is alive and the edges wrap around. Instead of one picture, the cells learn "
                            "the *style* of a contour map, so no two runs draw the same terrain."),
    "Mountain, trained against chaos": ("models/mountain_chaos.npz",
                                        "The mountain model fine-tuned while the chaos agent attacks it. "
                                        "Try it with the chaos agent on, next to the plain mountain."),
    "Texture, trained against chaos": ("models/texture_chaos.npz",
                                       "The texture model fine-tuned while the chaos agent attacks it."),
    "Network": ("models/network.npz",
                "Connect the hub (gold) to every target (teal) with as little network as possible, like the slime "
                "mold that redrew the Tokyo rail map. Cells only see their neighbors, so they have to signal each "
                "other where the hub and targets are."),
}
VIEWS = ["Color", "Hidden channel", "All channels"]

st.set_page_config(page_title="Neural Cellular Automata", page_icon="🏔️", layout="centered")


@st.cache_resource
def weights(path):
    return ca.load_weights(HERE / path)


@st.cache_resource
def presets():
    return cn.load_presets(HERE / "chaos_presets.json")


def clear_chaos():
    st.session_state.chaos = np.zeros_like(st.session_state.x[0], dtype=np.int8)
    st.session_state.sparked = False
    st.session_state.burned_out = None


def reset(new_layout=True):
    """Start over. Network models get a new random hub and targets, unless new_layout is False."""
    ss = st.session_state
    cfg = weights(MODELS[ss.model][0])["cfg"]
    if cfg["render"] == "network":
        if new_layout or "layout" not in ss:
            ss.layout = nw.make_layout(ss.rng, cfg["grid"])
        ss.inp = nw.inputs(ss.layout, cfg["grid"])
        ss.x = nw.seed(cfg, ss.layout)
    else:
        ss.x = ca.make_seed(cfg)
    st.session_state.steps = 0
    st.session_state.ms = None
    clear_chaos()


def chaos_off():
    """Turning chaos off returns its cells to the NCA as empty cells."""
    if not st.session_state.chaos_on:
        x = st.session_state.x.copy()
        x[:, st.session_state.chaos > 0] = 0
        st.session_state.x = x
        clear_chaos()


available = [name for name, (path, _) in MODELS.items() if (HERE / path).exists()]
if "model" not in st.session_state:
    st.session_state.model = available[0]
    st.session_state.rng = np.random.default_rng()
    st.session_state.playing = True
    reset()

st.title("🏔️ Neural Cellular Automata")
st.markdown(
    "Every pixel runs the same tiny neural network and sees only its 8 neighbors. Nothing stores the picture: it "
    "exists only as the cells' agreement with each other, which is why it can repair itself. "
    f"PyTorch reimplementations of [Growing Neural Cellular Automata]({GROWING}) and "
    f"[Self-Organising Textures]({TEXTURES}), running in your browser."
)

model = st.selectbox("Model", available, key="model", on_change=reset)
wts = weights(MODELS[model][0])
cfg = wts["cfg"]
rng = st.session_state.rng
grid = cfg["grid"]
is_net = cfg["render"] == "network"
n_visible = 4 if cfg["render"] == "rgba" else 3  # network: density, hub, targets
st.caption(MODELS[model][1])

# Controls. Any widget change reruns the whole script, which also re-registers the fragment timer below.
c1, c2, c3, c4 = st.columns(4)
c1.button("Reset", on_click=reset, kwargs={"new_layout": False}, width="stretch", icon=":material/restart_alt:",
          help="Start over" + (" on the same layout" if is_net else ""))
st.session_state.playing = c2.toggle("Play", value=st.session_state.playing)
if c3.button("Damage", width="stretch", icon=":material/bolt:", help="Erase a random circle"):
    cy, cx = rng.integers(grid // 4, 3 * grid // 4, size=2)
    st.session_state.x = ca.erase(st.session_state.x, cy, cx, radius=rng.integers(6, 14))
if is_net:
    c4.button("New layout", on_click=reset, width="stretch", icon=":material/shuffle:",
              help="A new random hub and targets")
elif c4.button("Cut in half", width="stretch", icon=":material/content_cut:"):
    st.session_state.x = st.session_state.x.copy()
    st.session_state.x[:, :, grid // 2:] = 0
if is_net:  # damage never removes the hub or targets
    st.session_state.x = nw.clamp(st.session_state.x, st.session_state.inp)

v1, v2 = st.columns([2, 1])
view = v1.selectbox("View", VIEWS)
channel = n_visible
if view == "Hidden channel":
    channel = v2.slider("Channel", n_visible, cfg["channels"] - 1, n_visible,
                        help="The hidden channels are the state the cells use to coordinate.")
speed = st.slider("Steps per frame", 1, 8, 2)

with st.container(border=True):
    chaos_on = st.toggle("Chaos agent", key="chaos_on", on_change=chaos_off, disabled=is_net,
                         help="Not available for the network model yet." if is_net else
                         "A stochastic process that spreads through the grid and fights the NCA for it.")
    chaos_on = chaos_on and not is_net
    if chaos_on:
        names = list(presets())
        k1, k2 = st.columns(2)
        preset = k1.selectbox("Preset", names, format_func=lambda n: presets()[n]["label"])
        start = k2.number_input("Chaos starts at step", 0, 5000, 150 if cfg["alive_mask"] else 100, step=10,
                                help="0 starts it together with the NCA. The default lets the NCA form first.")
        k3, k4 = st.columns(2)
        every = k3.slider("NCA steps per chaos tick", 1, 5, int(presets()[preset]["every"]))
        thaw = k4.slider("Thaw rate", 0.0, 1.0, float(presets()[preset]["thaw"]), 0.05,
                         help="Chance per tick that a frozen (blue) cell thaws back to an empty NCA cell. "
                              "0 means frozen cells are permanent, so every run eventually ends.")
        P = presets()[preset] | {"thaw": thaw}
        if st.button("Spark", icon=":material/local_fire_department:", help="Add one agent at a random cell"):
            st.session_state.chaos = cn.spark(st.session_state.chaos, rng.random)
            st.session_state.sparked, st.session_state.burned_out = True, None


def png_data_uri(img):
    """Encode an (H, W) or (H, W, 3) uint8 array as a PNG data URI, without Pillow.

    The frame goes inline in an <img>, not through st.image: st.image stores each frame as a media file that the
    page fetches separately, and at animation speed the file is replaced before the fetch arrives (404).
    """
    h, w = img.shape[:2]
    rows = np.concatenate([np.zeros((h, 1), np.uint8), img.reshape(h, -1)], axis=1)  # filter byte 0 per row

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data))

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2 if img.ndim == 3 else 0, 0, 0, 0)
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(rows.tobytes(), 1)) + chunk(b"IEND", b"")
    return "data:image/png;base64," + base64.b64encode(png).decode()


def render(x):
    if view == "Color":
        return cn.overlay(ca.to_rgb(x, cfg), st.session_state.chaos)
    if view == "Hidden channel":
        return ca.to_heatmap(x, channel)
    c = x.shape[0]
    tiles = [np.pad(ca.to_heatmap(x, i), 1, constant_values=255) for i in range(c)]  # visible, then hidden
    tiles += [np.full_like(tiles[0], 255)] * (-c % 4)
    return np.vstack([np.hstack(tiles[r * 4:(r + 1) * 4]) for r in range(math.ceil(c / 4))])


@st.fragment(run_every=FRAME_SECONDS if st.session_state.playing else None)
def board():
    ss = st.session_state
    if ss.playing:
        t = time.perf_counter()
        x, c = ss.x, ss.chaos
        for _ in range(speed):
            x = ca.step(x, wts, rng)
            if is_net:
                x = nw.clamp(x, ss.inp)
            ss.steps += 1
            if chaos_on:
                x = cn.clamp(x, c, cfg)
                if ss.steps >= start:
                    if not ss.sparked:
                        c, ss.sparked = cn.spark(c, rng.random), True
                    if (ss.steps - start) % every == 0:
                        c, x = cn.tick(c, x, cfg, P, rng.random)
                    if ss.burned_out is None and not (c == cn.RED).any():
                        ss.burned_out = ss.steps
        ss.x, ss.chaos = x, c
        ss.ms = (time.perf_counter() - t) * 1000 / speed
    # Upscaled by the browser with pixelated sampling, so cells stay crisp and the payload stays ~10 KB.
    st.html(f'<img src="{png_data_uri(render(st.session_state.x))}" alt="NCA grid, {view.lower()}" '
            'style="display:block;margin:auto;width:100%;max-width:512px;image-rendering:pixelated">')
    ms = st.session_state.get("ms")
    line = f"Step {st.session_state.steps}" + (f" · {ms:.0f} ms per step" if ms else "")
    if is_net:
        k = len(st.session_state.layout[1])
        cells = int((nw.density(st.session_state.x) > 0.5).sum())
        line += (f" · connected {round(nw.connected(st.session_state.x) * k)}/{k} targets"
                 f" · network {cells} cells (spanning-tree baseline {nw.mst_cells(st.session_state.layout)})")
    if chaos_on:
        f = cn.counts(st.session_state.chaos)
        line += f" · NCA {f['nca']:.0%} · red {f['red']:.0%} · black {f['black']:.0%} · blue {f['blue']:.0%}"
        if st.session_state.burned_out is not None:
            line += f" · chaos burned out at step {st.session_state.burned_out}"
        elif not st.session_state.sparked:
            line += f" · chaos starts at step {start}"
    st.caption(line)


board()

with st.expander("How it works"):
    st.markdown(f"""
The grid is {grid}×{grid} cells, each holding {cfg["channels"]} numbers: {n_visible} visible color channels plus
{cfg["channels"] - n_visible} hidden "chemical" channels. Each step, every cell:

1. **Perceives** its 3×3 neighborhood with fixed filters: its own state, and horizontal and vertical gradients
   (Sobel), giving {3 * cfg["channels"]} numbers.
2. **Updates** by passing those through a small network shared by all cells: {3 * cfg["channels"]} →
   {cfg["hidden"]} (ReLU) → {cfg["channels"]}, added to its state. Only a random half of cells update each step,
   so the cells can't rely on a global clock.
3. **Lives or dies** (mountain only): a cell with no neighbor above 0.1 opacity is cleared to empty. The texture
   has no empty space: every cell is alive, and the grid wraps around at the edges.

The mountain is trained to match the emoji pixel by pixel. The texture is trained to match the *style* of a
contour map: the feature statistics a pretrained image network (VGG16) sees, not the exact pixels. Both train on a
pool of past states, some damaged, so the rule learns to stay stable and repair, not only grow once.

Code: [{REPO.removeprefix("https://")}]({REPO}).
""")

with st.expander("The chaos agent"):
    st.markdown("""
Cells are either **NCA**, 🟥 **active agents**, ⬛ **dead agents**, or 🟦 **frozen**. Each chaos tick:

1. Frozen cells thaw back to empty NCA cells at the thaw rate (0 by default: frozen is permanent).
2. Dead agents resolve with probability 0.1: half become empty NCA cells, half freeze.
3. The NCA reclaims chaos cells that touch a living NCA cell: active agents with probability 0.2, dead ones 0.8.
4. Every active agent rolls 0-99 on its preset's table: die, spread to the nearest 1-40 cells (always succeeding on
   NCA cells, mountain or empty alike; dead cells resist 20% of the time), die after spreading, spawn new
   agents anywhere on the grid, or sabotage: die and take each neighboring agent with it half the time.

The NCA sees chaos cells as fixed colors through its normal 3×3 view, but can't change them. With permanent freezing,
every run eventually ends: chaos burns out, and what's left is NCA and frozen cells.
""")

st.caption("Emoji: Noto Emoji 🏔️ by Google, Apache License 2.0. Contour map: generated (tools/make_topo_texture.py).")
