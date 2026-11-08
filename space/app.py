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

import nca_numpy as ca

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
}
VIEWS = ["Color", "Hidden channel", "All channels"]

st.set_page_config(page_title="Neural Cellular Automata", page_icon="🏔️", layout="centered")


@st.cache_resource
def weights(path):
    return ca.load_weights(HERE / path)


def reset():
    st.session_state.x = ca.make_seed(weights(MODELS[st.session_state.model][0])["cfg"])
    st.session_state.steps = 0
    st.session_state.ms = None


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
n_visible = 4 if cfg["render"] == "rgba" else 3
st.caption(MODELS[model][1])

# Controls. Any widget change reruns the whole script, which also re-registers the fragment timer below.
c1, c2, c3, c4 = st.columns(4)
c1.button("Reset", on_click=reset, width="stretch", icon=":material/restart_alt:")
st.session_state.playing = c2.toggle("Play", value=st.session_state.playing)
if c3.button("Damage", width="stretch", icon=":material/bolt:", help="Erase a random circle"):
    cy, cx = rng.integers(grid // 4, 3 * grid // 4, size=2)
    st.session_state.x = ca.erase(st.session_state.x, cy, cx, radius=rng.integers(6, 14))
if c4.button("Cut in half", width="stretch", icon=":material/content_cut:"):
    st.session_state.x = st.session_state.x.copy()
    st.session_state.x[:, :, grid // 2:] = 0

v1, v2 = st.columns([2, 1])
view = v1.selectbox("View", VIEWS)
channel = n_visible
if view == "Hidden channel":
    channel = v2.slider("Channel", n_visible, cfg["channels"] - 1, n_visible,
                        help="The hidden channels are the state the cells use to coordinate.")
speed = st.slider("Steps per frame", 1, 8, 2)


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
        return ca.to_rgb(x, cfg)
    if view == "Hidden channel":
        return ca.to_heatmap(x, channel)
    c = x.shape[0]
    tiles = [np.pad(ca.to_heatmap(x, i), 1, constant_values=255) for i in range(c)]  # visible, then hidden
    tiles += [np.full_like(tiles[0], 255)] * (-c % 4)
    return np.vstack([np.hstack(tiles[r * 4:(r + 1) * 4]) for r in range(math.ceil(c / 4))])


@st.fragment(run_every=FRAME_SECONDS if st.session_state.playing else None)
def board():
    if st.session_state.playing:
        t = time.perf_counter()
        x = st.session_state.x
        for _ in range(speed):
            x = ca.step(x, wts, rng)
        st.session_state.x = x
        st.session_state.steps += speed
        st.session_state.ms = (time.perf_counter() - t) * 1000 / speed
    # Upscaled by the browser with pixelated sampling, so cells stay crisp and the payload stays ~10 KB.
    st.html(f'<img src="{png_data_uri(render(st.session_state.x))}" alt="NCA grid, {view.lower()}" '
            'style="display:block;margin:auto;width:100%;max-width:512px;image-rendering:pixelated">')
    ms = st.session_state.get("ms")
    st.caption(f"Step {st.session_state.steps}" + (f" · {ms:.0f} ms per step" if ms else ""))


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

st.caption("Emoji: Noto Emoji 🏔️ by Google, Apache License 2.0. Contour map: generated (tools/make_topo_texture.py).")
