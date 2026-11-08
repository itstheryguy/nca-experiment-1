"""Growing NCA: the Hugging Face Space.

Runs in the visitor's browser with stlite (Streamlit on WebAssembly), so the Space needs no server. Pyodide has
no PyTorch, so the CA runs on a numpy port of the model (nca_numpy.py) with weights exported from training.
The same file also runs as a normal Streamlit app: `streamlit run space/app.py`.
"""

from pathlib import Path

import numpy as np
import streamlit as st

import nca_numpy as ca

HERE = Path(__file__).parent
CHANNELS, GRID, SCALE = 16, 64, 8
REPO = "https://github.com/itstheryguy/nca-experiment-1"
PAPER = "https://distill.pub/2020/growing-ca/"
VIEWS = ["Color", "Hidden channel", "All 16 channels"]

st.set_page_config(page_title="Growing NCA", page_icon="🏔️", layout="centered")


@st.cache_resource
def weights():
    return ca.load_weights(HERE / "nca_mountain.npz")


def reset():
    st.session_state.x = ca.make_seed(CHANNELS, GRID)
    st.session_state.steps = 0


if "x" not in st.session_state:
    st.session_state.rng = np.random.default_rng()
    reset()
    st.session_state.playing = True

wts = weights()
rng = st.session_state.rng

st.title("🏔️ Growing a mountain from one cell")
st.markdown(
    "Every pixel runs the same tiny neural network (about 8,000 weights), and sees only its 8 neighbors. "
    "From a single live cell they grow the mountain, hold it steady, and regrow it when you damage it. "
    f"A PyTorch reimplementation of [Growing Neural Cellular Automata]({PAPER}), running in your browser."
)

# Controls. Any widget change reruns the whole script, which also re-registers the fragment timer below.
c1, c2, c3, c4 = st.columns(4)
c1.button("Reset", on_click=reset, width="stretch", icon=":material/restart_alt:")
st.session_state.playing = c2.toggle("Play", value=st.session_state.playing)
if c3.button("Damage", width="stretch", icon=":material/bolt:", help="Erase a random circle"):
    cy, cx = rng.integers(GRID // 4, 3 * GRID // 4, size=2)
    st.session_state.x = ca.erase(st.session_state.x, cy, cx, radius=rng.integers(6, 14))
if c4.button("Cut in half", width="stretch", icon=":material/content_cut:"):
    st.session_state.x = st.session_state.x.copy()
    st.session_state.x[:, :, GRID // 2:] = 0

v1, v2 = st.columns([2, 1])
view = v1.selectbox("View", VIEWS)
channel = v2.slider("Channel", 4, 15, 4, disabled=view != "Hidden channel",
                    help="Channels 4-15 are the hidden state the cells use to coordinate.")
speed = st.slider("Steps per frame", 1, 8, 2)


def render(x):
    if view == "Color":
        img = ca.to_rgb(x)
    elif view == "Hidden channel":
        img = ca.to_heatmap(x, channel)
    else:  # 4x4 tiles: RGBA, then the 12 hidden channels
        tiles = [ca.to_heatmap(x, c) for c in range(CHANNELS)]
        tiles = [np.pad(t, 1, constant_values=255) for t in tiles]
        img = np.vstack([np.hstack(tiles[r * 4:(r + 1) * 4]) for r in range(4)])
        return img.repeat(4, 0).repeat(4, 1)
    return img.repeat(SCALE, 0).repeat(SCALE, 1)  # nearest-neighbor upscale, so cells stay crisp


@st.fragment(run_every=0.05 if st.session_state.playing else None)
def board():
    if st.session_state.playing:
        x = st.session_state.x
        for _ in range(speed):
            x = ca.step(x, wts, rng)
        st.session_state.x = x
        st.session_state.steps += speed
    st.image(render(st.session_state.x), width="stretch")
    st.caption(f"Step {st.session_state.steps}")


board()

with st.expander("How it works"):
    st.markdown(f"""
The grid is 64×64 cells, each holding 16 numbers: color and opacity (RGBA) plus 12 hidden "chemical" channels.
Each step, every cell:

1. **Perceives** its 3×3 neighborhood with fixed filters: its own state, and horizontal and vertical gradients
   (Sobel), giving 48 numbers.
2. **Updates** by passing those through a small network shared by all cells: 48 → 128 (ReLU) → 16, added to its
   state. Only a random half of cells update each step, so the cells can't rely on a global clock.
3. **Lives or dies**: a cell with no neighbor above 0.1 opacity is cleared to empty.

Training starts from the seed, runs 64–96 steps, and compares the RGBA channels to the emoji. A pool of past
states is fed back in, and some are damaged, so the rule learns to keep the mountain stable and regrow missing
parts, not only grow it once.
"""
    + (f"\nCode: [{REPO.removeprefix('https://')}]({REPO})." if REPO else ""))

st.caption("Emoji: Noto Emoji 🏔️ by Google, Apache License 2.0.")
