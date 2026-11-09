# Neural Cellular Automata: mountains, maps, and chaos

PyTorch Neural Cellular Automata deployed as a static Hugging Face Space that runs in the browser with stlite.

<img src="assets/mountain.gif" width="256" alt="The mountain growing from one cell, cut in half at step 300, and regrowing">

| Model | Status | Based on |
|---|---|---|
| `mountain`: grows the 🏔️ emoji from one cell and regrows damage | 8,000 iterations, 50 min on an M1 Pro; loss 0.0034, stable through 2,000+ steps | [Growing NCA](https://distill.pub/2020/growing-ca/) |
| `texture`: grows a topographic contour-map texture on a wrapping grid | 6,000 iterations, ~10 min on an A100; learns the contour lines, not yet the color bands | [Self-Organising Textures](https://distill.pub/selforg/2021/textures/) |
| `mountain_chaos`: `mountain` fine-tuned against the chaos agent | v1.1, 4,000 iterations on an A100. Better than the base model against brushfire (survives 51% vs 45%, median integrity 0.80 vs 0.37), even against smolder and wildfire, slightly hazier without chaos (error 0.007 vs 0.003). v1 (counter-seeds on in training, lr 1e-3) was worse on every measure | |
| `network`: connect a random hub to 3-6 targets with as little network as possible (after Physarum's Tokyo rail network) | v1.1, 8,000 iterations, ~21 min on an A100. On 256 fresh layouts: all targets connected in 69% (86% of targets), network 1.03x the size of a spanning tree, median 95 steps to connect, 89% reconnect after damage. v1 (gaps cost 10, not 30): 27% fully connected, 1.53x | |
| `texture_chaos`: `texture` fine-tuned against the chaos agent | 3,000 iterations on an A100; repairs around chaos cells. Chaos outcomes on the texture don't depend on the model (every cell defends), so it can't change who wins | |

## Layout

| Path | What |
|---|---|
| `nca.py` | Model (`CAModel`, configured by `CAConfig`), seeds, targets, damage, weight export |
| `train.py` | Experiment configs, pool training loop, pixel and VGG style losses |
| `train.ipynb` | Training driver: runs locally (MPS) or on Colab (CUDA), with a live flipbook |
| `space/` | The Space: `app.py` (Streamlit), `nca_numpy.py` (numpy port of the model), `index.html` (stlite), `models/*.npz` |
| `tests/test_parity.py` | Checks the numpy port matches PyTorch for both model types |
| `tools/make_topo_texture.py` | Generates the contour-map texture target |
| `peek.py` | Progress and a preview image during a local training run |
| `runs/` | Training loss logs |

## Setup

```sh
pip install torch torchvision numpy matplotlib jupyter   # global
python -m venv --system-site-packages .venv             # sees the global packages
.venv/bin/pip install pillow streamlit
```

## Train

Set `EXPERIMENT` in `train.ipynb` and run all cells.

- **Locally:** pick the `.venv` interpreter. About 50 minutes for `mountain` and 17 for `texture` on an M1 Pro.
- **On Colab** (VS Code Colab extension, or colab.research.google.com): pick a GPU runtime. The notebook clones this
  repo from GitHub `main`, so push first. It needs a GitHub fine-grained token with *Contents: read and write* on
  this repo, from the Colab secret `GH_TOKEN` or pasted when asked. When training ends it pushes the weights, GIF,
  and loss log to a `run/<experiment>-<time>` branch to merge.

Each run writes `models/<name>.pth`, `space/models/<name>.npz`, `assets/<name>.gif`, and `runs/<name>_log.csv`.
To show a new model in the Space, add it to `MODELS` in `space/app.py` and to the file list in `space/index.html`.

## Run the Space locally

```sh
.venv/bin/streamlit run space/app.py                 # as a normal Streamlit app
python -m http.server -d space 8000                  # as the static stlite page, at localhost:8000
```

## Deploy

```sh
hf upload itstheryguy/growing-nca space . --repo-type space --exclude "__pycache__/*"
```

## Notes

- Mountain: 44 px emoji on a 64 px grid (the paper: 40 px on 72 px).
- Texture: 12 channels and 96 hidden units (the paper's size), trained at 64×64 with the VGG loss on a 2× upsample.
- Perception is one grouped `conv2d` (about 13% faster on MPS than three). The numpy runtime only evaluates cells
  that fire and are near a live cell, which is exact and about 7× faster under WebAssembly.
