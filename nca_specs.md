# Project Specification: Neural Cellular Automata (NCA) in PyTorch

## Objective
Build a PyTorch-based Neural Cellular Automaton (NCA) capable of "growing" a 2D mountain emoji from a single seed pixel. The project consists of a training loop (designed to run locally if possible) and a deployment architecture for a Hugging Face Space (static Streamlit).

## 1. Core Architecture (The Model)
Create a PyTorch `nn.Module` (e.g., `CAModel`) that updates a state tensor.
*   **State Tensor Shape:** `(Batch, Channels, Height, Width)` where Channels = 16.
    *   Channels 0-3: RGBA (Visible state).
    *   Channels 4-15: Hidden "chemical" state.
*   **Perception (Neighborhood math):**
    *   Use fixed 3x3 depthwise convolutions to compute the perception vector.
    *   Filters needed: Identity (center pixel), Sobel X (horizontal gradient), and Sobel Y (vertical gradient).
    *   Output of perception will have 48 channels (16 * 3).
*   **Update Rule (The Perceptrons):**
    *   A 1x1 Convolution pipeline acting as a per-pixel MLP.
    *   Architecture: `Conv2d(48, 128)` -> `ReLU` -> `Conv2d(128, 16)`.
    *   **Crucial Initialization:** Initialize the weights and biases of the final `Conv2d(128, 16)` layer to zero so the system starts entirely stable.
*   **Stochasticity & Alive Masking:**
    *   Only apply the update to a random ~50% of cells per step to simulate asynchronous biological updates.
    *   Implement an "Alive Mask": Only allow a cell to update if the alpha channel (channel 3) of that cell or its immediate neighbors is > 0.1. This prevents the empty background from randomly mutating.

## 2. Training Pipeline
*   **Target Image:** Load a 64x64 RGBA mountain emoji. 
*   **Seed:** A 64x64 tensor of zeros, with the exact center pixel set to `1.0` across the first 4 channels (RGBA) or just the black/alpha channels.
*   **The Sample Pool (Critical for Stability):**
    *   Maintain a "pool" of 1024 past grid states.
    *   When creating a batch (e.g., batch size 8), pull 7 states from the pool and initialize 1 state as a fresh seed.
    *   Run the model for a random number of steps (e.g., 64 to 96 steps).
    *   Calculate the loss, update the model, and then put the updated states back into the pool. This forces the model to learn how to *maintain* the mountain, not just grow it once and explode.
*   **Loss Function:** Mean Squared Error (MSE) between the first 4 channels (RGBA) of the output tensor and the target image.
*   **Optimizer:** Adam (`lr=2e-3`, with a learning rate scheduler).

## 3. Visualization (Development)
Write a helper function to render the tensor directly in a notebook cell.
*   Slice the first 3 channels (RGB) or 4 channels (RGBA).
*   Use `matplotlib.pyplot.imshow` combined with `IPython.display.clear_output(wait=True)` to create a flipbook animation running directly in the training loop.

## 4. Streamlit Deployment Architecture
Create an `app.py` script for Hugging Face Spaces.
*   Load the trained `.pth` weights into the `CAModel`.
*   Initialize the 16-channel tensor and store it in `st.session_state`.
*   **UI Elements:**
    *   "Reset" button (clears grid, places seed).
    *   "Play/Pause" toggle to run the forward pass continuously.
    *   A dropdown selector to choose the "View":
        *   Option 1: Standard RGB (Channels 0-2).
        *   Option 2: Hidden State Heatmap (e.g., visualizing Channel 4 in grayscale).
    *   Display the selected slice in real-time using `st.image()`.

## 5. Other Notes
*   I would like to train locally if possible. If not, we can leverage sagemaker on aws. 
*   I have other live spaces running, I would like to match the static deployment being used for those if possible. 
*   I am indifferent how we run this: notebook or py file. 