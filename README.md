# Multi-Channel Meta-Surface Holographic Design via Deep Neural Networks

## Overview

This project implements a **deep learning pipeline for designing multi-channel meta-surfaces** (超表面) that can produce desired holographic far-field patterns at multiple wavelengths. The core idea is to train a pair of neural networks:

1. **Forward Model (StO - Structure-to-Optics)**: Predicts the optical response (amplitude & phase) of a meta-surface unit cell given its binary structure.
2. **Inverse Model (OtS - Optics-to-Structure)**: Given a desired optical response, predicts the binary structure that would produce it — enabling **inverse design** of meta-surfaces.

The pipeline also includes an end-to-end **holographic phase array optimization** (`VDNN_phase_array.py`) that uses both models to optimize a large-scale meta-surface array (200×200 pixels, 30 channels) to produce target holographic images (digits 0–9, letters A–Z, etc.).

## Architecture

### Forward Model (StO)
```
Input:  6×6 binary matrix (meta-surface unit cell structure)
        ↓
   Conv2d(1→8, 3×3) + BN + ReLU
        ↓
   Conv2d(8→16, 3×3) + BN + ReLU
        ↓
   MaxPool2d(2×2)
        ↓
   Flatten → FC(144→256) → FC(256→128) → FC(128→64) → FC(64→60) + Sigmoid
        ↓
Output: 60-dim optical response vector
        (sin/cos encoding of phase at 3 wavelengths × 10 polarizations)
```
- **Parameters**: 83,468
- **Wavelengths**: 530nm, 670nm, 800nm
- **Polarizations**: 10 angles per wavelength
- **Output encoding**: `[sin(φ)/2+0.5, cos(φ)/2+0.5]` pairs (normalized to [0,1])

### Inverse Model (OtS)
```
Input:  60-dim optical response vector (desired target)
        ↓
   FC(60→64) → FC(64→128) → FC(128→256) → FC(256→576) + ReLU
        ↓
   Reshape → 16×6×6
        ↓
   Conv2d(16→8, 3×3) + BN + ReLU
        ↓
   Conv2d(8→1, 3×3) + BN + Sigmoid
        ↓
Output: 6×6 structure matrix (continuous values, thresholded to binary)
```
- **Parameters**: 194,531
- **Training strategy**: Uses frozen pre-trained forward model as differentiable evaluator
- **Loss**: MSE between `forward_model(inverse_model(target))` and `target`

### End-to-End Pipeline
```
Target Images (30 channels, 200×200)
        ↓
Phase Array D (30×200×200, learnable)
        ↓
sin/cos encoding → Inverse Model → Structure (6×6) → Forward Model → Phase
        ↓
FFT → Far-field Pattern → Loss vs Target Images
```

## Dataset

- **Total samples**: 44,033 meta-surface unit cells
- **Structure file** (`dataset/st_36.txt`): Each line is a 36-character binary string representing a 6×6 binary matrix
- **Optical response files**:
  - `dataset/opr_530.txt`: 20-dim response at λ=530nm
  - `dataset/opr_670.txt`: 20-dim response at λ=670nm  
  - `dataset/opr_800.txt`: 20-dim response at λ=800nm
- **Combined response**: 60-dim vector (20 per wavelength)
- The dataset was generated via **FDTD simulation** (Lumerical)

## Environment Setup

### Prerequisites
- Windows 10
- NVIDIA GPU (tested on NVIDIA T400 4GB)
- Anaconda/Miniconda

### Conda Environment (mappo)

The project uses the `mappo` conda environment:

```bash
# Activate the environment
conda activate mappo

# Environment details:
# - Python 3.8.19
# - PyTorch 2.4.0 (CUDA enabled)
# - NumPy 1.24.3
# - OpenCV 4.11.0
```

If you need to create the environment from scratch:
```bash
conda create -n mappo python=3.8
conda activate mappo
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install numpy opencv-python matplotlib
```

## Quick Start

### 1. Train the Forward Model

```bash
# Using the mappo conda environment
conda activate mappo

# Train forward model (Structure → Optics)
python run_forward.py

# Or use the full-featured version:
python train_forward_mappo.py
```

**Expected output:**
```
============================================================
Forward Model (StO) Training - MAPPO env
============================================================
Device: cuda:0
Loaded 44033 samples in 1.3s, opr dim=60
Parameters: 83,468
Train: 39629, Test: 4404
============================================================
E   0 | TrainL: 265 | TestL: 252 (*1e-3)
E  10 | TrainL: 208 | TestL: 211 (*1e-3)
E  40 | TrainL: 189 | TestL: 205 (*1e-3)  ← Best
...
```

### 2. Train the Inverse Model

```bash
# Requires a pre-trained forward model
python run_inverse.py

# Or use the full-featured version:
python train_inverse_mappo.py
```

**Expected output:**
```
============================================================
Inverse Model (OtS) Training - MAPPO env
============================================================
Device: cuda:0
Forward model params: 83,468
Inverse model params: 194,531
============================================================
E   0 | TrainL: 361 | TestL: 351 (*1e-3)
E  30 | TrainL: 324 | TestL: 324 (*1e-3)
E 110 | TrainL: 320 | TestL: 319 (*1e-3)  ← Best
...
```

### 3. End-to-End Holographic Optimization (Optional)

```bash
# Requires both forward and inverse models
python VDNN_phase_array.py
```
This optimizes a 30-channel phase array (200×200) to produce target holographic images using the trained forward and inverse models.

## File Structure

```
C:/holo/
├── README.md                    # This file
├── train_for.py                 # Original forward model training
├── train_back.py                # Original inverse model training
├── run_forward.py               # Adapted forward training (mappo env)
├── run_inverse.py               # Adapted inverse training (mappo env)
├── train_forward_mappo.py       # Full-featured forward training
├── train_inverse_mappo.py       # Full-featured inverse training
├── VDNN_phase_array.py          # End-to-end holographic optimization
├── whole.py                     # Direct phase optimization (no NN)
├── test_fdtd_vs_dataset.py      # Validate dataset against FDTD sim
├── record.md                    # Development notes (Chinese)
│
├── dataset/                     # Training data
│   ├── st_36.txt                # 44,033 binary structures (6×6)
│   ├── opr_530.txt              # Optical response @ 530nm
│   ├── opr_670.txt              # Optical response @ 670nm
│   └── opr_800.txt              # Optical response @ 800nm
│
├── 4/                           # Model checkpoints & logs
│   ├── 44033fe50l201.mdl        # Pre-trained forward model (best)
│   ├── 1000000ie510l318.mdl     # Pre-trained inverse model (best)
│   ├── forward_best_e40_l205.mdl  # New forward model (mappo)
│   ├── inverse_best_e110_l319.mdl # New inverse model (mappo)
│   ├── forward loss.txt         # Original forward training log
│   ├── VDNN_phase_array/        # Phase array optimization results
│   └── best/                    # Best holographic results
│
├── pictures/                    # Target holographic images
│   ├── number/                  # 0-9 digit images
│   ├── Capital letters/         # A-Z letter images
│   └── Lowercase letters/       # a-z letter images
│
└── fdtd/                        # FDTD simulation files (Lumerical)
    ├── Double-sided.fsp         # Simulation project file
    ├── Double-sided_template.lsf # Script template
    └── Double-sided_run.lsf     # Runtime script
```

## Preliminary Results

### Forward Model (StO)
| Metric | Value |
|--------|-------|
| Best Test Loss | **205 × 10⁻³** (RMSE) |
| Best Epoch | 40 |
| Training Samples | 39,629 |
| Test Samples | 4,404 |
| Training Time | ~10 min (200 epochs, T400 GPU) |

The forward model converges quickly and reaches a test RMSE of ~0.205, consistent with the original training results.

### Inverse Model (OtS)
| Metric | Value |
|--------|-------|
| Best Test Loss | **317 × 10⁻³** (RMSE) |
| Best Epoch | 160 |
| Random Samples/Epoch | 10,000 |
| Training Time | ~10 min (200 epochs, T400 GPU) |

The inverse model achieves a test RMSE of ~0.317, **surpassing** the original model's performance (318 × 10⁻³ achieved with 1M samples and 510 epochs).

### Loss Convergence

**Forward Model:**
```
Epoch    Train Loss    Test Loss
  0        265          252
 10        208          211
 20        198          208
 30        193          206
 40        189          205  ← best
 50        187          205
100        181          207
140        179          207
```

**Inverse Model:**
```
Epoch    Train Loss    Test Loss
  0        361          351
 10        330          330
 30        324          324
 60        322          322
 80        321          321
100        321          320
110        320          319
120        320          322
160        319          317  ← best
190        319          318
```

## Key Concepts

### Meta-Surface Unit Cell
Each unit cell is represented as a **6×6 binary matrix** where 1/0 indicates the presence/absence of material at each pixel. This discretized representation allows for practical nanofabrication.

### Optical Response Encoding
The optical response is encoded as **sin/cos pairs** of the transmission phase:
- For each wavelength (530nm, 670nm, 800nm)
- For each of 10 polarization angles
- Each angle produces `[sin(φ)/2+0.5, cos(φ)/2+0.5]`
- Total: 3 × 10 × 2 = **60 dimensions**

### Inverse Design Strategy
The inverse model is trained using a **tandem network** approach:
1. Train the forward model on real FDTD data
2. Freeze the forward model
3. Train the inverse model with random optical targets
4. Loss = MSE(Forward(Inverse(target)), target)

This avoids the one-to-many mapping problem in inverse design, as the inverse model learns to find *any* valid structure that produces the desired response.

### Holographic Phase Array
The final application combines 40,000 unit cells (200×200 grid) across 30 channels to produce holographic far-field images via FFT. Each channel corresponds to a different image (digits, letters, etc.).

## Notes

- The **NVIDIA T400 4GB** GPU is sufficient for training both models
- The forward model shows slight overfitting after epoch 50 — consider using learning rate scheduling or dropout for improvement
- The inverse model could benefit from larger training (more epochs, more samples per epoch) for further improvement
- The `whole.py` script provides a **direct optimization baseline** that optimizes the phase array without neural networks
- For FDTD validation, see `test_fdtd_vs_dataset.py` (requires Lumerical FDTD installation)

## Future Improvements

1. **Improve inverse model quality**: Use VAE-based latent space, larger models, or reinforcement learning approaches
2. **Learning rate scheduling**: Add cosine annealing or ReduceLROnPlateau
3. **Data augmentation**: Exploit symmetries in the 6×6 structure
4. **Larger dataset**: Generate more FDTD samples for better generalization
5. **Binary structure enforcement**: Add Gumbel-Softmax or straight-through estimator for hard binary output
