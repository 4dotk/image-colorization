# Legacy models (pre-rewrite)

These are the original models from this repository, archived for reference.
They are **frozen** at tag [`v0-legacy`](https://github.com/4dotk/image-colorization/releases/tag/v0-legacy)
and are **not used by the new pipeline** or by the LAB comparison study.

The weights are not tracked in git. Download them from the `v0-legacy` release
and verify them against [`SHA256SUMS.txt`](./SHA256SUMS.txt).

## What is here

| File | Model | Input → output | Original training data | Notes |
|---|---|---|---|---|
| `autoencoder.py` | `AutoEncoder` | 1ch → RGB, 150×150 | Kaggle landscape colorization dataset (~7k images) | 4 down / 4 up with skip connections, sigmoid output |
| `imagenette_autoencoder.py` | `ImageNetAutoEncoder` | 1ch → RGB, 224×224 | Imagenette | No skip connections |
| `cnn.py` | `ColorizationNet` | 1ch → RGB, fully convolutional | CIFAR-10 (32×32) | 10 dilated 5×5 convs, sigmoid output |
| `switch_case.py` | UI name → checkpoint name | — | — | Also handles `hsv`, which the old app never exposed |
| `app_legacy.py` | Old Streamlit app | — | — | Contains the known color-space conversion bug described below |

## Weights

| Release asset | Used by | SHA-256 |
|---|---|---|
| `rgb_model.pth` | `AutoEncoder`, RGB mode | `3f49e1e9b2bafbc1ad041b10eca2494d96ce0a0cdce49f5f1239bd99fd457c7d` |
| `hsv_model.pth` | `AutoEncoder`, HSV mode | `a2b0d34a30851093b36a2a6639b09ffcc38bbd0d8e94ad16e97f51381944e111` |
| `xyz_model.pth` | `AutoEncoder`, XYZ mode | `5d6cfa9bf2fda54d63d6e62316637d3d756fdbbfec40f5ddefd40b676ec0d85b` |
| `imagenette_model.pth` | `ImageNetAutoEncoder` | `5937c64416a63efcb73fb17a379f0cdb77b9d74c2adad0127ab9060413522a36` |
| `cnn_model.pth` | `ColorizationNet` | `91b01f1e9628d11968f57be4c8f074e2d116d89ac4e6c3d082e83e7ccdfb4ecb` |
| `autoencoder_model.pth` | Duplicate of `rgb_model.pth` | `3f49e1e9b2bafbc1ad041b10eca2494d96ce0a0cdce49f5f1239bd99fd457c7d` |

The checkpoints were saved from CUDA tensors, so load them with
`map_location=...` on CPU-only machines.

## Known issues

These are preserved as-is; they are why the project is being rewritten.

- **No LAB.** All models regress RGB/HSV/XYZ from a grayscale input instead of
  predicting chroma in CIELAB. Outputs are desaturated, and luminance is not
  preserved.
- **HSV is ill-posed.** Hue is circular, but the model uses a sigmoid and an
  ordinary pixel loss, so the 0/1 wraparound is not handled.
- **XYZ is not a chroma space.** Predicting X, Z from sRGB luma is physically
  inconsistent, and Y still carries luminance.
- **The old app overwrites its color conversion.** `app_legacy.py` computes
  `hsv_to_rgb` / `xyz_to_rgb`, then unconditionally overwrites that result on the
  next line with `ToPILImage()(colorized_tensor)`. XYZ output is therefore
  displayed as if it were RGB.
- **HSV is unreachable in the UI.** The dropdown omits it even though
  `switch_case.py` supports it.
- **CNN resolution mismatch.** Trained on CIFAR-10 at 32×32, then applied to
  150×150 images.
- **Narrow, mismatched datasets.** Landscape-only, CIFAR-10, and Imagenette were
  used for different models, so the old checkpoints are not comparable to each
  other.
- **No training code.** The repository never contained the training scripts,
  losses, splits, or preprocessing, so these checkpoints cannot be reproduced.

## Running the old app

Install the old dependencies manually. The original `requirements.txt` was
invalid (colon syntax, missing Streamlit/Kornia, `PIL` instead of `Pillow`), so
do not use it:

```bash
pip install torch torchvision streamlit kornia pillow
```

Download the release assets into a `models/` directory at the repository root:

```
models/rgb_model.pth
models/hsv_model.pth
models/xyz_model.pth
models/imagenette_model.pth
models/cnn_model.pth
```

Then run from the repository root, with `legacy/` on `PYTHONPATH`:

```bash
PYTHONPATH=legacy streamlit run legacy/app_legacy.py
```

The model paths in the old app are relative to the current working directory
(`./models/...`), so it must be launched from the repository root.

## Historical baseline status

The old checkpoints are **not part of the controlled comparison**. Their
datasets, resolutions, and training budgets differ, and no training code
exists. Any numbers from them are not comparable to Panel A.

| Model | ΔE00 ↓ | Chroma MAE ↓ | PSNR ↑ | SSIM ↑ | Colorfulness | Notes |
|---|---|---|---|---|---|---|
| Landscape autoencoder (RGB) | — | — | — | — | — | Not evaluated on the new COCO benchmark |
| Landscape autoencoder (HSV) | — | — | — | — | — | Not evaluated |
| Landscape autoencoder (XYZ) | — | — | — | — | — | Not evaluated |
| Imagenette autoencoder | — | — | — | — | — | Not evaluated |
| CIFAR-10 CNN | — | — | — | — | — | Not evaluated |

They may be evaluated once on the new COCO test split purely as historical
context, and this table will be filled in then. They will remain excluded from
the controlled color-space study.

## Provenance

- Archived at tag `v0-legacy` (commit `e4c870a`).
- Original repository layout: models and app at the repository root.
- The project continues in the new `src/` layout with the LAB comparison study.
