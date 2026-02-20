# CSSE463-Team01

Automatic image colorization — converting grayscale images to color using deep learning.

---

## Project Structure

### Top-Level Training Scripts

| Script | Description |
|---|---|
| `unet_train.py` | Trains a UNet-based colorization model |
| `vgg16_train.py` | Trains a VGG16-based colorization model |

---

### `analysis/`

Scripts for evaluating and analyzing model performance.

| Script(s) | Description |
|---|---|
| `baseline.py`, `CNNtest.py`, `CNNsamples.py` | Baseline and CNN evaluation |
| `compare_losses.py`, `Comparison.py` | Comparing model losses and results |
| `grid_train.py`, `grid_search/` | Hyperparameter grid search |
| `find_examples.py`, `GranulatiryPreds.py` | Qualitative example finding and granularity predictions |
| `initial_dataset_characterization.py` | Dataset exploration and statistics |
| `imagenet_test.py`, `surface_render.py` | Additional analysis tools |

---

### `dataset_tools/`

Data pipeline scripts organized by dataset.

- **`coco/`** — COCO dataset processing (LAB color space conversion, chunking)
- **`ImageNet/`** — ImageNet-256 processing (`redone.sh`, `clean_folders.py`, `LAB.py`)

---

### `demo/`

Interactive demo for running colorization on new images.

| Path | Description |
|---|---|
| `demo.py` | Main demo script |
| `demo_images/` | Input images to colorize |
| `demo_outputs/` | Colorized results |
| `evaluate.py` | Evaluation script |

---

### `samples/`

Output artifacts and qualitative results.

- **`colorization_examples/`**, **`colorization_samples/`** — Side-by-side comparison images (grayscale / prediction / ground truth)
- **`granularity_results/`** — Edge-based evaluation images for UNet and VGG16
- **`qualitative_outputs/`** — Per-model output summaries and CSVs
- **`unet_colorizer_best/`** — Saved best model checkpoint outputs

---

### `data/`

Raw and processed datasets (COCO and ImageNet-256). Not tracked in git due to size.

---

## Environment Setup

### Create and Activate the Conda Environment

```bash
/opt/miniforge3/bin/conda create -y -n csse463-torch python=3.11
conda activate csse463-torch
```

### Install PyTorch with CUDA Support

```bash
conda install -y -c pytorch -c nvidia pytorch torchvision torchaudio pytorch-cuda
```

### Install Python Dependencies

```bash
pip install pillow scikit-image numpy opencv-python
```

### Select a Specific GPU (Optional)

```bash
export CUDA_VISIBLE_DEVICES=6
```

---

## Dataset Setup

> **Warning:** This process may take 10 or more hours depending on machine and network speed.

### 1. Create Data Folders

```bash
mkdir -p data/coco data/imagenet
```

### 2. Download Datasets

**COCO (train2017):**
```bash
wget -c http://images.cocodataset.org/zips/train2017.zip -P data/coco
```

**ImageNet-256 (Kaggle):**
```bash
curl -L -o data/imagenet/imagenet-256.zip \
  https://www.kaggle.com/api/v1/datasets/download/dimensi0n/imagenet-256
```

> **Note:** The ImageNet-256 download may require Kaggle authentication.

### 3. Unzip and Process

Move the zipped datasets into their respective `data/coco` and `data/imagenet` folders, then unzip both. This step may take several hours.

Copy the processing scripts from `dataset_tools/` into their respective dataset folders and run them from the top of each directory.

**ImageNet:**
```bash
bash redone.sh           # Generate folder structure
python clean_folders.py  # Remove unused image folders (provide path as argument)
python LAB.py            # Generate chunks and convert RGB to La*b*
```

**COCO:**
```bash
python LAB_convert.py    # Generate chunks and convert RGB to La*b*
```

---

## Training

Run from the project root.

```bash
# Train the UNet model
python unet_train.py

# Train the VGG16 model
python vgg16_train.py
```

---

## Running the Demo

1. Navigate to the `demo/` directory.
2. Place images to be colorized in `demo_images/`.
3. Run the demo script:

```bash
cd demo
python demo.py
```

---

## Notes

Generative AI was used to assist in writing and debugging various parts of this project. Models used include ChatGPT, Claude Sonnet, and Copilot for auto-complete.