# CSSE463-Team01

Automatic image colorization.

## Code Structure

**Top-level training scripts:**
- `unet_train.py`:  trains a UNet-based colorization model
- `vgg16_train.py`:  trains a VGG16-based colorization model

**`analysis/`** — scripts for evaluating and analyzing model performance, including baseline and CNN evaluation, loss comparisons, hyperparameter grid search, qualitative example finding, dataset statistics, and additional analysis tools.

**`dataset_tools/`** — data pipeline scripts organized by dataset. The `coco/` subfolder handles COCO dataset processing (LAB color space conversion and chunking). The `ImageNet/` subfolder handles ImageNet-256 processing via `redone.sh`, `clean_folders.py`, and `LAB.py`.

**`demo/`** — interactive demo for running colorization on new images. Place inputs in `demo_images/`, run `demo.py`, and find results in `demo_outputs/`. An `evaluate.py` script is also included.

**`samples/`** — output artifacts and qualitative results, including side-by-side comparison images (grayscale / prediction / ground truth), edge-based granularity evaluation images, per-model CSVs, and saved model checkpoint outputs.

**`data/`** — raw and processed datasets (COCO and ImageNet-256). Not tracked in git due to size.

## Environment Setup

Create and activate the conda environment:

```bash
/opt/miniforge3/bin/conda create -y -n csse463-torch python=3.11
conda activate csse463-torch
```

Install PyTorch with CUDA support:

```bash
conda install -y -c pytorch -c nvidia pytorch torchvision torchaudio pytorch-cuda
```

Install Python dependencies:

```bash
pip install pillow scikit-image numpy opencv-python
```

To use a specific GPU, set the following environment variable before training:

```bash
export CUDA_VISIBLE_DEVICES=6
```

## Dataset Setup

> **Warning:** This process may take 10 or more hours depending on machine and network speed.

Create the data folders:

```bash
mkdir -p data/coco data/imagenet
```

Download COCO (train2017):

```bash
wget -c http://images.cocodataset.org/zips/train2017.zip -P data/coco
```

Download ImageNet-256 from Kaggle (may require authentication):

```bash
curl -L -o data/imagenet/imagenet-256.zip \
  https://www.kaggle.com/api/v1/datasets/download/dimensi0n/imagenet-256
```

Move each zip into its respective folder and unzip. Then copy the scripts from `dataset_tools/` into their corresponding dataset directories and run them from the top of each directory.

For ImageNet:

```bash
bash redone.sh
python clean_folders.py  # provide the path to the folder as an argument
python LAB.py
```

For COCO:

```bash
python LAB_convert.py
```

## Training

Run from the project root:

```bash
python unet_train.py   # train the UNet model
python vgg16_train.py  # train the VGG16 model
```

## Demo

Navigate to the `demo/` directory, place any images you want colorized into `demo_images/`, then run:

```bash
cd demo
python demo.py
```

## Notes

Generative AI was used to assist in writing and debugging various parts of this project. Models used include ChatGPT 5.2, Claude Sonnet 4.5, and GitHub Copilot.
