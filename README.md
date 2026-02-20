CSSE463-Team01
==============

CONDA ENV SETUP
---------------
Create and activate the conda environment:

  /opt/miniforge3/bin/conda create -y -n csse463-torch python=3.11
  conda activate csse463-torch

Install PyTorch + CUDA support:

  conda install -y -c pytorch -c nvidia pytorch torchvision torchaudio pytorch-cuda

Install Python dependencies:

  pip install pillow scikit-image numpy opencv-python

(Optional) Select a specific GPU:

  export CUDA_VISIBLE_DEVICES=6


DATASET SETUP (NOT RECOMMENDED)
-------------------------------
WARNING: This process may take 10+ hours depending on your machine and network speed.

1) Create a "data" folder with two subfolders: "coco" and "imagenet"

  mkdir -p data/coco data/imagenet

2) Download the zipped datasets:

  COCO (train2017):
    wget -c http://images.cocodataset.org/zips/train2017.zip -P data/coco

  ImageNet-256 (Kaggle):
    curl -L -o data/imagenet/imagenet-256.zip \
      https://www.kaggle.com/api/v1/datasets/download/dimensi0n/imagenet-256

  NOTE: The ImageNet-256 download may require Kaggle authentication.

  Move your two zipped datasets into the data/coco folder and the data/imagenet folder.

  Unzip both folders (this may take several hours).

  Copy over the files from dataset_tools into their respective folders (dataset_tools/coco and dataset_tools/ImageNet), these are scripts which will help process the data into 512 image chunks so that they can be loaded into the networks faster during training. Run those files in the top of the directory (data/coco or data/ImageNet).

  ImageNet:
    Run redone.sh to generate the folders.
    Run clean_folders.py (you will have to provide it with the path to the folder) to remove all of the folders of images we don't use.
    Run LAB.py to generate the chunks and convert the RGB images into La*b*.

  COCO:  
    Run LAB_convert.py to generate the chunks and convert the RGB images into La*b*.


TRAINING MODELS
---------------
Run from the project root:

  Train the UNet model:
    python unet_train.py

  Train the VGG16 model:
    python vgg26_train.py

DEMO
----
1) Navigate to the demo folder
2) Put any images you want into the demo_images folder
3) Run the demo script from inside the demo folder:

  cd demo
  python demo.py