# CSSE463-Team01

We downloaded the COCO2017 dataset for our training dataset.

wget -c http://images.cocodataset.org/zips/train2017.zip


/opt/miniforge3/bin/conda create -y -n csse463-torch python=3.11
conda activate csse463-torch 
conda install -y -c pytorch -c nvidia pytorch torchvision torchaudio pytorch-cuda
pip install pillow scikit-image numpy
pip install opencv-python

export CUDA_VISIBLE_DEVICES=6