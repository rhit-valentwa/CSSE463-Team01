# CSSE463-Team01

We downloaded the COCO2017 dataset for our training dataset.

wget -c http://images.cocodataset.org/zips/train2017.zip

/opt/miniforge3/bin/conda create -y -n csse463-torch python=3.11
conda activate csse463-torch 
conda install -y -c pytorch -c nvidia pytorch torchvision torchaudio pytorch-cuda
pip install pillow scikit-image numpy
pip install opencv-python

export CUDA_VISIBLE_DEVICES=6

Qualitative Error Analysis
    This section documents preliminary qualitative analysis of the image
colorization model's outputs. The goal of the analysis is to identify commonfailure modes and understand how preceptual quality relates to quantitative metrics succh as MAE. This analysis is exploratory and will be revisited as model architectures and training strategies evolve in later weeks.

    Initial observations across multiple training epochs and model 
checkpoints indicate several recurring patter:
- Color bleeding: Predicted colors frequently bleed across object 
boundaries, particularly in regions with fine structural detail such as treebranches or mountain edges.
- Desaturation: Many outputs exhibit muted or washed-out colors despite relatively low MAE values, suggesting that pixel-wise loss does not fully 
capture perceptual realism.
- Certain regions (e.g., sky and distant terrain) are occasionally assigned implausible colors, even when global structure is preserved.
Epoch instability: While quantitative metrics improve monotonically across 
epochs, qualitative performance sometimes degrades after intermediate epochs, indicating potential overfitting or loss-function limitations.

To support this analysis, inference is performed using multiple saved model checkpoints on the same input images. Outputs are compared side-by-side against ground truth color images to assess perceptual quality. These qualitative findings will guide architectural changes, loss-function experimentation, and evaluation strategies in Weeks 8–10.
