conda create -n detrex python=3.7 -y
conda activate detrex
git clone https://github.com/IDEA-Research/detrex.git
cd detrex
git submodule init
git submodule update
python -m pip install -e detectron2
pip install -e .
wget https://github.com/IDEA-Research/detrex-storage/releases/download/v0.2.0/dino_swin_large_384_4scale_36ep.pth