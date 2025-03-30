# Setup

Clone this repository.
```bash
cd workspace
git clone https://github.com/thwan11/Disentangle_CLIP.git
cd Disentangle_CLIP
```

If you don't have `conda`...
```bash
sh setup.sh
source ~/.bashrc
```

Create conda environment.
```bash
conda env create -f environment.yaml
conda activate clip
```

Download `VOC2007` and `VOC2012` to the `parent` directory.
```bash
python VOC_download.py
```