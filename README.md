# Setup

Clone this repository.
```bash
git clone https://github.com/thwan11/Disentangle_CLIP.git
```

If you don't have conda...
```bash
sh setup.sh
source ~/.bashrc
```

Create conda environment.
```bash
conda env create -f environment.yaml
conda activate clip
```

Download VOC2007 and VOC 2012.
```bash
python VOC_download.py
```