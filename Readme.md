# CLIPSep
This repository contains the official PyTorch implementation of "CLIPSep: Learning text-queried sound separation with noisy unlabeled videos", presented in ICLR 2023.  
All source code, including training code and data preprocessing, will be made publicly available soon.

### Setup environment
```
conda env create -f environment.yml
```


### Download the pretrained model
Download the pretrained model from [here](https://drive.google.com/file/d/1fgwT_wFyNjXxmN45D7jU001Azub0PR54/view?usp=sharing)
and extract it in the root directory of the project.

### Run inference
```
OMP_NUM_THREADS=1 python inference.py -o checkpoints/clipsep_nit/ -t data/MUSIC/test.csv -t2 data/vggsound/test.csv --vis_dir outputs
```
### Run evaluation
```
OMP_NUM_THREADS=1 python evaluate.py -o checkpoints/clipsep_nit/ -t data/MUSIC/test.csv -t2 data/vggsound/test.csv --no-pit
```