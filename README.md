# DocNLC
This repo is the official implementation of "[DocNLC: A Document Image Enhancement Framework with Normalized and Latent Contrastive Representation for Multiple Degradations](https://doi.org/10.1609/aaai.v38i6.28366)" (AAAI 2024)

![image](https://github.com/RylonW/DocNLC/assets/44424589/88777402-8196-4a18-8c9c-022eb27b552a)

# Requirements
torch == 1.7.1+cu101 

numpy == 1.19.2

opencv-python == 4.5.1.48
# Data Preparation
The structure of the training data is shown below:

├── Hybrid/

│   ├── Degraded/

│   │   ├── Blur/

│   │   ├── Noise/

│   │   ├── Shadow/

│   │   ├── Watermark/

│   │   ├── WithBack/

To generate the training dataset, run:
```bash
python generate_training_dataset.py (Coming soon)
```
Or download from: [Pre-training Dataset](https://1drv.ms/u/s!Aj6X7kgt6NgZjz4ZCAxiVQehsyl6?e=vSTrko) (21.5G)
# Train & Test
We control our hyper-parameters, such as batch size or learning rate, through exclusive yaml files. They are stored in the options folder. For pre-training, fine-tuning and testing, you should specify an appropriate yaml file. We have provided a sample file in the options folder.
## Pre-train
1. Edit ./options/pretrain.yml
2. ```python pretrain.py```
## Fine-tune
1. Edit ./options/finetune.yml
2. ```python finetune.py```
## Test
1. Edit ./options/test.yml
2. ```python test.py```
   
Note that the terminal output during the PSNR test is meaningless. In the next step we will evaluate the output images using the standard skimage.metrics.

# Model Zoo
| Pretrained Model | Pretrained Model |
| :-: | :-: |
| Asymmetric Comparison | [One Drive](https://1drv.ms/u/s!Aj6X7kgt6NgZj0E2cTAuRYDiExC0?e=AldOcL)|
| Symmetric Comparison | [One Drive](https://1drv.ms/u/s!Aj6X7kgt6NgZj0C8umOSrs4CIKtv?e=jYNFca)|
## Acknowledge
Our work is based on the following theoretical works:
- [Barlow Twins](https://proceedings.mlr.press/v139/zbontar21a.html)
- [Instance Normalization](https://openaccess.thecvf.com/content_iccv_2017/html/Huang_Arbitrary_Style_Transfer_ICCV_2017_paper.html)

and we are benefiting a lot from the following projects:
- [facebookresearch/barlowtwins](https://github.com/facebookresearch/barlowtwins)
- [KevinJ-Huang/ExposureNorm-Compensation](https://github.com/KevinJ-Huang/ExposureNorm-Compensation)

## Citation
```bib
@inproceedings{wang2024docnlc,
  title={DocNLC: A Document Image Enhancement Framework with Normalized and Latent Contrastive Representation for Multiple Degradations},
  author={Wang, Ruilu and Xue, Yang and Jin, Lianwen},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={6},
  pages={5563--5571},
  year={2024}
}
```
