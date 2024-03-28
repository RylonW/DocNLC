# DocNLC

[Paper](https://drive.google.com/file/d/1F2K9PFLcK77QWtOweXbkV5SQKwxwUTV2/view?usp=sharing](https://ojs.aaai.org/index.php/AAAI/article/view/28366)) 
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
```python
python generate_training_dataset.py
```

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
