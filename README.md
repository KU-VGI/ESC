# [CVPR 2025 Highlight] ESC: Erasing Space Concept for Knowledge Deletion

Official PyTorch implementation for CVPR 2025 paper:  

**ESC: Erasing Space Concept for Knowledge Deletion**   

[Tae-Young Lee](https://github.com/TY-LEE-KR)\*, Sundong Park*, Minwoo Jeon*, Hyoseok Hwang<sup>$\dagger$</sup> , and Gyeong-Moon Park<sup>$\dagger$</sup> 

[![arXiv](https://img.shields.io/badge/arXiv-2405.09879-b31b1b.svg)](https://arxiv.org/abs/2504.02199) 

## 📋 To-Do

- [x] detailed scripts.
- [x] upload *requirements.txt*.
- [x] upload the original model checkpoints.


# Environment
- Python 3.10.x
- Torch 2.1.0
- Torchvision 0.16.0
- NVIDIA GeForce RTX 3090 / A5000 / A6000
- CUDA 12.2



# Getting Started
## Environment
```bash
git clone git@github.com/KHU-VGI/ESC.git
cd ESC
conda create -n ESC python=3.10
conda activate ESC
pip install -r requirements.txt
```

# Model Checkpoints
[[checkpoints]](https://drive.google.com/drive/folders/1yzahmyaNcP9Y10PTDzGdJGfqP617vrzB?usp=drive_link) 


# For conducting unlearning
```bash
# ESC
python unlearn.py --exp ESC_AllCNN_CIFAR10 \
                  --model_name AllCNN \
                  --method ESC \
                  --evaluation \
                  --mia \
                  --kr

# ESC-T
python unlearn.py --exp ESC_T_AllCNN_CIFAR10 \
                  --model_name AllCNN \
                  --method ESC_T \
                  --evaluation \
                  --mia \
                  --kr
```


# BibTex
```
@inproceedings{lee2025esc,
    author    = {Lee, Tae-Young and Park, Sundong and Jeon, Minwoo and Hwang, Hyoseok and Park, Gyeong-Moon},
    title     = {ESC: Erasing Space Concept for Knowledge Deletion},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2025},
    pages     = {5010--5019}
}
```
