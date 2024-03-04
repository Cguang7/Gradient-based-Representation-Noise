# Gradient-based-Representation-Noise
Our code is based on https://github.com/Mi-Peng/Sparse-Sharpness-Aware-Minimization

## Requirements

Python 3.8, Pytorch 1.8

## Training

To train the model(s) in the paper, run this command:

For example: 
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 --use_env train.py --model wideresnet28x10 --dataset CIFAR10_cutout --datadir /Your/data/path --opt sgd --gradeps 0.09 --lr 0.25 --weight_decay 5e-4 --seed 3
