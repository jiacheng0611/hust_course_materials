"""
@Description :   超参数和工具函数
@Author      :   tqychy 
@Time        :   2023/12/13 16:22:30
"""
import os
import random

import numpy as np
import torch

from net import *

params = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",  # 设备
    "random_seed": 1024,  # 随机数种子
    "net": ResNet18,

    "batch_size": 600,
    "epoches": 100,
    "lr": 0.0001,
    "is_continue": False,  # 是否从断点开始继续训练

    "param_path": "./params/20231215_092753/ResNet18checkpoints/best.pth"  # 保存参数的文件的路径
}


def set_seed(seed: int):
    """
    设置随机数种子
    :param seed: 随机数种子
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save(checkpoint: dict, save_path: str, name: str, enable=True):
    """
    保存模型参数
    :param checkpoint: 模型参数信息
    :param save_path: 模型参数保存地址
    :param name: 模型参数名称
    :param enable: 是否启用本函数
    """
    if enable:
        os.makedirs(save_path, exist_ok=True)
        torch.save(checkpoint, os.path.join(save_path, name))
