"""
@Description :   超参数和工具函数
@Author      :   tqychy 
@Time        :   2023/12/20 16:22:30
"""
import os
import random

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

matplotlib.use("Agg")
plt.rcParams["font.sans-serif"] = ['KaiTi']
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.figsize"] = (15.2, 8.8)
fontdict = {"fontsize": 15}

params = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",  # 设备
    "random_seed": 1024,  # 随机数种子
    "data_path": "./实验四模型和测试图片（PyTorch）/data4",  # 测试集的路径
    "net_path": "./实验四模型和测试图片（PyTorch）/torch_alex.pth",  # 预训练网络参数的路径

    "cam_type": "LayerCAM",  # 可解释性算法的名称
    "flip": False
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


def scale_cam_image(cam, target_size=None):
    """
    对 CAM 图像进行缩放
    :params cam: CAM 矩阵
    :params target_size: 目标大小
    :returns: 结果图像
    """
    result = []
    for img in cam:
        img = img - np.min(img)
        img = img / (1e-7 + np.max(img))
        if target_size is not None:
            img = cv2.resize(img, target_size)
        result.append(img)
    result = np.float32(result)

    return result


def save_result_image(img: np.ndarray, mask: np.ndarray, save_path: str) -> None:
    """
    将 CAM 热力图和原图像叠加，生成结果图像
    :param img: 原始图像。
    :param mask: cam热力图。
    :param save_path: 结果图像的存储路径
    """
    _, ax = plt.subplots()
    ax.imshow(img)
    ax.set_xticks([])
    ax.set_yticks([])
    cbar = ax.figure.colorbar(ax.imshow(mask, cmap='jet', alpha=0.5))
    cbar.ax.set_ylabel('Heatmap Intensity', rotation=90)
    plt.savefig(save_path)


def save_feature_maps(feature_maps: torch.Tensor, fig_name: str):
    """
    绘制指定卷积层的特征图
    :params feature_maps: 卷积层的特征图的张量
    :params fig_name: 保存的图片的路径
    """
    feature_maps = feature_maps.squeeze()

    fig = plt.figure()
    for i in range(feature_maps.shape[0]):
        feature_map = feature_maps[i, :, :].squeeze()
        ax = fig.add_subplot(16, 16, i+1)
        cbar = ax.imshow(feature_map, alpha=0.8)
        ax.axis("off")

    cax = fig.add_axes([0.92, 0.15, 0.01, 0.7])
    fig.colorbar(cbar, cax=cax)
        
    plt.savefig(fig_name, bbox_inches='tight')
