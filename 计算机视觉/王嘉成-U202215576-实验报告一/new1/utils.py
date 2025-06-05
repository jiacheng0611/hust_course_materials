"""
@Description :   工具函数
@Author      :   tqychy
@Time        :   2023/12/05 19:25:46
"""
import os
import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
plt.style.use(["ggplot"])
plt.rcParams["font.sans-serif"] = ['KaiTi']
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.figsize"] = (15.2, 8.8)
fontdict = {"fontsize": 15}


# 超参数字典
params = {
    "device": "cuda" if torch.cuda.is_available() else "cpu", # 设备
    "random_seed": 1024, # 随机数种子
    "valid_ratio": 0.1, # 测试集比例
    "hidden_layers_list": [50], # 隐藏层神经元个数
    "activate_func": nn.LeakyReLU, # 激活函数
    "early_stop": False, # 是否使用早停
    "patience":20, # 早停容忍轮数

    "batch_size": 3600, # 批大小
    "epoches": 200, # 最大训练轮数
    "lr": 1, # 学习率
    "dropout": 0.4, # dropout 概率
    "l1_weight": 0.001, # L1 损失系数
    "l2_weight": 0.0,# L2 损失系数

    "param_path": "./params/latest/best.pth" # 测试使用的参数
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
        full_path = os.path.join(save_path, name)
        print(f"Saving model to {full_path}")
        torch.save(checkpoint, os.path.join(save_path, name))

def early_stop(valid_accs, patience=20, delta=0.):
    """
    早停判断函数
    :param valid_accs: 每个epoch的验证分数列表
    :param patience: 容忍的epoch数量，默认为5
    :param delta: 验证分数的最小变化量，默认为0
    :returns bool: 是否应该进行早停
    """
    best_acc = max(valid_accs)  # 获取最佳验证分数
    best_epoch = valid_accs.index(best_acc)  # 获取最佳验证分数对应的epoch
    cnt = 0
    for i in range(best_epoch, len(valid_accs)):
        if best_acc > delta + valid_accs[i]:
            cnt += 1
    if cnt >= patience:
        # print("满足早停条件，停止训练。")
        return True
    return False


if __name__ == "__main__":
    save({}, "params/latest","best.pth")