"""
@Description :   模型测试
@Author      :   tqychy 
@Time        :   2023/12/13 17:20:16
"""
import os

import matplotlib
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from sklearn.metrics import (accuracy_score, classification_report, f1_score,
                             roc_auc_score, roc_curve)
from torch.utils.data import DataLoader

from dataset import DigitsDataset
from utils import *

matplotlib.use("Agg")
plt.style.use(["science", "grid", "no-latex"])
plt.rcParams["font.sans-serif"] = ['KaiTi']
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.figsize"] = (15.2, 8.8)
fontdict = {"fontsize": 15}


def testing(verbose=True, save_logs=True, save_figs=True):
    """
    模型测试函数
    :param verbose: 是否打印输出
    :param save_logs: 是否保存模型参数和日志
    :returns tuple[float, float, float, float]: 准确率和 macro f1 分数和 roc 曲线
    """

    set_seed(params["random_seed"])
    device = params["device"]
    norm_mean = [0.449]  # ImageNet 中所有图像的平均值
    norm_std = [0.226]  # ImageNet 中所有图像的标准差
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(norm_mean, norm_std),
    ])
    test_data = DigitsDataset(
        info_names=("./MNIST/valid0.txt", "./MNIST/valid1.txt"), is_train=False,  transform=test_transform)
    test_data_loader = DataLoader(test_data, batch_size=params["batch_size"])

    net = params["net"](classes=2).to(device)
    try:
        check_point = torch.load(params["param_path"])
        net.load_state_dict(check_point["net_stat_dicts"])
    except:
        raise RuntimeError("未找到保存参数的文件")
    if save_logs:
        log_path = os.path.dirname(os.path.dirname(params["param_path"]))
        log = open(os.path.join(log_path, "test_log.txt"), "w")
    net.eval()
    with torch.no_grad():
        total_val = 0.
        correct_val = 0.
        all_labels = np.array([])
        all_preds = np.array([])
        all_probas = np.array([])
        for j, data in enumerate(test_data_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs).to(device)
            probas = torch.nn.Softmax()(outputs)[:, 1]

            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted ==
                            labels).squeeze().sum().cpu().numpy()
            all_labels = np.hstack((all_labels, labels.cpu().numpy()))
            all_preds = np.hstack((all_preds, predicted.cpu().numpy()))
            all_probas = np.hstack((all_probas, probas.cpu().numpy()))
            if verbose:
                print("Valid:\t Iteration[{:0>3}/{:0>3}] Acc:{:.2%}".format(
                    1+j, len(test_data_loader), correct_val / total_val))
            if save_logs:
                log.write("Valid:\t Iteration[{:0>3}/{:0>3}] Acc:{:.2%}\n".format(
                    1+j, len(test_data_loader), correct_val / total_val))

        valid_f1, valid_acc = f1_score(
            all_labels, all_preds, average="macro"), accuracy_score(all_labels, all_preds)
        roc_auc = roc_auc_score(all_labels, all_probas)
        fpr, tpr, _ = roc_curve(all_labels, all_probas)
        if verbose:
            print(f"f1:{valid_f1}")
            print(f"acc:{valid_acc}")
            print(f"ROC AUC Score: {roc_auc}")
            print(classification_report(all_labels, all_preds))
        if save_logs:
            log.write(f"f1:{valid_f1}\n")
            log.write(f"acc:{valid_acc}\n")
            log.write(f"ROC AUC Score: {roc_auc}\n")
            log.write(classification_report(all_labels, all_preds))
            log.close()
        if save_figs:
            plt.plot(fpr, tpr, c="purple",
                     label=f'ROC curve (AUC = {round(roc_auc, 3)})')
            plt.fill_between(fpr, tpr, color="purple", alpha=0.2)
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.legend()
            plt.savefig(f"./images/{net.name()}ROC曲线.pdf")
            plt.clf()

    return valid_f1, valid_acc, fpr, tpr


if __name__ == "__main__":
    test, compare = True, False
    if test:
        testing()
    if compare:
        le_params = {
            "device": "cuda" if torch.cuda.is_available() else "cpu",  # 设备
            "random_seed": 1024,  # 随机数种子
            "net": LeNet_RB,

            "dataset_thres": 0.5, # 标签为 0 的数据的占比

            "batch_size": 10000,
            "epoches": 30,
            "lr": 0.005,
            "is_continue": False,  # 是否从断点开始继续训练

            "param_path": "./params/20231219_025513/LeNet_RBcheckpoints/best.pth"  # 保存参数的文件的路径
        }
        res_params = {
            "device": "cuda" if torch.cuda.is_available() else "cpu",  # 设备
            "random_seed": 1024,  # 随机数种子
            "net": ResNet18, # LeNet_RB,

            "dataset_thres": 0.5, # 标签为 0 的数据的占比

            "batch_size": 4000, # 10000,
            "epoches": 30,
            "lr": 0.01, # 0.005,
            "is_continue": False,  # 是否从断点开始继续训练

            "param_path": "./params/20231218_142134/ResNet18checkpoints/best.pth"  # 保存参数的文件的路径
        }
        params = le_params
        _, _, fpr1, tpr1 = testing(False, False, False)
        params = res_params
        _, _, fpr2, tpr2 = testing(False, False, False)
        plt.plot(fpr1, tpr1, c="g", label='LeNet_RB')
        plt.fill_between(fpr1, tpr1, color="g", alpha=0.2)
        plt.plot(fpr2, tpr2, c="purple", label='ResNet18')
        plt.fill_between(fpr2, tpr2, color="purple", alpha=0.2)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()
        plt.savefig(f"./images/模型对比ROC曲线.pdf")
        plt.clf()

    
