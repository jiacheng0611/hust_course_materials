"""
@Description :   模型测试
@Author      :   tqychy 
@Time        :   2023/12/13 17:20:16
"""
import os

import torch
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import DataLoader

from dataset import DigitsDataset
from utils import *


def testing(verbose=True, save_logs=True):
    """
    模型测试函数
    :param verbose: 是否打印输出
    :param save_logs: 是否保存模型参数和日志
    :returns tuple[float, float]: 准确率和 macro f1 分数
    """

    set_seed(params["random_seed"])
    device = params["device"]
    norm_mean = [0.449]  # ImageNet 中所有图像的平均值
    norm_std = [0.226]  # ImageNet 中所有图像的标准差
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
    ])
    test_data = DigitsDataset(
        info_name="./MNIST/valid.txt", transform=test_transform)
    test_data_loader = DataLoader(test_data)

    net = params["net"](classes=10).to(device)
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
        for j, data in enumerate(test_data_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs).to(device)

            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted ==
                            labels).squeeze().sum().cpu().numpy()
            all_labels = np.hstack((all_labels, labels.cpu().numpy()))
            all_preds = np.hstack((all_preds, predicted.cpu().numpy()))
            if verbose:
                print("Valid:\t Iteration[{:0>3}/{:0>3}] Acc:{:.2%}".format(
                    1+j, len(test_data_loader), correct_val / total_val))
            if save_logs:
                log.write("Valid:\t Iteration[{:0>3}/{:0>3}] Acc:{:.2%}\n".format(
                    1+j, len(test_data_loader), correct_val / total_val))

        valid_f1, valid_acc = f1_score(
            all_labels, all_preds, average="macro"), accuracy_score(all_labels, all_preds)
        if verbose:
            print(f"f1:{valid_f1}")
            print(f"acc:{valid_acc}")
            print(classification_report(all_labels, all_preds))
        if save_logs:
            log.write(f"f1:{valid_f1}\n")
            log.write(f"acc:{valid_acc}\n")
            log.write(classification_report(all_labels, all_preds))
            log.close()


if __name__ == "__main__":
    testing()
