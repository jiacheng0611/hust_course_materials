"""
@Description :   网络训练
@Author      :   tqychy 
@Time        :   2023/12/13 16:20:46
"""

import datetime
import os
import sys
from copy import deepcopy

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

from dataset import DigitsDataset
from utils import *

matplotlib.use("Agg")
plt.style.use(["science", "grid", "no-latex"])
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.figsize"] = (15.2, 8.8)
fontdict = {"fontsize": 15}

sys.path.append("./")

"""
TODO:
2. tensorboard
"""


def training(verbose=True, save_logs=True, save_fig=True):
    """
    模型训练函数
    :param verbose: 是否打印输出
    :param save_logs: 是否保存模型参数和日志
    :param save_fig: 是否画图
    :returns tuple[float, float]: 准确率和 macro f1 分数
    """
    set_seed(params["random_seed"])
    device = params["device"]
    norm_mean = [0.449]  # ImageNet 中所有图像的平均值
    norm_std = [0.226]  # ImageNet 中所有图像的标准差
    # 训练集数据增强
    train_transform = transforms.Compose([
        # transforms.RandomCrop(28, padding=4),
        transforms.ToTensor(),
        # transforms.Normalize(norm_mean, norm_std),
    ])
    # 验证集数据增强
    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(norm_mean, norm_std),
    ])

    # 构建 Dataset
    train_data = DigitsDataset(
        info_names=("./MNIST/train0.txt", "./MNIST/train1.txt"), is_train=True,  transform=train_transform)
    valid_data = DigitsDataset(
        info_names=("./MNIST/valid0.txt", "./MNIST/valid1.txt"), is_train=False,  transform=valid_transform)

    # 构建 DataLoader
    train_data_loader = DataLoader(
        dataset=train_data, batch_size=params["batch_size"], shuffle=True)
    valid_data_loader = DataLoader(
        dataset=valid_data, batch_size=params["batch_size"])

    # 构建网络
    net = params["net"](classes=2).to(device)
    # net.initialize_weights()

    # 损失函数
    criterion = nn.CrossEntropyLoss()

    # 优化器
    optimizer = optim.AdamW(net.parameters(), lr=params["lr"])

    # 训练
    log_path = os.path.join(
        "params", datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    save_path = os.path.join(log_path, f"{net.name()}checkpoints")
    if save_logs:
        os.makedirs(log_path, exist_ok=True)
        log = open(os.path.join(log_path, "train_log.txt"), "w")
        os.makedirs(save_path, exist_ok=True)

    training_losses_per_batch = []
    training_losses_per_epoch = []
    training_f1_per_epoch = []
    training_f1_per_batch = []
    training_acc_per_epoch = []
    training_acc_per_batch = []
    valid_losses = []
    valid_f1s = []
    valid_accs = []
    if verbose:
        print("Using {} for training.".format(device))
    if save_logs:
        log.write("Using {} for training.\n".format(device))

    # 断点继续训练
    min_epoch = 0
    if params["is_continue"] == True:
        try:
            check_point = torch.load(params["param_path"])
            net.load_state_dict(check_point["net_stat_dicts"])
            optimizer.load_state_dict(check_point["optimizer_stat_dicts"])
            min_epoch = check_point["epoch"]
        except:
            raise RuntimeError("未找到保存参数的文件")

    for epoch in range(min_epoch, params["epoches"]):

        mean_loss = 0.
        mean_f1 = 0.
        mean_acc = 0.
        correct = 0.
        total = 0.
        max_acc = 0.  # 每个epoch的最大的准确率
        max_f1 = 0.
        is_best = False  # 是否是当前最优的参数
        correct_val = 0.
        total_val = 0.
        loss_val = 0.

        net.train()
        for i, data in enumerate(train_data_loader):

            inputs, labels = data
            # print(f"training labels are: {labels} {any(labels)} {all(labels)}")
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs).to(device)

            optimizer.zero_grad()
            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).squeeze().sum().cpu().numpy()
            y_true, y_pred = labels.cpu().numpy(), predicted.cpu().numpy()
            training_f1, training_acc = f1_score(
                y_true, y_pred, average="macro"), accuracy_score(y_true, y_pred)
            mean_loss += loss.item()
            mean_f1 += training_f1
            mean_acc += training_acc
            training_losses_per_batch.append(loss.item())
            training_f1_per_batch.append(training_f1)
            training_acc_per_batch.append(training_acc)
            if verbose:
                print("Training:Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".format(
                    epoch, params["epoches"], i+1, len(train_data_loader), loss.item(), correct / total))
            if save_logs:
                log.write("Training:Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}\n".format(
                    epoch, params["epoches"], i+1, len(train_data_loader), loss.item(), correct / total))

        training_losses_per_epoch.append(mean_loss / len(train_data_loader))
        training_f1_per_epoch.append(mean_f1 / len(train_data_loader))
        training_acc_per_epoch.append(mean_acc / len(train_data_loader))

        # 模型验证
        net.eval()
        with torch.no_grad():
            all_labels = np.array([])
            all_preds = np.array([])
            for j, data in enumerate(valid_data_loader):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = net(inputs).to(device)
                loss = criterion(outputs, labels)

                _, predicted = torch.max(outputs.data, 1)
                # predicted = 1 - predicted
                total_val += labels.size(0)
                correct_val += (predicted ==
                                labels).squeeze().sum().cpu().numpy()
                all_labels = np.hstack((all_labels, labels.cpu().numpy()))
                all_preds = np.hstack((all_preds, predicted.cpu().numpy()))
                loss_val += loss.item()

                if verbose:
                    print("Valid:\t Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".format(
                        epoch, params["epoches"], 1+j, len(valid_data_loader), loss.item(), correct_val / total_val))
                if save_logs:
                    log.write("Valid:\t Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}\n".format(
                        epoch, params["epoches"], 1+j, len(valid_data_loader), loss.item(), correct_val / total_val))

            loss_val_epoch = loss_val / len(valid_data_loader)
            valid_losses.append(loss_val_epoch)
            valid_f1, valid_acc = f1_score(
                all_labels, all_preds, average="macro"), accuracy_score(all_labels, all_preds)
            valid_f1s.append(valid_f1)
            valid_accs.append(valid_acc)
            if max_acc < valid_accs[-1]:
                max_acc = valid_accs[-1]
                max_f1 = valid_f1s[-1]
                is_best = True

        # 保存模型的参数
        check_point = {
            "net_stat_dicts": net.state_dict(),
            "optimizer_stat_dicts": optimizer.state_dict(),
            "epoch": epoch
        }
        save(check_point, save_path, f"Epoch{epoch}.pth", save_logs)
        if is_best == True:  # 保存最优的参数
            check_point = {
                "net_stat_dicts": net.state_dict(),
                "optimizer_stat_dicts": optimizer.state_dict(),
                "epoch": epoch
            }
            save(check_point, save_path, "best.pth", save_logs)

    # 绘图函数
    if save_fig:
        os.makedirs("./images", exist_ok=True)
        # 训练损失函数（batch)
        # plt.title("训练损失函数及评价指标", fontdict=fontdict)
        min_training_loss,  min_training_loss_id = min(
            training_losses_per_batch), training_losses_per_batch.index(min(training_losses_per_batch))
        plt.plot(range(len(training_losses_per_batch)),
                 training_losses_per_batch, "g")
        plt.plot(range(len(training_losses_per_batch)), [min_training_loss for _ in range(
            len(training_losses_per_batch))], "orange", linestyle="--")
        plt.scatter(min_training_loss_id, min_training_loss,
                    c="orange", marker="s")
        plt.text(0, min_training_loss,
                 f"min training loss is {round(min_training_loss, 3)}")
        plt.xlabel("batch number")
        plt.ylabel("loss values")

        plt.savefig(f"./images/{net.name()}训练损失函数（batch).pdf")
        plt.clf()

        # 训练 acc & macro f1（batch)
        min_training_acc,  min_training_acc_id = max(
            training_acc_per_batch), training_acc_per_batch.index(max(training_acc_per_batch))
        plt.plot(range(len(training_acc_per_batch)),
                 training_acc_per_batch, "purple", label="acc")
        plt.plot(range(len(training_acc_per_batch)), [min_training_acc for _ in range(
            len(training_acc_per_batch))], "r", linestyle="--")
        plt.scatter(min_training_acc_id, min_training_acc, c="r", marker="s")
        plt.text(0, min_training_acc + 0.01,
                 f"max training acc is {round(min_training_acc, 3)}")

        min_training_f1,  min_training_f1_id = max(
            training_f1_per_batch), training_f1_per_batch.index(max(training_f1_per_batch))
        plt.plot(range(len(training_f1_per_batch)),
                 training_f1_per_batch, "b", label="macro f1")
        plt.plot(range(len(training_f1_per_batch)), [min_training_f1 for _ in range(
            len(training_f1_per_batch))], "orange", linestyle="--")
        plt.scatter(min_training_f1_id, min_training_f1, c="r", marker="s")
        plt.text(0, min_training_f1 - 0.02,
                 f"max training f1 is {round(min_training_f1, 3)}")
        plt.xlabel("batch number")
        plt.ylabel("metrics")
        plt.legend()
        plt.savefig(f"./images/{net.name()}训练评价指标（batch).pdf")
        plt.clf()

        # 训练验证损失函数
        # plt.title("训练损失函数及评价指标", fontdict=fontdict)
        min_training_loss,  min_training_loss_id = min(
            training_losses_per_epoch), training_losses_per_epoch.index(min(training_losses_per_epoch))
        min_valid_loss,  min_valid_loss_id = min(
            valid_losses), valid_losses.index(min(valid_losses))
        plt.plot(range(len(training_losses_per_epoch)),
                 training_losses_per_epoch, "g", label="train", marker=".")
        plt.plot(range(len(valid_losses)), valid_losses,
                 "purple", label="valid", marker="*")
        plt.plot(range(len(training_losses_per_epoch)), [min_training_loss for _ in range(
            len(training_losses_per_epoch))], "r", linestyle="--")
        plt.plot(range(len(valid_losses)), [min_valid_loss for _ in range(
            len(valid_losses))], "r", linestyle="--")
        plt.scatter(min_training_loss_id, min_training_loss,
                    c="orange", marker="s")
        plt.scatter(min_valid_loss_id, min_valid_loss, c="r", marker="s")
        plt.text(0, min_training_loss - 0.02,
                 f"min training loss is {round(min_training_loss, 3)}")
        plt.text(0, min_valid_loss + 0.01,
                 f"min valid loss is {round(min_valid_loss, 3)}")
        plt.xlabel("epoch number")
        plt.ylabel("loss values")
        plt.legend()

        plt.savefig(f"./images/{net.name()}训练验证损失函数.pdf")
        plt.clf()

        # 训练验证评价指标
        # plt.title("训练验证评价指标", fontdict=fontdict)
        min_training_acc,  min_training_acc_id = max(
            training_acc_per_epoch), training_acc_per_epoch.index(max(training_acc_per_epoch))
        min_valid_acc,  min_valid_acc_id = max(
            valid_accs), valid_accs.index(max(valid_accs))
        plt.plot(range(len(training_acc_per_epoch)),
                 training_acc_per_epoch, "g", label="train acc", marker=".")
        plt.plot(range(len(valid_accs)), valid_accs,
                 "purple", label="valid acc", marker="*")
        plt.plot(range(len(training_acc_per_epoch)), [min_training_acc for _ in range(
            len(training_acc_per_epoch))], "orange", linestyle="--")
        plt.plot(range(len(valid_accs)), [min_valid_acc for _ in range(
            len(valid_accs))], "r", linestyle="--")
        plt.scatter(min_training_acc_id, min_training_acc,
                    c="orange", marker="s")
        plt.scatter(min_valid_acc_id, min_valid_acc, c="r", marker="s")
        plt.text(0, min_training_acc - 0.004,
                 f"max training acc is {round(min_training_acc, 3)}")
        plt.text(0, min_valid_acc + 0.002,
                 f"max valid acc is {round(min_valid_acc, 3)}")
        plt.xlabel("epoch number")
        plt.ylabel("metrics")
        plt.legend()
        plt.savefig(f"./images/{net.name()}训练验证acc.pdf")
        plt.clf()

        min_training_f1,  min_training_f1_id = max(
            training_f1_per_epoch), training_f1_per_epoch.index(max(training_f1_per_epoch))
        min_valid_f1,  min_valid_f1_id = max(
            valid_f1s), valid_f1s.index(max(valid_f1s))
        plt.plot(range(len(training_f1_per_epoch)),
                 training_f1_per_epoch, "g", label="train macro f1", marker=".")
        plt.plot(range(len(valid_f1s)), valid_f1s, "purple",
                 label="valid macro f1", marker="*")
        plt.plot(range(len(training_f1_per_epoch)), [min_training_f1 for _ in range(
            len(training_f1_per_epoch))], "orange", linestyle="--")
        plt.plot(range(len(valid_f1s)), [min_valid_f1 for _ in range(
            len(valid_f1s))], "r", linestyle="--")
        plt.scatter(min_training_f1_id, min_training_f1,
                    c="orange", marker="s")
        plt.scatter(min_valid_f1_id, min_valid_f1, c="r", marker="s")
        plt.text(0, min_training_f1 - 0.004,
                 f"max training f1 is {round(min_training_f1, 3)}")
        plt.text(0, min_valid_f1 + 0.002,
                 f"max valid f1 is {round(min_valid_f1, 3)}")
        plt.xlabel("batch number")
        plt.ylabel("metrics")
        plt.legend()
        plt.savefig(f"./images/{net.name()}训练验证f1.pdf")
        plt.clf()

    if verbose:
        print(f"max acc:{max_acc} max f1:{max_f1}")
    if save_logs:
        log.write(f"max acc:{max_acc} max f1:{max_f1}\n")
        log.close()

    return max_acc, max_f1, training_losses_per_epoch, training_acc_per_epoch


if __name__ == "__main__":
    train, compare = False,  True

    if train:
        params["net"] = LeNet_RB
        training()

    if train:
        params["net"] = ResNet18
        training()

    if compare:
        print("comparing models...")
        params["net"] = LeNet_RB
        _, _, le_losses, le_acc = training(False, False, False)
        params["net"] = ResNet18
        params["batch_size"] = 4000
        params["lr"] = 0.01
        _, _, res_losses, res_acc = training(False, False, False)

        plt.plot(range(len(le_losses)), le_losses,
                 "g", label="LeNet_RB", marker=".")
        plt.plot(range(len(res_losses)), res_losses,
                 "purple", label="ResNet18", marker="*")
        plt.xlabel("epoch number")
        plt.ylabel("loss value")
        plt.legend()
        plt.savefig("./images/模型对比loss.pdf")
        plt.clf()

        plt.plot(range(len(le_acc)), le_acc, "g", label="LeNet_RB", marker=".")
        plt.plot(range(len(res_acc)), res_acc,
                 "purple", label="ResNet18", marker="*")
        plt.xlabel("epoch number")
        plt.ylabel("acc")
        plt.legend()
        plt.savefig("./images/模型对比acc.pdf")
        plt.clf()
