"""
@Description :   模型测试
@Author      :   tqychy 
@Time        :   2023/12/06 16:43:35
"""
# 测试

import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report
from utils import *
from net import FCNet
from dataset import generate_dataset


def testing(params_path: str, save_fig=True):
    """
    模型测试函数
    """
    print(data.columns)
    _, valid_dataset = generate_dataset(
        data[["feature1", "feature2"]].values, data["label"].values)
    valid_dataloader = DataLoader(
        dataset=valid_dataset, batch_size=len(valid_dataset))

    net = FCNet(2, params["hidden_layers_list"], 4,
                params["activate_func"], params["dropout"]).to(device)

    net.load_state_dict(torch.load(params_path)["net"])
    net.eval()
    with torch.no_grad():
        for feature, label in valid_dataloader:
            inputs, labels = feature.float().to(device), label.to(device)
            outputs = net(inputs)

            _, prediction = torch.max(outputs.data, 1)
            y_true, y_pred = labels.cpu().numpy(), prediction.cpu().numpy()
            valid_f1, valid_acc = f1_score(
                y_true, y_pred, average="macro"), accuracy_score(y_true, y_pred)
            print(f"f1:{valid_f1}")
            print(f"acc:{valid_acc}")
            print(classification_report(y_true, y_pred))

    if save_fig:
        # 绘图函数
        # 网格化
        M, N = 800, 800
        x1_min, x2_min = -3, -3
        x1_max, x2_max = 3, 3
        t1 = np.linspace(x1_min, x1_max, M)
        t2 = np.linspace(x2_min, x2_max, N)
        x1, x2 = np.meshgrid(t1, t2)

        # 预测
        x_show = np.stack((x1.flat, x2.flat), axis=1)
        outputs = net(torch.tensor(x_show).float().to(device))
        _, prediction = torch.max(outputs.data, 1)
        y_predict = prediction.cpu().numpy()

        # 配色
        cm_light = matplotlib.colors.ListedColormap(
            ["#A0FFA0", "#8286BE", "#FFA0A0", "#A0A0FF"])
        cm_dark = matplotlib.colors.ListedColormap(["g", "purple", "r", "b"])

        # 绘制预测区域图
        plt.pcolormesh(t1, t2, y_predict.reshape(x1.shape), cmap=cm_light)

        # 绘制原始数据点
        plt.scatter(data["feature1"], data["feature2"], label=None,
                    c=data["label"], cmap=cm_dark, marker='o', s=5)
        plt.xlabel("feature1")
        plt.ylabel("feature2")

        # 绘制图例
        color = ["g", "purple", "r", "b"]
        labels = ["标签1", "标签2", "标签3", "标签4"]
        for i in range(4):
            plt.scatter([], [], c=color[i], s=5,
                        label=labels[i])    # 利用空点绘制图例
        plt.legend(loc="best")
        plt.title("模型的分类结果")

        plt.savefig("./images/模型的分类结果.pdf")
        plt.clf()

    return valid_acc, valid_f1


if __name__ == "__main__":
    set_seed(params["random_seed"])
    data = pd.read_csv("实验一数据.csv", header=0)
    device = params["device"]

    testing(params["param_path"])
