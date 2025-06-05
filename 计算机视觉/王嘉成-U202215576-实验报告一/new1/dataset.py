"""
@Description :   数据预处理
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import torch
from torch.utils.data import TensorDataset
from scipy.stats import multivariate_normal
from sklearn.model_selection import train_test_split
from utils import params
matplotlib.use("Agg")
plt.style.use(["ggplot"])
plt.rcParams["font.sans-serif"] = ['KaiTi']
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.figsize"] = (15.2, 8.8)
fontdict = {"fontsize": 15}


def generate_dataset(features: np.ndarray, labels: np.ndarray):
    """
    生成 Dataset
    """
    labels -= min(labels)
    X_train, X_valid, y_train, y_valid = train_test_split(
        features, labels, test_size=params["valid_ratio"], random_state=params["random_seed"], shuffle=True)
    return (TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
            TensorDataset(torch.tensor(X_valid), torch.tensor(y_valid)))


if __name__ == "__main__":

    raw_data = pd.read_csv("实验一数据.csv", header=0)
    print("数据个数分布")
    print(raw_data.head())
    nums = raw_data["label"].value_counts()
    print(nums)
    # plt.title("数据个数分布")
    # plt.bar(range(len(nums)), nums.values, width=0.3, tick_label=["标签1", "标签2", "标签3", "标签4"], color=["#A0FFA0", "#8286BE", "#FFA0A0", "#A0A0FF"])
    # plt.savefig("./images/数据个数分布图.pdf")
    # plt.clf()

    # plt.title("数据分布图")
    # sns.scatterplot(x="feature1", y="feature2", data=raw_data, hue="label")
    # plt.savefig("./images/数据分布图.pdf")
    # plt.clf()

    means, stds = raw_data.groupby(
        "label").mean(), raw_data.groupby("label").std()
    print("数据拟合值")
    print(means)
    print(stds)
    fig = plt.figure()
    fig.suptitle("数据分布拟合图")
    cmaps = ["", "summer", "viridis", "rainbow", "autumn"]
    for label in means.index:
        ax = fig.add_subplot(2, 2, label, projection='3d')
        ax.set_title(f"标签 {label} 数据分布拟合图")
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$y$')
        x = np.linspace(means.loc[label]["feature1"] - 6 * stds.loc[label]["feature1"],
                        means.loc[label]["feature1"] + 6 * stds.loc[label]["feature1"], 10000)
        y = np.linspace(means.loc[label]["feature2"] - 6 * stds.loc[label]["feature2"],
                        means.loc[label]["feature2"] + 6 * stds.loc[label]["feature2"], 10000)
        X, Y = np.meshgrid(x, y)
        pos = np.dstack((X, Y))
        Z = multivariate_normal.pdf(
            pos, mean=means.loc[label], cov=stds.loc[label])
        ax.plot_surface(X, Y, Z, cmap=cmaps[label])
    plt.savefig("./images/数据分布拟合图.pdf")
    plt.clf()