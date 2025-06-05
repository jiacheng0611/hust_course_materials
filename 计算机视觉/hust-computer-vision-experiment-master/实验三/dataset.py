"""
@Description :   读取MNIST数据集并转化成图片，放入 train、valid 和 test 文件夹
@Author      :   tqychy 
@Time        :   2023/12/13 15:23:03
"""
import os
import random
import shutil
import sys
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torchvision.datasets.mnist as mnist
from PIL import Image
from skimage import io
from torch.utils.data import Dataset
from tqdm import tqdm

from utils import *

matplotlib.use("TKAgg")
plt.style.use(["science", "grid", "no-latex"])
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.figsize"] = (15.2, 8.8)
fontdict = {"fontsize": 15}

sys.path.append("./")
warnings.filterwarnings("ignore")


class DigitsDataset(Dataset):
    """
    MNIST Dataset
    """
    def __init__(self, info_names: tuple[str], is_train:bool, transform=None):
        """
		:params info_names: 元组，两种标签文件的路径
		:params is_train: 是否是训练集
		:params transform: 数据增强函数，默认无数据增强 (None)
		"""
        super().__init__()
        set_seed(params["random_seed"])
        self.train = is_train
        self.data_info_0 = self.get_img_info(info_names[0])
        self.data_info_1 = self.get_img_info(info_names[1])
        random.shuffle(self.data_info_0)
        random.shuffle(self.data_info_1)
        # self.data_info_0 = self.data_info_0[:1 + len(self.data_info_1)]
        self.transform = transform

    def __getitem__(self, index):
        # set_seed(params["random_seed"])
        if self.train:
            if random.random() < params["dataset_thres"]:
                data_info = self.data_info_0
                label = 0
            else:
                data_info = self.data_info_1
                label = 1
            path_img1, path_img2 = data_info[index]
        else:
            if index < len(self.data_info_0):
                path_img1, path_img2 = self.data_info_0[index]
                label = 0
            else:
                path_img1, path_img2 = self.data_info_1[index - len(self.data_info_0)]
                label = 1
        
        img1, img2 = Image.open(path_img1), Image.open(path_img2)
        if self.transform is not None:
            img1, img2 = self.transform(img1), self.transform(img2)

        return torch.cat((img1, img2), dim=0), label

    def __len__(self):
        if self.train:
            return len(self.data_info_1)
        else:
            return len(self.data_info_0) + len(self.data_info_1)

    @staticmethod
    def get_img_info(info_name):
        """
        读入含有图片路径和标签的文档
        :params info_name: 含有图片路径和标签的文档的名称
        """
        data_info = []
        with open(info_name) as f:
            if f == None:
                print("未找到图片预处理文档")
            else:
                for line in f.readlines():
                    path1, path2 = line.split()
                    data_info.append((path1, path2))
        return data_info


def convert_to_img(pic_file_dir: str):
    """
    把MNIST数据集转换成图片
    :params pic_file_dir: 数据集文件所在目录和train、valid和test文件夹所在目录
    """
    train_set = (
        mnist.read_image_file(os.path.join(
            pic_file_dir, 'train-images.idx3-ubyte')),
        mnist.read_label_file(os.path.join(
            pic_file_dir, 'train-labels.idx1-ubyte'))
    )
    valid_set = (
        mnist.read_image_file(os.path.join(
            pic_file_dir, 't10k-images.idx3-ubyte')),
        mnist.read_label_file(os.path.join(
            pic_file_dir, 't10k-labels.idx1-ubyte'))
    )
    train_thres, valid_thres = int(
        train_set[0].size()[0] * 0.1), int(valid_set[0].size()[0] * 0.1)
    train_set_len0, train_set_len1, valid_set_len0, valid_set_len1 = 0, 0, 0, 0

    with open(os.path.join(pic_file_dir, 'train0.txt'), 'w') as f_train_0:
        with open(os.path.join(pic_file_dir, 'train1.txt'), 'w') as f_train_1:
            print("正在生成训练集...")
            train_data_path = os.path.join(pic_file_dir, 'train')
            valid_data_path = os.path.join(pic_file_dir, "valid")
            if os.path.exists(train_data_path):
                shutil.rmtree(train_data_path)
            if os.path.exists(valid_data_path):
                shutil.rmtree(valid_data_path)
            os.makedirs(train_data_path)
            os.makedirs(valid_data_path)

            train_set = list(zip(train_set[0], train_set[1]))
            for i in tqdm(range(train_thres)):
                img1, label1 = train_set[i]
                img_path1 = os.path.join(train_data_path, str(i)+'.jpg')
                io.imsave(img_path1, img1.numpy())
                for j in range(1 + i):
                    _, label2 = train_set[j]
                    img_path2 = os.path.join(train_data_path, str(j)+'.jpg')
                    if label1 != label2:
                        train_set_len0 += 1
                        f_train_0.write(img_path1 + ' ' + img_path2 + '\n')
                    else:
                        train_set_len1 += 1
                        f_train_1.write(img_path1 + ' ' + img_path2 + '\n')

    with open(os.path.join(pic_file_dir, 'valid0.txt'), 'w') as f_valid_0:
        with open(os.path.join(pic_file_dir, 'valid1.txt'), 'w') as f_valid_1:
            print("正在生成验证集...")
            data_path = os.path.join(pic_file_dir, 'valid')
            if os.path.exists(data_path):
                shutil.rmtree(data_path)
            os.makedirs(data_path)

            valid_set = list(zip(valid_set[0], valid_set[1]))
            for i in tqdm(range(valid_thres)):
                img1, label1 = valid_set[i]
                img_path1 = os.path.join(data_path, str(i) + '.jpg')
                io.imsave(img_path1, img1.numpy())
                for j in range(1 + i):
                    _, label2 = valid_set[j]
                    img_path2 = os.path.join(data_path, str(j) + '.jpg')
                    if label1 != label2:
                        valid_set_len0 += 1
                        f_valid_0.write(img_path1 + ' ' + img_path2 + '\n')
                    else:
                        valid_set_len1 += 1
                        f_valid_1.write(img_path1 + ' ' + img_path2 + '\n')

    print("数据集生成完成：")
    print(f"训练集：0: {train_set_len0} 1: {train_set_len1}")
    print(f"验证集：0: {valid_set_len0} 1: {valid_set_len1}")


if __name__ == "__main__":
    gen_data, figs = False, True
    if gen_data:
        convert_to_img("MNIST")
    if figs:
        train_info = [0, 0]
        valid_info = [0, 0]
        with open(os.path.join("MNIST", 'train0.txt'), 'r') as f_train_0:
            with open(os.path.join("MNIST", 'train1.txt'), 'r') as f_train_1:
                lines0 = f_train_0.readlines()
                lines1 = f_train_1.readlines()
                train_info[0], train_info[1] = len(lines0), len(lines1)

        with open(os.path.join("MNIST", 'valid0.txt'), 'r') as f_valid_0:
            with open(os.path.join("MNIST", 'valid1.txt'), 'r') as f_valid_1:
                lines0 = f_valid_0.readlines()
                lines1 = f_valid_1.readlines()
                valid_info[0], valid_info[1] = len(lines0), len(lines1)

        x = np.arange(2)
        train_x = x
        valid_x = x
        sns.barplot(x=train_x, y=train_info,
                    palette=sns.color_palette("Greens"))
        for i in range(2):
            plt.text(train_x[i], train_info[i]+5, str(int(train_info[i])),
                     color="k", fontsize=10, horizontalalignment="center")
        plt.savefig("./images/训练集数据个数分布图.pdf")
        plt.clf()

        sns.barplot(x=valid_x, y=valid_info,
                    palette=sns.color_palette("Purples"))
        for i in range(2):
            plt.text(valid_x[i], valid_info[i]+5, str(int(valid_info[i])),
                     color="k", fontsize=10, horizontalalignment="center")
        plt.savefig("./images/验证集数据个数分布图.pdf")
        plt.clf()
