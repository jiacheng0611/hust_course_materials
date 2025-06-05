"""
@Description :   读取MNIST数据集并转化成图片，放入 train、valid 和 test 文件夹
@Author      :   tqychy 
@Time        :   2023/12/13 15:23:03
"""
import os
import shutil
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torchvision.datasets.mnist as mnist
from PIL import Image
from skimage import io
from torch.utils.data import Dataset
from tqdm import tqdm

matplotlib.use("TKAgg")
plt.style.use(["science", "grid", "no-latex"])
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.figsize"] = (15.2, 8.8)
fontdict = {"fontsize": 15}

sys.path.append("./")


class DigitsDataset(Dataset):
    """
    MNIST Dataset
    """

    def __init__(self, info_name: str, transform=None):
        super().__init__()
        self.data_info = self.get_img_info(info_name)
        self.transform = transform

    def __getitem__(self, index):
        path_img, label = self.data_info[index]
        img = Image.open(path_img)

        if self.transform is not None:
            img = self.transform(img)   # 在这里做transform，转为tensor等等

        return img, label

    def __len__(self):
        return len(self.data_info)

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
                    path, label = line.split()
                    label = int(label)
                    data_info.append((path, label))
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
    print("training set :", train_set[0].size())
    print("valid set :", valid_set[0].size())

    with open(os.path.join(pic_file_dir, 'train.txt'), 'w') as f_train:
        print("正在生成训练集...")
        train_data_path = os.path.join(pic_file_dir, 'train')
        valid_data_path = os.path.join(pic_file_dir, "valid")
        if os.path.exists(train_data_path):
            shutil.rmtree(train_data_path)
        if os.path.exists(valid_data_path):
            shutil.rmtree(valid_data_path)
        os.makedirs(train_data_path)
        os.makedirs(valid_data_path)

        train_set = zip(train_set[0], train_set[1])
        cnt = 0
        for img, label in tqdm(train_set):
            img_path = os.path.join(train_data_path, str(cnt)+'.jpg')
            io.imsave(img_path, img.numpy())
            f_train.write(img_path+' '+str(int(label))+'\n')
            cnt += 1

    with open(os.path.join(pic_file_dir, 'valid.txt'), 'w') as f_valid:
        print("正在生成验证集...")
        data_path = os.path.join(pic_file_dir, 'valid')
        if os.path.exists(data_path):
            shutil.rmtree(data_path)
        os.makedirs(data_path)

        valid_set = zip(valid_set[0], valid_set[1])
        cnt = 0
        for img, label in tqdm(valid_set):
            img_path = os.path.join(data_path, str(cnt) + '.jpg')
            io.imsave(img_path, img.numpy())
            f_valid.write(img_path + ' ' + str(int(label)) + '\n')
            cnt += 1


if __name__ == "__main__":
    gen_data = False
    if gen_data:
        convert_to_img("MNIST")
    train_info = [0 for _ in range(10)]
    valid_info = [0 for _ in range(10)]
    with open(os.path.join("MNIST", 'train.txt'), 'r') as f_train:
        lines = f_train.readlines()

    for line in lines:
        label = int(line.split(' ')[-1].strip('\n'))
        train_info[label] += 1

    with open(os.path.join("MNIST", 'valid.txt'), 'r') as f_valid:
        lines = f_valid.readlines()

    for line in lines:
        label = int(line.split(' ')[-1].strip('\n'))
        valid_info[label] += 1

    x = np.arange(10)
    train_x = x
    valid_x = x
    sns.barplot(x=train_x, y=train_info, palette=sns.color_palette("Greens"))
    for i in range(10):
        plt.text(train_x[i], train_info[i]+5, str(int(train_info[i])),
                 color="k", fontsize=10, horizontalalignment="center")
    plt.savefig("./images/训练集数据个数分布图.pdf")
    plt.clf()

    sns.barplot(x=valid_x, y=valid_info, palette=sns.color_palette("Purples"))
    for i in range(10):
        plt.text(valid_x[i], valid_info[i]+5, str(int(valid_info[i])),
                 color="k", fontsize=10, horizontalalignment="center")
    plt.savefig("./images/验证集数据个数分布图.pdf")
    plt.clf()
