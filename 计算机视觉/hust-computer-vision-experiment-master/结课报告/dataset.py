"""
@Description :   猫狗数据集的 Dataset
@Author      :   tqychy 
@Time        :   2023/12/20 10:27:27
"""
import os

import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset


class PicDataset(Dataset):
    """
    猫狗分类的 Dataset
    :params data_path: 数据集路径
    """

    def __init__(self, data_path: str):
        super().__init__()
        self.data_path = data_path
        self.data_info = os.listdir(self.data_path)
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __getitem__(self, index):
        img_path = os.path.join(self.data_path, self.data_info[index])
        img = self.transform(Image.open(img_path))
        return img

    def __len__(self):
        return len(self.data_info)


if __name__ == "__main__":
    data_path = "./实验四模型和测试图片（PyTorch）/data4"
    dataset = PicDataset(data_path)
    for pic in dataset:
        print(pic.shape)
