from loader import load_images, load_labels, decompress_file, create_pairs
from train import train_model
from model import SimpleCNN
from evaluate import evaluate_model
import torch

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 解压数据文件
def decompress_data():
    decompress_file('C:/zhuomianruanjian/python_projects/test2/dataset/data/MNIST/raw/train-images-idx3-ubyte.gz',
                    'train-images-idx3-ubyte')
    decompress_file('C:/zhuomianruanjian/python_projects/test2/dataset/data/MNIST/raw/train-labels-idx1-ubyte.gz',
                    'train-labels-idx1-ubyte')
    decompress_file('C:/zhuomianruanjian/python_projects/test2/dataset/data/MNIST/raw/t10k-images-idx3-ubyte.gz',
                    't10k-images-idx3-ubyte')
    decompress_file('C:/zhuomianruanjian/python_projects/test2/dataset/data/MNIST/raw/t10k-labels-idx1-ubyte.gz',
                    't10k-labels-idx1-ubyte')


# 加载数据
def load_data():
    train_images = load_images('train-images-idx3-ubyte')
    train_labels = load_labels('train-labels-idx1-ubyte')
    test_images = load_images('t10k-images-idx3-ubyte')
    test_labels = load_labels('t10k-labels-idx1-ubyte')

    # 创建图片对
    train_pairs, train_pair_labels = create_pairs(train_images, train_labels)
    test_pairs, test_pair_labels = create_pairs(test_images, test_labels)

    # 数据预处理：标准化图片
    train_pairs = train_pairs.astype('float32') / 255.0
    test_pairs = test_pairs.astype('float32') / 255.0

    # 划分训练集和测试集
    train_pairs = train_pairs[:int(0.1 * len(train_pairs))]  # 90% 用于训练
    train_pair_labels = train_pair_labels[:int(0.1 * len(train_pair_labels))]

    return train_pairs, train_pair_labels, test_pairs, test_pair_labels


def main():
    # 解压数据文件
    decompress_data()

    # 加载数据
    train_pairs, train_pair_labels, test_pairs, test_pair_labels = load_data()

    # 创建模型
    model = SimpleCNN()

    # 训练模型并保存
    train_accuracy, test_accuracy = train_model(
        model,
        train_pairs,
        train_pair_labels,
        test_pairs,
        test_pair_labels,
        num_epochs=10,  # 训练10轮
        batch_size=64,
        lr=0.001
    )

    # 保存训练好的模型
    torch.save(model.state_dict(), 'simple_cnn_model.pth')
    print(f"Model saved to 'simple_cnn_model.pth'")

    # 加载保存的模型权重
    model.load_state_dict(torch.load('simple_cnn_model.pth'))
    model.eval()  # 设置模型为评估模式

    # 打印最终训练集和测试集的准确率
    print(f"Final Train Accuracy: {train_accuracy * 100:.2f}%")
    print(f"Final Test Accuracy: {test_accuracy * 100:.2f}%")

    # 评估模型
 #   evaluate_model(model, test_pairs, test_pair_labels)  # 这里传递的是 test_pair_labels，而不是 test_labels


if __name__ == "__main__":
    main()
