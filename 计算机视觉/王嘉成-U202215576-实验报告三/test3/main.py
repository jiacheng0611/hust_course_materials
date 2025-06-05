import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# 检查 CUDA 可用性
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")

# 定义简单 CNN 模型
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(64 * 14 * 14, 10)  # Adjust for MNIST 28x28 input

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        self.last_conv_output = x  # Save output of last conv layer
        x = x.view(-1, 64 * 14 * 14)  # Adjusted for correct feature map size
        x = self.fc(x)
        return x

# 获取最后一层卷积层的输出
def get_average_activation(model, test_loader):
    model.eval()
    activations = []

    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)
            _ = model(images)  # 通过网络，计算特征图
            feature_maps = model.last_conv_output  # 获取最后一层卷积输出

            # 计算每个特征图的平均激活值
            mean_activation = feature_maps.mean(dim=(0, 2, 3))  # 对每个特征图求平均
            activations.append(mean_activation.cpu().numpy())

    # 使用 numpy.array 来合并数据，然后再转为 tensor
    activations = torch.tensor(np.array(activations)).mean(dim=0)  # 平均所有批次的激活值
    return activations


def prune(model, K, activations):
    # 获取 conv1 权重和偏置
    conv1_weights = model.conv1.weight.data
    conv1_biases = model.conv1.bias.data
    # 将 activations 按平均激活值排序，获取索引
    sorted_indices = torch.argsort(activations)
    # 获取需要剪枝的通道索引
    prune_indices = sorted_indices[:K]
    # 确保索引在有效范围内
    prune_indices = prune_indices[prune_indices < conv1_weights.size(0)]  # 移除超出范围的索引
    # 对 conv1 卷积核进行剪枝
    for idx in prune_indices:
        conv1_weights[idx] = 0  # 剪掉对应特征图的卷积核
        conv1_biases[idx] = 0  # 对应的偏置也剪掉

    # 更新模型中的卷积层权重
    model.conv1.weight.data = conv1_weights
    model.conv1.bias.data = conv1_biases

    # print(f"Pruned {K} feature maps.")

# 绘制特征图并保存
def plot_feature_maps(feature_maps, rows, cols, save_path=None):
    fig, axes = plt.subplots(rows, cols, figsize=(12, 8))
    for i, ax in enumerate(axes.flat):
        if i < feature_maps.shape[0]:
            ax.imshow(feature_maps[i], cmap='gray')
            ax.axis('off')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)  # 保存图像
    plt.show()

def test_accuracy(model, test_loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy


# 画出剪枝量 K 和准确率之间的折线图并保存
def plot_pruning_accuracy(K_values, accuracies, save_path="pruning_accuracy.png"):
    plt.plot(K_values, accuracies, marker='o', color='b', linestyle='-', markersize=6)
    plt.title("Pruning vs Accuracy")
    plt.xlabel("Number of Pruned Feature Maps (K)")
    plt.ylabel("Test Accuracy (%)")
    plt.grid(True)

    # 保存图像
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
    plt.show()


# 数据路径
data_root = "C:/zhuomianruanjian/python_projects/test3/data"

# 数据加载
train_dataset = datasets.MNIST(
    root=data_root,
    train=True,
    transform=transforms.ToTensor(),
    download=True
)
test_dataset = datasets.MNIST(
    root=data_root,
    train=False,
    transform=transforms.ToTensor(),
    download=True
)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# # 初始化模型并转移到设备上
# model = SimpleCNN().to(device)
#
# # 定义损失函数和优化器
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#
# # 训练模型
# num_epochs = 5
# for epoch in range(num_epochs):
#     model.train()
#     running_loss = 0.0
#     for images, labels in train_loader:
#         # 数据转移到 GPU
#         images, labels = images.to(device), labels.to(device)
#
#         # 前向传播
#         outputs = model(images)
#         loss = criterion(outputs, labels)
#
#         # 反向传播
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#         running_loss += loss.item()
#     print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")
#
# # 保存模型
# torch.save(model.state_dict(), 'model.pth')

# 加载模型
model = SimpleCNN().to(device)  # 重新实例化模型
# 加载模型，只加载模型的权重
model.load_state_dict(torch.load('model.pth', weights_only=True))  # 加载训练好的模型
model.eval()  # 切换为评估模式

correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")

# 用于累加所有特征图
sum_feature_maps = None
total_samples = 0

# 遍历测试数据集
for images, _ in test_loader:
    images = images.to(device)
    _ = model(images)  # 前向传播，计算特征图
    feature_maps = model.last_conv_output  # 获取最后一层卷积输出

    # 累加特征图
    feature_maps = feature_maps.mean(dim=0).cpu().detach().numpy()  # 平均批次维度
    if sum_feature_maps is None:
        sum_feature_maps = feature_maps
    else:
        sum_feature_maps += feature_maps
    total_samples += 1


activations = get_average_activation(model, test_loader)

# 求平均特征图
average_feature_maps = sum_feature_maps / total_samples  # M*N*P
print(f"Average feature maps shape: {average_feature_maps.shape}")

# 保存特征图（例如，6行10列）
plot_feature_maps(average_feature_maps, rows=8, cols=8, save_path='feature_maps.png')

# 剪枝并评估
K_values = range(1, activations.shape[0])  # K 从 1 到 P-1
accuracies = []

for K in K_values:
    prune(model, K, activations)  # 剪枝 K 个特征图
    accuracy = test_accuracy(model, test_loader)  # 测试准确率
    accuracies.append(accuracy)
    print(f"Pruning {K} feature maps, Accuracy: {accuracy:.2f}%")


# 画图并保存
plot_pruning_accuracy(K_values, accuracies, save_path="pruning_accuracy.png")

# 显示测试集上的一个样例
# example_image, example_label = test_dataset[0]
# plt.imshow(example_image.squeeze(), cmap="gray")
# plt.title(f"Label: {example_label}")
# plt.savefig('example_image.png')  # 保存图像
# plt.show()
