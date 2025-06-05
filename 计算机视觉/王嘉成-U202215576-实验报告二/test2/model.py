import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # 定义网络层
        self.conv1 = nn.Conv2d(2, 32, kernel_size=3, padding=1)  # 2个输入通道
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 28 * 28, 128)
        self.fc2 = nn.Linear(128, 1)  # 输出一个值（0或1）

    def forward(self, x):
        # 前向传播
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # 展平
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))  # 使用 sigmoid 输出一个值
        return x

