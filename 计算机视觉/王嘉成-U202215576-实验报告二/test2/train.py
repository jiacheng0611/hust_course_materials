import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from model import SimpleCNN
import matplotlib.pyplot as plt


# 计算准确率的函数
def calculate_accuracy(model, data_loader):
    correct = 0
    total = 0
    model.eval()  # 切换到评估模式
    with torch.no_grad():
        for batch_pairs, batch_labels in data_loader:
            # 拆分成两张图片
            x1 = batch_pairs[:, 0]  # 第一个图像
            x2 = batch_pairs[:, 1]  # 第二个图像

            # 合并两个输入为一个张量
            batch_input = torch.stack([x1, x2], dim=1)  # 形状 (batch_size, 2, 28, 28)

            # 前向传播：将合并后的输入传入模型
            outputs = model(batch_input)  # 传递合并后的输入

            # 将输出进行二分类转换，输出 0 或 1
            predicted = (outputs > 0.5).float()

            # 统计预测正确的数量
            correct += (predicted.squeeze() == batch_labels).sum().item()
            total += batch_labels.size(0)

    # 返回准确率
    return correct / total


# 训练模型
# 训练模型
def train_model(model, train_pairs, train_labels, test_pairs, test_labels, num_epochs=10, batch_size=64, lr=0.001):
    criterion = nn.BCELoss()  # 二分类交叉熵损失
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 将数据转换为 PyTorch 张量
    train_pairs = torch.tensor(train_pairs).float()  # (num_pairs, 2, 28, 28)
    train_labels = torch.tensor(train_labels, dtype=torch.float32)
    test_pairs = torch.tensor(test_pairs).float()
    test_labels = torch.tensor(test_labels, dtype=torch.float32)

    # 创建数据加载器
    train_dataset = TensorDataset(train_pairs, train_labels)
    test_dataset = TensorDataset(test_pairs, test_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 损失和准确率记录
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []

    # 训练过程
    for epoch in range(num_epochs):
        model.train()  # 切换到训练模式
        epoch_train_loss = 0.0
        for batch_pairs, batch_labels in train_loader:
            optimizer.zero_grad()

            # 拆分成两张图片
            x1 = batch_pairs[:, 0]  # 第一个图像
            x2 = batch_pairs[:, 1]  # 第二个图像

            # 合并两个输入为一个张量 [batch_size, 2, 28, 28]
            batch_input = torch.stack([x1, x2], dim=1)  # 形状 (batch_size, 2, 28, 28)

            # 前向传播：将合并后的输入传入模型
            outputs = model(batch_input)  # 传递合并后的输入

            # 计算损失并反向传播
            loss = criterion(outputs.squeeze(), batch_labels)
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数

            epoch_train_loss += loss.item()

        # 记录每个 epoch 的训练损失
        train_losses.append(epoch_train_loss / len(train_loader))

        # 计算训练集准确率
        train_accuracy = calculate_accuracy(model, train_loader)
        train_accuracies.append(train_accuracy)

        # 测试集损失和准确率
        test_loss = 0.0
        model.eval()  # 切换到评估模式
        with torch.no_grad():
            for batch_pairs, batch_labels in test_loader:
                # 拆分成两张图片
                x1 = batch_pairs[:, 0]  # 第一个图像
                x2 = batch_pairs[:, 1]  # 第二个图像

                # 合并两个输入为一个张量
                batch_input = torch.stack([x1, x2], dim=1)  # 形状 (batch_size, 2, 28, 28)

                # 前向传播：将合并后的输入传入模型
                outputs = model(batch_input)  # 传递合并后的输入
                loss = criterion(outputs.squeeze(), batch_labels)
                test_loss += loss.item()

        # 记录每个 epoch 的测试损失
        test_losses.append(test_loss / len(test_loader))

        # 计算测试集准确率
        test_accuracy = calculate_accuracy(model, test_loader)
        test_accuracies.append(test_accuracy)

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_losses[-1]:.4f}, Test Loss: {test_losses[-1]:.4f}")
        print(f"Train Accuracy: {train_accuracy * 100:.2f}%, Test Accuracy: {test_accuracy * 100:.2f}%\n")

    # 绘制损失曲线并保存
    plt.figure(figsize=(10, 5))
    plt.plot(range(num_epochs), train_losses, label='Train Loss')
    plt.plot(range(num_epochs), test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss vs. Epochs')
    plt.savefig('loss_vs_epochs.png')  # 保存为 .png 文件
    plt.show()  # 如果你依然想查看图形

    # 绘制准确率曲线并保存
    plt.figure(figsize=(10, 5))
    plt.plot(range(num_epochs), train_accuracies, label='Train Accuracy')
    plt.plot(range(num_epochs), test_accuracies, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy vs. Epochs')
    plt.savefig('accuracy_vs_epochs.png')  # 保存为 .png 文件
    plt.show()  # 如果你依然想查看图形

    # 返回最终训练和测试准确率
    return train_accuracies[-1], test_accuracies[-1]
