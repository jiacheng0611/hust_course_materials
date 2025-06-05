import torch

def evaluate_model(model, test_pairs, test_pair_labels):
    model.eval()  # Set the model to evaluation mode
    total = 0
    correct = 0

    # 逐个遍历测试样本
    for i in range(len(test_pairs)):
        x1, x2 = test_pairs[i]
        label = test_pair_labels[i]

        # 转换为 Torch 张量并增加批次和通道维度
        x1 = torch.tensor(x1).unsqueeze(0).unsqueeze(0).float()  # (1, 1, 28, 28)
        x2 = torch.tensor(x2).unsqueeze(0).unsqueeze(0).float()  # (1, 1, 28, 28)
        label = torch.tensor(label).unsqueeze(0).float()  # 转换为单一标签（batch_size=1）

        # 推理
        with torch.no_grad():
            output = model(x1, x2)

        # 计算准确率
        pred = output.squeeze().round()  # 假设输出是一个概率值，使用 .round() 将其变为0或1
        correct += (pred == label).sum().item()
        total += label.size(0)

    accuracy = correct / total
    print(f"Accuracy: {accuracy * 100:.2f}%")


