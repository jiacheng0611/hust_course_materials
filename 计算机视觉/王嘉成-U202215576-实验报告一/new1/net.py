import torch
import torch.nn as nn

class FCNet(nn.Module):
    def __init__(self, input_size, hidden_layers_list, output_size, activate_func=nn.ReLU, dropout=0.5):
        """
        全连接神经网络
        """
        super().__init__()
        assert len(hidden_layers_list) != 0, "隐藏层层数不能为0"

        # 第一层
        layers = []
        layers.append(nn.Linear(input_size, hidden_layers_list[0]))
        layers.append(nn.BatchNorm1d(hidden_layers_list[0]))
        layers.append(activate_func())
        layers.append(nn.Dropout(dropout))

        # 隐藏层
        for i in range(1, len(hidden_layers_list)):
            layers.append(nn.Linear(hidden_layers_list[i-1], hidden_layers_list[i]))
            layers.append(nn.BatchNorm1d(hidden_layers_list[i]))
            layers.append(activate_func())
            layers.append(nn.Dropout(dropout))

        # 输出层
        layers.append(nn.Linear(hidden_layers_list[-1], output_size))

        # 使用 Sequential 组合所有层
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        """
        前向传播
        """
        return self.model(x)

    def init_weights(self):
        """
        权值初始化
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if isinstance(module, nn.Linear):
                    nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
                    nn.init.zeros_(module.bias)

if __name__ == "__main__":
    net = FCNet(2, [50, 30], 4, nn.LeakyReLU)
    print(net)
