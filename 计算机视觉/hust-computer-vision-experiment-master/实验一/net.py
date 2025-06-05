"""
@Description :   网络结构
@Author      :   tqychy 
@Time        :   2023/12/05 19:33:42
"""

import torch.nn as nn



class FCNet(nn.Module):
    def __init__(self, input_size, hidden_layers_list, output_size, activate_func=nn.ReLU, dropout=0.5):
        """
        全连接神经网络
        :param input_size: 输入层神经元个数
        :param hidden_layers_list: 隐藏层神经元个数列表
        :param output_size: 输入层神经元个数
        :param activate_func: 激活函数的种类
        :param dropout: dropout 概率
        """
        super().__init__()
        assert len(hidden_layers_list) != 0, "隐藏层层数不能为0"
        self.input_layer = nn.Sequential(
            nn.Linear(input_size, hidden_layers_list[0]),
            nn.BatchNorm1d(hidden_layers_list[0]),
            activate_func(),
            nn.Dropout(dropout)
        )
        self.hidden_layer = nn.Sequential()
        for i in range(len(hidden_layers_list) - 1):
            self.hidden_layer.add_module(f"hidden_layer_linear_{i}", nn.Linear(
                hidden_layers_list[i], hidden_layers_list[1 + i]))
            self.hidden_layer.add_module(
                f"hidden_layer_batchnorm_{i}", nn.BatchNorm1d(hidden_layers_list[1 + i]))
            self.hidden_layer.add_module(
                f"hidden_layer_activate_func_{i}", activate_func())
            self.hidden_layer.add_module(
                f"hidden_layer_dropout_{i}", nn.Dropout(dropout))
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_layers_list[-1], output_size)
        )

    def forward(self, x):
        """
        前向传播
        :param x: 输入矩阵 batch_size * 2
        :returns: 输出矩阵 batch_size * 4
        """
        out = self.input_layer(x)
        out = self.hidden_layer(out)
        out = self.output_layer(out)
        return out
    
    def init_weights(self):
        """
        权值初始化
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight)
                nn.init.zeros_(module.bias)


if __name__ == "__main__":
    net = FCNet(2, [50], 4, nn.LeakyReLU)
    print(net)
