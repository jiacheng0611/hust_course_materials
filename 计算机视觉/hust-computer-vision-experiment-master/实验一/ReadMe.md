# 环境配置
本实验所需的第三方库如下：
```bash
matplotlib==3.7.2
matplotlib==3.5.1
numpy==1.23.1
pandas==2.1.4
scikit_learn==1.0.2
scipy==1.8.0
seaborn==0.11.2
torch==1.13.0+cu116
tqdm==4.66.1
```
将上述内容保存为 `requirements.txt` 后使用
```bash
pip install -r requirements.txt
```
命令即可安装所有的第三方库。

# 文件结构
```
│  dataset.csv
│  dataset.py
│  net.py
│  ReadMe.md
│  test.py
│  train.py
│  utils.py
│  
├─images
│      不同的l1_weight.pdf
│      不同的学习率.pdf
│      不同的神经元个数.pdf
│      不同的网络层数.pdf
│      数据个数分布图.pdf
│      数据分布图.pdf
│      数据分布拟合图.pdf
│      数据分布拟合图.png
│      模型的分类结果.pdf
│      模型的分类结果.png
│      训练损失函数及评价指标(batch).pdf
│      训练验证acc.pdf
│      训练验证f1.pdf
│      训练验证损失函数.pdf
│      
└─params
   └─20231207_230630
           best.pth
           Epoch1.pth
 	     ...
           Epoch99.pth
```

其中，*dataset.csv* 是原始数据集。*dataset.py* 中含有数据预处理的相关函数。*net.py* 含有模型架构的定义。
*ReadMe.md* 中介绍了如何对模型进行训练和测试。*test.py* 是进行模型测试的源代码文件。*train.py*是进行模型训练的源代码文件。
*utils.py* 中含有一些工具函数。各函数和类的源代码均含有详细的注释，方便使用者了解其完成的工作。

*images* 目录存放所有绘制的图像。*params* 目录存放模型的参数。当在源代码中设置存储参数选项为 True 并运行一次 *train.py* 时，
*params* 目录中会增加一个以当前时间为名称的子目录。子目录内存放着每一轮训练结束后的模型参数以及效果最佳的模型参数。

# 超参数修改
如果要修改超参数。请修改 *utils.py* 文件中的 params 字典中的内容。默认的超参数取值如下。
```python
params = {
    "device": "cuda" if torch.cuda.is_available() else "cpu", # 设备
    "random_seed": 1024, # 随机数种子
    "valid_ratio": 0.1, # 测试集比例
    "hidden_layers_list": [50], # 隐藏层神经元个数
    "activate_func": nn.LeakyReLU, # 激活函数
    "early_stop": False, # 是否使用早停
    "patience":20, # 早停容忍轮数

    "batch_size": 3600, # 批大小
    "epoches": 200, # 最大训练轮数
    "lr": 1, # 学习率
    "dropout": 0.4, # dropout 概率
    "l1_weight": 0.001, # L1 损失系数
    "l2_weight": 0.0,# L2 损失系数

    "param_path": "./params/20231207_230630/best.pth" # 测试使用的参数
}
```

# 测试
params 字典中的 `param_path` 控制测试时加载的模型参数地址。默认的地址即是实验报告中用于测试的模型参数的地址。
运行
```bash
python test.py
```
命令即可进行测试。

## 训练
运行
```bash
python train.py
```
命令即可进行训练。