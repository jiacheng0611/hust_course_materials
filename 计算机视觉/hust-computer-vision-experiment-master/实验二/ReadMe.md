# 环境配置
本实验所需的第三方库如下：
```bash
matplotlib==3.7.2
numpy==1.23.1
Pillow==10.1.0
scikit_image==0.19.3
scikit_learn==1.0.2
seaborn==0.11.2
torch==1.13.0+cu116
torchvision==0.14.0+cu116
tqdm==4.66.1
```
将上述内容保存为 `requirements.txt` 后使用
```bash
pip install -r requirements.txt
```
命令即可安装所有的第三方库。

# 文件结构
```
│  dataset.py
│  net.py
│  ReadMe.md
│  test.py
│  train.py
│  utils.py
│  
├─images
│    ...
│      
├─MNIST
│  │  t10k-images.idx3-ubyte
│  │  t10k-labels.idx1-ubyte
│  │  train-images.idx3-ubyte
│  │  train-labels.idx1-ubyte
│  │  train.txt
│  │  valid.txt
│  │  
│  ├─train
│  │      0.jpg
│  │      1.jpg
│  │      ...
│  │      59999.jpg
│  │      
│  └─valid
│          0.jpg
│          1.jpg
│          ...
│          9999.jpg
│          
└──params
   └─20231215_000404
       │  test_log.txt
       │  train_log.txt
       │  
       ├─LeNet_RBcheckpoints
       │        best.pth
       │        Epoch0.pth
       │        Epoch1.pth
       │        ...
       │        Epoch99.pth
       │
       │
       └─ResNet18_RBcheckpoints
              best.pth
              Epoch0.pth
              Epoch1.pth
              ...
              Epoch99.pth
```

其中，*dataset.csv* 是原始数据集。*dataset.py* 中含有数据预处理的相关函数。*net.py* 含有模型架构的定义。
*ReadMe.md* 中介绍了如何对模型进行训练和测试。*test.py* 是进行模型测试的源代码文件。*train.py*是进行模型训练的源代码文件。
*utils.py* 中含有一些工具函数。各函数和类的源代码均含有详细的注释，方便使用者了解其完成的工作。

*MNIST* 目录存放数据集相关信息。*images* 目录存放所有绘制的图像。*params* 目录存放模型的参数。当在源代码中设置存储参数选项为 True 并运行一次 *train.py* 时，
*params* 目录中会增加一个以当前时间为名称的子目录。子目录内存放着每一轮训练结束后的模型参数以及效果最佳的模型参数。

# 超参数修改
如果要修改超参数。请修改 *utils.py* 文件中的 params 字典中的内容。默认的超参数取值如下。
```python
params = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",  # 设备
    "random_seed": 1024,  # 随机数种子
    "net": ResNet18,

    "batch_size": 600,
    "epoches": 100,
    "lr": 0.0001,
    "is_continue": False,  # 是否从断点开始继续训练

    "param_path": "./params/20231215_092753/ResNet18checkpoints/best.pth"  # 保存参数的文件的路径
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