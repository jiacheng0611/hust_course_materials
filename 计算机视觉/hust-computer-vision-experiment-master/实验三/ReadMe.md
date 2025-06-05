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
│   ...
│      
├─MNIST
│  │  t10k-images.idx3-ubyte
│  │  t10k-labels.idx1-ubyte
│  │  train-images.idx3-ubyte
│  │  train-labels.idx1-ubyte
│  │  train0.txt
│  │  train1.txt
│  │  valid0.txt
│  │  valid1.txt
│  │  
│  ├─train
│  │      0.jpg
│  │      1.jpg
│  │      ...
│  │      5999.jpg
│  │      
│  └─valid
│          0.jpg
│          1.jpg
│          ...
│          999.jpg
│          
└─params
   ├─20231218_142134
   │  │  test_log.txt
   │  │  train_log.txt
   │  │  
   │  └─ResNet18checkpoints
   │          best.pth
   │          Epoch0.pth
   │          ...
   │          Epoch29.pth
   │          
   └─20231219_025513
       │  test_log.txt
       │  train_log.txt
       │  
       └─LeNet_RBcheckpoints
               best.pth
               Epoch0.pth
               ...
               Epoch29.pth
```

其中，*dataset.csv* 是原始数据集。*dataset.py* 中含有数据预处理的相关函数。*net.py* 含有模型架构的定义。
*ReadMe.md* 中介绍了如何对模型进行训练和测试。*test.py* 是进行模型测试的源代码文件。*train.py*是进行模型训练的源代码文件。
*utils.py* 中含有一些工具函数。各函数和类的源代码均含有详细的注释，方便使用者了解其完成的工作。

*MNIST* 目录存放数据集相关信息。*images* 目录存放所有绘制的图像。*params* 目录存放模型的参数。当在源代码中设置存储参数选项为 True 并运行一次 *train.py* 时，
*params* 目录中会增加一个以当前时间为名称的子目录。子目录内存放着每一轮训练结束后的模型参数以及效果最佳的模型参数。

# 超参数修改
如果要修改超参数。请修改 *utils.py* 文件中的 params 字典中的内容。默认的超参数取值如下。
```python
# LeNet_RB 的默认参数
params = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",  # 设备
    "random_seed": 1024,  # 随机数种子
    "net": LeNet_RB,

    "dataset_thres": 0.5, # 标签为 0 的数据的占比

    "batch_size": 10000,
    "epoches": 30,
    "lr": 0.005,
    "is_continue": False,  # 是否从断点开始继续训练

    "param_path": "./params/20231219_025513/LeNet_RBcheckpoints/best.pth"  # 保存参数的文件的路径
}
# ResNet18 的默认参数
params = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",  # 设备
    "random_seed": 1024,  # 随机数种子
    "net": ResNet18,

    "dataset_thres": 0.5, # 标签为 0 的数据的占比

    "batch_size": 4000,
    "epoches": 30,
    "lr": 0.01,
    "is_continue": False,  # 是否从断点开始继续训练

    "param_path": "./params/20231218_142134/ResNet18checkpoints/best.pth"  # 保存参数的文件的路径
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