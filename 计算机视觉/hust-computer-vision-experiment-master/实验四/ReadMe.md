# 环境配置
本实验所需的第三方库如下：
```bash
matplotlib==3.7.2
numpy==1.23.1
opencv_contrib_python_rolling==4.6.0.20220920
opencv_python==4.8.0.76
Pillow==10.1.0
torch==1.13.0+cu116
torchvision==0.14.0+cu116
```
将上述内容保存为 `requirements.txt` 后使用
```bash
pip install -r requirements.txt
```
命令即可安装所有的第三方库。

# 文件结构
```
│  dataset.py
│  main.py
│  net.py
│  ReadMe.md
│  utils.py
│  
├─images
│      all.png
│      both.jpg
│      cat.jpg
│      dog.jpg
│      GradCAM-test0.png
│      GradCAM-test1.png
│      GradCAM-test2.png
│      LayerCAM-test0.png
│      LayerCAM-test1.png
│      LayerCAM-test2.png
│      
│      
└─实验四模型和测试图片（PyTorch）
   │  torch_alex.pth
   │  加载方法.txt
   │  
   └─data4
           both.jpg
           cat.jpg
           dog.jpg
```

其中，*dataset.py* 中含有数据预处理的相关函数。*net.py* 含有可解释性算法的定义和实现。
*ReadMe.md* 中介绍了如何对模型进行训练和测试。*main.py* 是实现可解释性计算的主程序。
*utils.py* 中含有一些工具函数。各函数和类的源代码均含有详细的注释，方便使用者了解其完成的工作。
*images* 目录存放所有绘制的图像。

# 超参数修改
如果要修改超参数。请修改 *utils.py* 文件中的 params 字典中的内容。默认的超参数取值如下。
```python
params = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",  # 设备
    "random_seed": 1024,  # 随机数种子
    "data_path": "./实验四模型和测试图片（PyTorch）/data4",  # 测试集的路径
    "net_path": "./实验四模型和测试图片（PyTorch）/torch_alex.pth",  # 预训练网络参数的路径

    "cam_type": "LayerCAM", # 可解释性算法的名称
    "flip": False
}
```

# 运行可解释性算法
运行
```bash
python main.py
```
命令即可。