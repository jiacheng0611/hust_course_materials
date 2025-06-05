# 环境配置
本实验所需的第三方库如下：
```bash
matplotlib==3.7.2
numpy==1.23.1
opencv_contrib_python_rolling==4.6.0.20220920
opencv_python==4.8.0.76
Pillow==10.2.0
psutil==5.9.1
pytest==7.4.4
scikit_image==0.19.3
scikit_learn==1.0.2
scipy==1.8.0
setuptools==67.0.0
timm==0.9.2
torch==1.13.0+cu116
torchvision==0.14.0+cu116
tqdm==4.66.1
ttach==0.0.3
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
│  ReadMe.md
│  utils.py
│  
├─examples
│  │  torch_alex.pth
│  │  加载方法.txt
│  │  
│  └─data4
│          both.jpg
│          cat.jpg
│          dog.jpg
│          
├─images
│      both.jpg
│      ...
│      dog.jpg
│      
├─model_zoo
│      CAMs.py
│      LIME.py
│          
│          
└─results
   ├─name-GradCAM++-flip-False-aug-False-eigen-False
   │      params.json
   │      test0.png
   │      test1.png
   │      test2.png
   │      
   ├─name-GradCAM++-flip-False-aug-False-eigen-True
   │      params.json
   │      test0.png
   │      test1.png
   │      test2.png
   │─...
   │      
   └─name-XGradCAM-flip-True-aug-True-eigen-True
           params.json
           test0.png
           test1.png
           test2.png
```
其中，*dataset.py* 中含有数据预处理的相关函数。
*ReadMe.md* 中介绍了如何对模型进行训练和测试。*main.py* 是实现可解释性计算的主程序。
*utils.py* 中含有一些工具函数。各函数和类的源代码均含有详细的注释，方便使用者了解其完成的工作。
*images* 目录存放所有绘制的图像。*model_zoo* 目录存放 CAM 和 LIME 的网络结构。
*results* 目录存放了各种超参数下的可视化结果热力图。

# 超参数修改
如果要修改超参数。请修改 *utils.py* 文件中的 params 字典中的内容。默认的超参数取值如下。
```python
params = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",  # 设备
    "random_seed": 1024,  # 随机数种子
    "data_path": "./examples/data4",  # 测试集的路径
    "net_path": "./examples/torch_alex.pth",  # 预训练网络参数的路径

    "cam_type": "ScoreCAM",  # 可解释性算法的名称
    "use_eigen": False, # 是否使用 eigen 平滑
    "use_aug": False, # 是否使用 aug 平滑
    "flip": False  # 绘制另一个标签的热力图
}
```

# 运行可解释性算法
运行
```bash
python main.py
```
命令即可。
该程序会枚举所有超参数的取值并将结果存储在 *results* 中。