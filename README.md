# CV-PJ1

## 项目简介
本实验实现了一个基于全连接神经网络的CIFAR-10图像分类模型，支持以下功能：
- CIFAR-10数据集的加载与预处理
- 带有ReLU/Sigmoid激活函数的三层神经网络
- 学习率衰减、L2正则化、交叉熵损失
- 超参数网格式搜索（隐藏层大小、学习率、正则化强度）
- 可视化loss曲线与accuracy曲线
- 模型权重保存
- 最佳模型的测试集评估

## 依赖项
- Python 3.6+
- numpy
- scikit-learn
- matplotlib
## 数据准备

### 数据集下载与输出文件地址说明
1. **下载CIFAR-10数据集**  
   从官方地址下载Python版本的CIFAR-10数据集：  
   [CIFAR-10 Dataset (Python version)](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz)
   并放置到项目目录下即可，保证能被成功找到。
   ```python
   data_dir = './cifar-10-batches-py'
   ```

2. **输出文件地址**
   运行`main.py`会有一个生成文件`weight.txt`，若有需要可以更改为绝对路径来妥善存放。


   
