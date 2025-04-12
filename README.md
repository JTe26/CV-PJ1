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
   
   运行`main.py`会有一个生成文件`weight.txt`，若有需要可以更改为绝对路径来妥善存放。（代码位于line 260）



## 代码简介

### NeuralNetwork类

这个类实现了三层神经网络的基本架构，其包括下面几个部分：
- `activation`与`activation_derivative`两个函数实现了ReLU与Sigmoid函数及它们对应的梯度。如果需要使用其他激活函数，需要在这两个函数下补充函数本身及其梯度。
- `forward`实现了前向传播过程。
- `backward`实现了反向传播过程。
- `compute_loss`实现了交叉熵损失的计算。


### Trainer类

这个类实现了训练过程，应用上面`NeuralNetwork`的函数进行模型参数的更新。

### hyperparameter_search函数

这个函数实现了超参数搜索过程。在这里可以修改默认参数值与超参数搜索域。参数说明如下：

- `hidden_size`：隐藏层维度，默认搜索域为128、256、512。（line 228）
- `lr`：初始学习率，默认搜索域为0.001、0.0005、0.0001。（line 229）
- `reg_lambda`：正则化参数，默认搜索域为0.01、0.001、0.0001（line 230）
- `lr_decay`：学习率衰减参数，默认为0.95（即每次学习率衰减触发时学习率会减少5%）。（line 234）
- `decay_interval`：学习率衰减频率，默认为10（即每10个epoch触发一次学习率衰减）。（line 234）
- `num_epochs`：每组超参数训练的epoch数量，默认为100。（line 235）
- `batch_size`：batch的大小，默认为64。（line 235）

### main函数

main函数主要是计算最佳模型在测试集上的正确率，并画出对应的loss曲线以及accuracy曲线。

   


   
