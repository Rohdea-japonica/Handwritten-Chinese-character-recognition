## 模型介绍
本模型使用的是AlexNet，即五层卷积层，三层全连接层的结构。具体可查看"OCR_Model.py"文件。

---

## 文件结构
\-- data 数据集文件夹  
\-- OCR_Project 源文件文件夹  
&nbsp;&nbsp;&nbsp;&nbsp;\-- __init__.py 启动文件  
&nbsp;&nbsp;&nbsp;&nbsp;\-- MyDataset.py 数据加载文件  
&nbsp;&nbsp;&nbsp;&nbsp;\-- OCR_Model.py 模型文件  
&nbsp;&nbsp;&nbsp;&nbsp;\-- show_image_test.py 单张图片预测文件  
&nbsp;&nbsp;&nbsp;&nbsp;\-- output_csv.py 输出label标签至csv文件  
&nbsp;&nbsp;&nbsp;&nbsp;\-- model.pt 模型参数文件  
\-- README.md 本文件

---

## 训练模型代码运行
将该项目导入至PyCharm中，打开__init__.py文件，该文件为启动文件。  
### 选择模式
点击运行后，会出现选择模式的提示，分为"train"和"dev"两种模式。
### 选择训练轮数
选择模式后可以选择训练轮数。   
在"train"模式下，输入多少就训练几个epoch。若存在model.pt文件就会读取后继续训练，反之参考"模型存储"。  
在"dev"模式下，epoch输入1即可获取当前模型在**测试数据集下**的准确率。  
### 模型存储
当OCR_Project文件夹下不存在model.pt文件时，使用"train"模式会在每轮epoch训练后将模型存储一次。  
**该模型存储的内容包括模型参数(键为"model")，优化器参数(键为"optimizer")，当前训练轮数(键为"epoch")**

---

## show_image_test.py文件  
该文件用于预测单张图片，使用时需单独运行该文件。  
运行后会提示选择图片地址，正确输入地址后会展示所选择的图片，并在控制台中显示其数字标签。

---

## output_csv.py文件  
该文件用于输出label标签至csv文件,使用时需要单独运行该文件。  
需要注意的是，追加写入的csv文件需要位于该文件的同级目录下。  
具体可以看示例视频中该部分的演示。