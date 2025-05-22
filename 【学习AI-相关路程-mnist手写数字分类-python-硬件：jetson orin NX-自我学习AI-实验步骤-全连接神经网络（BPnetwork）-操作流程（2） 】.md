
@[TOC](【学习AI-相关路程-mnist手写数字分类-python-硬件：jetson orin NX-自我学习AI-实验步骤-BP-遇到问题（2） 】)
# 1、前言
前一篇，我们铺垫了很多基础是知识，这一篇希望尽量少说的概念，直接说操作方式，先操练起来，然后后面找补概念。

相关铺垫知识可以看，前一篇。

[【学习AI-相关路程-mnist手写数字分类-python-硬件：jetson orin NX-自我学习AI-基础知识铺垫-遇到问题（1） 】](https://blog.csdn.net/qq_22146161/article/details/142551662?spm=1001.2014.3001.5501)




# 2、环境jetson orin NX确认
我们使用 jetson orin nx，虽然英伟达有工具帮我们做了很多事情，到每次做前，还是最好确认下，自己作为新手，至少大部分时间，都在解决环境问题。

要在 **Jetson Orin NX** 上运行 **MNIST 手写数字识别程序**，需要确保相关的开发环境和库都已正确安装和配置。以下是确认 Jetson 环境的方法：

### （1）JetPack 版本&检查 CUDA 版本&检查 cuDNN 版本
这几种，其实使用相关工具解决，也是前期需要解决的，但也需要检查，当然，装好了，就不用那么一次又一次检查，界面工具的话如下：NVIDIA SDK MANAGER。

[【学习AI-相关路程-工具使用-NVIDIA SDK MANAGER==NVIDIA-jetson刷机工具安装使用 】](https://blog.csdn.net/qq_22146161/article/details/138345625?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522A823F583-0EEB-42DE-A261-C7614AC94F72%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fblog.%2522%257D&request_id=A823F583-0EEB-42DE-A261-C7614AC94F72&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~blog~first_rank_ecpm_v1~rank_v31_ecpm-1-138345625-null-null.nonecase&utm_term=%20%E5%B7%A5%E5%85%B7%20%E5%88%B7%E6%9C%BA&spm=1018.2226.3001.4450)


1. **检查 JetPack 版本**：

   JetPack 包含了 CUDA、cuDNN、TensorRT 等必要组件。

   ```bash
   dpkg -l | grep jetpack
   ```

   或者查看版本信息：

   ```bash
   sudo apt-cache show nvidia-jetpack
   ```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/14cb28f90acd4fa0b10148b491724235.png)

2. **检查 CUDA 版本**：

   确认 CUDA 是否已安装以及版本号。

   ```bash
   nvcc --version
   ```

   或

   ```bash
   cat /usr/local/cuda/version.txt
   ```

确认环境：使用nvcc确认cuda环境


```shell
wjl-linux@ubuntu:~$ nvcc --versin
nvcc fatal   : Unknown option '--versin'
wjl-linux@ubuntu:~$ nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2022 NVIDIA Corporation
Built on Sun_Oct_23_22:16:07_PDT_2022
Cuda compilation tools, release 11.4, V11.4.315
Build cuda_11.4.r11.4/compiler.31964100_0
wjl-linux@ubuntu:~$ dpkg -l | grep libcudnn
ii  libcudnn8                                  8.6.0.166-1+cuda11.4                  arm64        cuDNN runtime libraries
ii  libcudnn8-dev                              8.6.0.166-1+cuda11.4                  arm64        cuDNN development libraries and headers
ii  libcudnn8-samples                          8.6.0.166-1+cuda11.4                  arm64        cuDNN samples
wjl-linux@ubuntu:~$ python3 -c "import torch; print(torch.__version__); import torchvision; print(torchvision.__version__)"
2.1.0a0+41361538.nv23.06
/home/wjl-linux/.local/lib/python3.8/site-packages/torchvision/io/image.py:13: UserWarning: Failed to load image Python extension: 
  warn(f"Failed to load image Python extension: {e}")
0.13.0

```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/a57f408fe70844abb5efbf6c7bbf9537.png)


3. **检查 cuDNN 版本**：

   查看 cuDNN 是否已安装及其版本。

   ```bash
   dpkg -l | grep cudnn
   ```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/092b0b2a52ae4046baa5f49fb3f229ce.png)

   或

   ```bash
   cat /usr/include/cudnn_version.h | grep CUDNN_MAJOR -A 2
   ```





### （2）检查 Python 版本和包 & PyTorch & 检查环境变量




1. **检查 Python 版本和包**：

   - **Python 版本**：

     ```bash
     python3 --version
     ```

   - **已安装的 Python 库**：

     ```bash
     pip3 list
     ```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/ff182e8af4db4aeaa1ff5e4711195278.png)

   确认是否安装了用于深度学习的库，如 TensorFlow、PyTorch 等。

2. **验证深度学习框架是否正常工作**：

   - **TensorFlow**：

     ```bash
     python3 -c "import tensorflow as tf; print(tf.__version__)"
     ```

   - **PyTorch**：

     ```bash
     python3 -c "import torch; print(torch.__version__)"
     ```
     或者
```c
python3 -c "import torch; print(torch.__version__); import torchvision; print(torchvision.__version__)"
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/48365b65c6264cfeaa48d5671295c59d.png)
3. **检查环境变量**：

   确保 CUDA 和 cuDNN 的路径已添加到系统环境变量中。

   ```bash
   echo $PATH
   echo $LD_LIBRARY_PATH
   ```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/5a7a0ba1cdb540c4bdd4467faf86ca8b.png)
### （3）其他：OpenCV &  验证 GPU & 检查系统

1. **检查 OpenCV 是否安装（如果需要处理图像）**：

   ```bash
   python3 -c "import cv2; print(cv2.__version__)"
   ```

2. **验证 GPU 是否正常工作**：

   运行 CUDA 示例程序：

   ```bash
   # 进入 CUDA 示例目录
   cd /usr/local/cuda/samples/1_Utilities/deviceQuery
   sudo make
   ./deviceQuery
   ```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/5f798eacd2154775a127ebf6bcd20e24.png)

   如果显示 GPU 的详细信息，说明 CUDA 环境正常。




3. **检查系统更新**：

    确保所有包都是最新的，以避免兼容性问题。

    ```bash
    sudo apt-get update
    sudo apt-get upgrade
    ```
这个根据自己情况 ，如果需要就更新，暂时不需要也可以不做 。

4. **检查 TensorRT 版本**：

   ```bash
   dpkg -l | grep tensorrt
   ```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/b808859f68514f8fa2f8c85cd550432a.png)


# 3、下载数据集
## （1）下载地址：
虽然我们只能直接访问官网，
链接：[https://yann.lecun.com/exdb/mnist/](https://yann.lecun.com/exdb/mnist/)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/2b5f7830f3b0418e987479ac105cf73b.png)

但是却无法直接下载，查资料说是需要用代码下载的，不是很会，于是在其他网站上找了资源。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/5e8ca66a984141828a3834b04383f773.png)



## （2）上网找包的资源
如下图所示，就是包里的内容了
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/af45e28badf74cc9a7c8a5b07aa413ef.png)
官网也以及解释了，每个包里有啥，就不在解释了，只不过是英文的。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/b94528477b3d4fe1a542bc3dd48d757b.png)

## （3）输出图片 演示
我们先不着急，编写代码直接训练模型，可以使用python代码试试，看看能不能包的图片

(1)开始的时候，我实在网上找到资料显示的图片，如下
参考链接：[https://blog.csdn.net/tony_vip/article/details/118735261](https://blog.csdn.net/tony_vip/article/details/118735261)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/dc35958f27984dcdaae39806a7cc0ea3.png)

（2）我仿照链接的说明使用如下代码

```python
import os
import struct
import numpy as np
 
# 读取标签数据集
with open('./train-labels.idx1-ubyte', 'rb') as lbpath:
    labels_magic, labels_num = struct.unpack('>II', lbpath.read(8))
    labels = np.fromfile(lbpath, dtype=np.uint8)
 
# 读取图片数据集
with open('./train-images.idx3-ubyte', 'rb') as imgpath:
    images_magic, images_num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
    images = np.fromfile(imgpath, dtype=np.uint8).reshape(images_num, rows * cols) 
 
# 打印数据信息
print('labels_magic is {} \n'.format(labels_magic),
      'labels_num is {} \n'.format(labels_num),
      'labels is {} \n'.format(labels))
 
print('images_magic is {} \n'.format(images_magic),
      'images_num is {} \n'.format(images_num),
      'rows is {} \n'.format(rows),
      'cols is {} \n'.format(cols),
      'images is {} \n'.format(images))
 
 
# 测试取出一张图片和对应标签
import matplotlib.pyplot as plt
 
choose_num = 1 # 指定一个编号，你可以修改这里
label = labels[choose_num]
image = images[choose_num].reshape(28,28)
 
plt.imshow(image)
plt.title('the label is : {}'.format(label))
plt.show()
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/6c0792a34e994583ba43918cef0e4d1e.png)



（3）但是我个人想知道 图片上每个点，因为是灰白的，只有一个通道，也就是一个点的颜色变化是0~255.，自己之前做过智能车，和那个很像，所以想看看那样的图。

那么可以使用如下代码。
```c
import os
import struct
import numpy as np

# 读取标签数据集
with open('./train-labels.idx1-ubyte', 'rb') as lbpath:
    labels_magic, labels_num = struct.unpack('>II', lbpath.read(8))
    labels = np.fromfile(lbpath, dtype=np.uint8)

# 读取图片数据集
with open('./train-images.idx3-ubyte', 'rb') as imgpath:
    images_magic, images_num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
    images = np.fromfile(imgpath, dtype=np.uint8).reshape(images_num, rows * cols) 

# 打印数据信息
print('labels_magic is {} \n'.format(labels_magic),
      'labels_num is {} \n'.format(labels_num),
      'labels is {} \n'.format(labels))

print('images_magic is {} \n'.format(images_magic),
      'images_num is {} \n'.format(images_num),
      'rows is {} \n'.format(rows),
      'cols is {} \n'.format(cols),
      'images shape is {} \n'.format(images.shape))

# 测试取出一张图片和对应标签
choose_num = 1  # 指定一个编号，你可以修改这里
label = labels[choose_num]
image = images[choose_num].reshape(28, 28)

# 在终端中以矩阵形式打印图片像素值（对齐格式）
print('The label is: {}'.format(label))
print('Image pixel values:')
for row in image.astype(int):
    formatted_row = ['{:3d}'.format(pixel) for pixel in row]
    print('[{}]'.format(' '.join(formatted_row)))
```
最后效果就是如下所示，我们其实要输入 28x28=784个数，最后输出一个（0~9）之间数的数学问题。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/76a7e88867ee4d9395f8309c7dc5ccef.png)


# 4、 安装必要的库
```shell
# 安装依赖库
sudo apt-get update
sudo apt-get install -y python3-pip libjpeg-dev zlib1g-dev
pip3 install --upgrade pillow
```

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/5cbab871ff9747b9b0699221541534e4.png)
# 5、 编写训练模型代码
## 1、代码解释

以下是训练时用的代码，也有标注

关于更详细解释，网上也很多，可以参考学习。
链接：[https://blog.csdn.net/ssheudjbdndnjd/article/details/137246079](https://blog.csdn.net/ssheudjbdndnjd/article/details/137246079)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/e87614029d9b47288988e959c491754d.png)



```python

import torch
import sys
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# 检查设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 1. 设置数据集的路径
data_dir = "/home/wjl-linux/Desktop/mnist_ai_work/MNIST"  # 替换为您的实际路径

# 2. 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # 使用MNIST的均值和标准差
])

# 3. 加载数据集
trainset = torchvision.datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
testset = torchvision.datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=1000, shuffle=False)

# 可视化部分训练数据
examples = enumerate(trainloader)
batch_idx, (example_data, example_targets) = next(examples)

fig = plt.figure()
for i in range(6):
    plt.subplot(2,3,i+1)
    plt.tight_layout()
    plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
    plt.title(f"标签: {example_targets[i].item()}")
    plt.xticks([])
    plt.yticks([])
plt.show()

# 4. 定义神经网络模型
class BPnetwork(nn.Module):
    def __init__(self):
        super(BPnetwork, self).__init__()
        self.linear1 = nn.Linear(28 * 28, 128)
        self.ReLU1 = nn.ReLU()
        self.linear2 = nn.Linear(128, 64)
        self.ReLU2 = nn.ReLU()
        self.linear3 = nn.Linear(64, 10)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # 将输入数据展平成一维
        x = self.linear1(x)
        x = self.ReLU1(x)
        x = self.linear2(x)
        x = self.ReLU2(x)
        x = self.linear3(x)
        x = self.log_softmax(x)
        return x

# 5. 创建模型、损失函数和优化器
model = BPnetwork().to(device)
criterion = nn.NLLLoss()  # 负对数似然损失函数
optimizer = optim.Adam(model.parameters(), lr=0.001)  # 使用Adam优化器

# 6. 模型训练
epochs = 10
for epoch in range(epochs):
    model.train()  # 设置模型为训练模式
    running_loss = 0.0
    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()  # 清空梯度
        output = model(images)  # 前向传播
        loss = criterion(output, labels)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新权重

        running_loss += loss.item()
    average_loss = running_loss / len(trainloader)
    print(f"Epoch {epoch+1}, Loss: {average_loss:.4f}")

    # 在每个epoch结束后，评估模型在测试集上的性能
    model.eval()  # 设置模型为评估模式
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            _, predicted = torch.max(output, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f"测试准确率: {accuracy:.2f}%")

# 7. 保存模型
torch.save(model.state_dict(), 'mnist_bpnetwork.pth')
print("模型已保存为 mnist_bpnetwork.pth")

# 8. 可视化部分测试结果
model.eval()
examples = enumerate(testloader)
batch_idx, (example_data, example_targets) = next(examples)
example_data, example_targets = example_data.to(device), example_targets.to(device)

with torch.no_grad():
    output = model(example_data)
fig = plt.figure()
for i in range(6):
    plt.subplot(2,3,i+1)
    plt.tight_layout()
    plt.imshow(example_data[i].cpu()[0], cmap='gray', interpolation='none')
    pred = output.data.max(1, keepdim=True)[1][i].item()
    plt.title(f"预测: {pred}")
    plt.xticks([])
    plt.yticks([])
plt.show()
```

如下是，训练结果，有全连接的模式（BP），关于解释请看前一篇。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/b008549ef6a34737af084e2979509de9.png)


## 2、过程记录
在训练前，会看一下标签与图片对应情况。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/6bdc1881158247bfaa2ace927715c9b2.png)
以下是shell脚本运行记录。
```shell
new_python_demo_minst_model1.py:36: UserWarning: Glyph 26631 (\N{CJK UNIFIED IDEOGRAPH-6807}) missing from current font.
  plt.tight_layout()
new_python_demo_minst_model1.py:36: UserWarning: Glyph 31614 (\N{CJK UNIFIED IDEOGRAPH-7B7E}) missing from current font.
  plt.tight_layout()
new_python_demo_minst_model1.py:41: UserWarning: Glyph 26631 (\N{CJK UNIFIED IDEOGRAPH-6807}) missing from current font.
  plt.show()
new_python_demo_minst_model1.py:41: UserWarning: Glyph 31614 (\N{CJK UNIFIED IDEOGRAPH-7B7E}) missing from current font.
  plt.show()
/usr/lib/python3/dist-packages/gi/overrides/Gio.py:44: UserWarning: Glyph 26631 (\N{CJK UNIFIED IDEOGRAPH-6807}) missing from current font.
  return Gio.Application.run(self, *args, **kwargs)
/usr/lib/python3/dist-packages/gi/overrides/Gio.py:44: UserWarning: Glyph 31614 (\N{CJK UNIFIED IDEOGRAPH-7B7E}) missing from current font.
  return Gio.Application.run(self, *args, **kwargs)
Epoch 1, Loss: 0.2687
测试准确率: 96.02%
Epoch 2, Loss: 0.1104
测试准确率: 97.02%
Epoch 3, Loss: 0.0790
测试准确率: 97.39%
Epoch 4, Loss: 0.0593
测试准确率: 97.45%
Epoch 5, Loss: 0.0494
测试准确率: 97.80%
Epoch 6, Loss: 0.0400
测试准确率: 97.37%
Epoch 7, Loss: 0.0339
测试准确率: 97.98%
Epoch 8, Loss: 0.0298
测试准确率: 97.59%
Epoch 9, Loss: 0.0240
测试准确率: 98.00%
Epoch 10, Loss: 0.0238
测试准确率: 97.85%
模型已保存为 mnist_bpnetwork.pth
new_python_demo_minst_model1.py:115: UserWarning: Glyph 39044 (\N{CJK UNIFIED IDEOGRAPH-9884}) missing from current font.
  plt.tight_layout()
new_python_demo_minst_model1.py:115: UserWarning: Glyph 27979 (\N{CJK UNIFIED IDEOGRAPH-6D4B}) missing from current font.
  plt.tight_layout()
new_python_demo_minst_model1.py:121: UserWarning: Glyph 39044 (\N{CJK UNIFIED IDEOGRAPH-9884}) missing from current font.
  plt.show()
new_python_demo_minst_model1.py:121: UserWarning: Glyph 27979 (\N{CJK UNIFIED IDEOGRAPH-6D4B}) missing from current font.
  plt.show()
/usr/lib/python3/dist-packages/gi/overrides/Gio.py:44: UserWarning: Glyph 39044 (\N{CJK UNIFIED IDEOGRAPH-9884}) missing from current font.
  return Gio.Application.run(self, *args, **kwargs)
/usr/lib/python3/dist-packages/gi/overrides/Gio.py:44: UserWarning: Glyph 27979 (\N{CJK UNIFIED IDEOGRAPH-6D4B}) missing from current font.
  return Gio.Application.run(self, *args, **kwargs)
^Awjl-linux@ubuntu:~/Desktop/minist_ai_work$ 

```
最后我们就会得到一个包，这个后续可以用来使用的了。

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/83d3cd12a7d14db0aa5fd5b146c47c6c.png)


# 6、尝试使用
### 1 初步测试 
之后就是，使用python去调用训练好的包，为此我准备了很多图片，这是其中之一，

但是恰巧是3，而且识别是3，我以为识别很准确呢，结果尝试了几个，发现都是3，第一次的3只是运气而已。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/db21c9b7f8084a1fb8adf95405084c69.png)
后来我又尝试了其他图片，有些图片确实可以的，或者说在特定情况28x28的那种，在网上看资料，也看到有人说过，并且我实际调试时，也遇到了。
链接：[https://blog.csdn.net/Ps_hello/article/details/134169021](https://blog.csdn.net/Ps_hello/article/details/134169021)

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/f077bffefdae48c4b29ca9407c44877c.png)


## 2、代码全部
废话不多说，直接上自己尝试使用的代码。其中有两个：

（1）一种是不转换图片，直接用28x28进行检测的，此种方式，需要你的图片也是28x28的。

```python
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 定义模型
class BPnetwork(nn.Module):
    def __init__(self):
        super(BPnetwork, self).__init__()
        self.linear1 = nn.Linear(28 * 28, 128)
        self.ReLU1 = nn.ReLU()
        self.linear2 = nn.Linear(128, 64)
        self.ReLU2 = nn.ReLU()
        self.linear3 = nn.Linear(64, 10)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.linear1(x)
        x = self.ReLU1(x)
        x = self.linear2(x)
        x = self.ReLU2(x)
        x = self.linear3(x)
        x = self.log_softmax(x)
        return x

# 加载模型
model = BPnetwork().to(device)
model.load_state_dict(torch.load('mnist_bpnetwork.pth', map_location=device))
model.eval()

# 定义预处理
transform = transforms.Compose([
    transforms.Grayscale(),  # 如果图片已是灰度图，可省略
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 读取和预处理图片
image_path = '/home/wjl-linux/Desktop/minist_ai_work/test_image2/test_iamge_ai/testNum10.jpg'  # 替换为您的图片路径
image = Image.open(image_path)

# 如果需要，转换为灰度图
if image.mode != 'L':
    image = image.convert('L')

# 根据需要，反转颜色
image = ImageOps.invert(image)

# 应用预处理
image_tensor = transform(image).unsqueeze(0).to(device)

# 可视化预处理后的图片
plt.imshow(image_tensor.cpu().squeeze(), cmap='gray')
plt.title('预处理后的图片')
plt.axis('off')
plt.show()

# 进行预测
with torch.no_grad():
    output = model(image_tensor)
    pred = output.argmax(dim=1, keepdim=True)
    print(f"模型预测的数字是: {pred.item()}")
```


（2）一种是可以对图片进行转换的，但是从自己实际测试来看，效果其实不好。

```python
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 定义模型（保持不变）
class BPnetwork(nn.Module):
    def __init__(self):
        super(BPnetwork, self).__init__()
        self.linear1 = nn.Linear(28 * 28, 128)
        self.ReLU1 = nn.ReLU()
        self.linear2 = nn.Linear(128, 64)
        self.ReLU2 = nn.ReLU()
        self.linear3 = nn.Linear(64, 10)
        self.log_softmax = nn.LogSoftmax(dim=1)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.linear1(x)
        x = self.ReLU1(x)
        x = self.linear2(x)
        x = self.ReLU2(x)
        x = self.linear3(x)
        x = self.log_softmax(x)
        return x

# 加载模型
model = BPnetwork().to(device)
model.load_state_dict(torch.load('mnist_bpnetwork.pth', map_location=device))
model.eval()

# 定义预处理（添加调整尺寸的步骤）
transform = transforms.Compose([
    transforms.Resize((28, 28)),  # 调整图片尺寸为28x28
    transforms.Grayscale(),       # 转为灰度图像（如果需要）
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 读取和预处理图片
image_path = '/home/wjl-linux/Desktop/minist_ai_work/test_image2/test_iamge_ai/test_num113.jpg'  # 替换为您的图片路径
image = Image.open(image_path)

# 如果需要，转换为灰度图
if image.mode != 'L':
    image = image.convert('L')

# 根据需要，反转颜色
image = ImageOps.invert(image)

# 应用预处理
image_tensor = transform(image).unsqueeze(0).to(device)

# 可视化预处理后的图片
plt.imshow(image_tensor.cpu().squeeze(), cmap='gray')
plt.title('预处理后的图片')
plt.axis('off')
plt.show()

# 进行预测
with torch.no_grad():
    output = model(image_tensor)
    pred = output.argmax(dim=1, keepdim=True)
    print(f"模型预测的数字是: {pred.item()}")
```

### 2测试其他图片
以下是自己收集到图片，进行测试情况。
#### （1）第一组图片：使用28x28的，反转颜色
#####  1-num=9
可以看到我实际反转颜色了，还是没有能成功，我们人类的话其实立刻能看到是9.
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/13d71ea1c22b4a89b6c07f98b91cbd07.png)
模型预测的数字是: 3

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/5390d138373b40a7b5aaa7fd27a7f4b2.png)
#####  2-num=5

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/4bd167ff822049809e194a98529e7580.png)
模型预测的数字是: 5
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/7de20e8d7f054081b02836f395acfe5c.png)
##### 3-num=6

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/4f603249916e4eeaa0ce3032a5e94700.png)模型预测的数字是: 6
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/b243082818de41daa1697d0266a3ccef.png)
#####  特别 4-num=1
这里特别准备了一个不需要反转颜色的，但是在计算机上自己带的绘图软件上，无法在黑色上画白色，只能用擦除的方式 写了一个1. 还是识别错了

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/7dd49946382c482fab4444c796888227.png)
模型预测的数字是: 8

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/11216414163b4f48b9bfe773fc49d249.png)

#### （2）第2组图片：使用非28x28的，不反转颜色
因为有失败情况，觉得可能是自己手写的问题，也是直接在训练的数据包，显示的时候，截图，这样其实不用反转颜色，但是截图，就不是28x28,所以我们需要第二个python文件
#####  1-num=3

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/db2fa809591641e3aca9bca78966198a.png)![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/4872aa10be334e8c867795be1ad09258.png)

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/d13c246a8de848f78050b0940bc4a67a.png)模型预测的数字是: 3

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/c37bdf74b04b4c5a9043be1b22d4e651.png)


#####  2-num=2
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/c090c7392f1944baaf00157a8d0eaba5.png)模型预测的数字是: 2
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/b6ce363c2142465dad200ae5cb1b4d51.png)

#### （3）第3组图片：使用自己写在纸上的，手动拍摄
在纸上这种，可能就超过本模型的能力，因为实际照相，可以看到如下图，人虽然能看出来，但是对于网络来说，至少目前这个网络模型来说，还是太复杂，所以基本识别失败了。
#####  1-num=3
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/c896493ecd754f919f594cc00732fbf8.png)
模型预测的数字是: 3
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/64ea029f9c8748679c2e19ca6b760054.png)
#####  2-num=6
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/e3dfe196aa604b06ab11f839e7ea05eb.png)模型预测的数字是: 3
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/ea44d853bc064d4bb9dc66f85e6e18ec.png)
#####  3-num=9
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/317ce12aa5454f038fdfd0228017113f.png)模型预测的数字是: 3
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/458ae440e1ee4b99adb03f897a604046.png)
#####   4-num=3
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/625c3b7edc724fa39d7c70b5364e44cd.png)
模型预测的数字是: 3
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/141d52d3a84b4a849184dada9c4dcfec.png)
#####   5-num=3

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/03436a183aa0460882e25b99d2111466.png)模型预测的数字是: 3
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/b222b004340f4541bfdc7115425ad1ab.png)

#### （4）第4组图片：使用自己写在纸上的，记号笔，手动拍摄
有了上边失败经验以后，以为是在纸上不显眼的原因，所以使用记号笔尝试，其实是一样的。
#####   1-num=9

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/f35785e4be1a40789bf1a7195a399a98.png)
模型预测的数字是: 3
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/5e5fa51b48a24938b6f0949159eb6dfd.png)

#####   2-num=7

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/10c81943b3a247ecbd047359699e294d.png)
模型预测的数字是: 3


![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/a92ae0c0dd9d49d491533861fd99f4de.png)
#### （5）第5组图片：电脑自带绘图软件，非28x28图片 
想到背景复杂这种情况后，想到，那么直接使用绘图软件，绘制非28x28的图片呢，自己感觉效果也很差。
#####   1-num=8
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/8119f7568ffb40fbbf253b51de7bb1b8.png)
模型预测的数字是: 1
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/a646305058144556a30d8afd324eb079.png)
#####    2-num=0

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/1128b9067153481d830b4797751fef86.png)模型预测的数字是: 1

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/18c2b33790674152a6baf35df407f599.png)
#####    3-num=7

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/d37a276890544da9b6efc4753ada920d.png)
模型预测的数字是: 1

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/3d36fe11c0584d10aef388648d50352c.png)
#### （6）第6组图片：电脑自带绘图软件，28x28图片，白底黑字 ,反转颜色
这是最后准备一组照片，比较严格，使用28x28，也大概能摸到规律，只有和训练集相近的，识别率才比较高，当时我们为了学习，失败也是可以接受的。
#####    1-num=1

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/c7d908cf4fbf4f0988fc368edcdf0c23.png)模型预测的数字是: 4
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/aa9edad5564343fcba4a4f17a26c1eb7.png)
#####    2-num=9
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/1322154725d0404d83e5e827f30ea583.png)
模型预测的数字是: 9

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/91d685bda94a4c3e828983ff7fdb3fe3.png)

#####    3-num=2
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/898b28e7fa604a16b47f2af29e9df408.png)模型预测的数字是: 2
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/057e808f3660476e9c368bb658a24aa6.png)

#####    4-num=7

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/2fe5c6d41aba4353b38e32d48ce5261f.png)模型预测的数字是: 7
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/79d66ce29bf84735ae89c7aa1f43d6ef.png)


#####    5-num=7
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/67ce2a311a034366be917982d89e4d91.png)
模型预测的数字是: 7

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/52ea2084696d41c5a01abe40e30238e2.png)
#####   6-num=4

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/54663aecb52b4f23aeacb91367ed632a.png)模型预测的数字是: 3
这里其实也识别错了，这里我想到，其实我们人类看到数字，有些条件默认了，但模型不一定这样认为，我们认为正面对着我们，下就是下，但是如果把数字旋转一下，你能一眼就看出是4么，如下图，所以模型不知道什么是。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/8118fb38b89b4274844a561e66059df4.png)

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/28406fa2542540ee972b1b1280068208.png)

#####    7-num=0
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/118a6aa988e847909a03091febc32d5a.png)模型预测的数字是: 0
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/d05a056b88ff442990ad5a20942ae1dd.png)
#####    8-num=7
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/d58bf1b34e8b4720a152e74cf2b4bdfa.png)

模型预测的数字是: 1



![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/6c8a9bdc5a424100ab93055ada71dca9.png)#####    9-num=2

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/d597dba6b6ca4680bbede410d4b77987.png)
模型预测的数字是: 2
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/ce175207adcd4a5a90178e345090dc37.png)#####   10-num=6
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/a7ca2c5f05e7424b8acf38f00d50205a.png)
模型预测的数字是: 6

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/6247513550604f42b6c27a0b920808d8.png)
# 7、数据集+使用代码
如下，为本次训练使用的代码和训练集，需要的拿走吧。
链接：[https://download.csdn.net/download/qq_22146161/89919700](https://download.csdn.net/download/qq_22146161/89919700)
# 8、综合说明&细节部分
## （1）说明过程中的体验
（1）特定网络模型，在特定情况有效 
简单说，整个模型是基于训练图片的，所以你用以检测的图片，必须相似，这可能是模型的限制或者数量量的限制。

（2）数据驱动，暴力破解，数据为王
第一次我自己想要是现识别屏幕，自己一共收集了大概50展图，就可以简单识别屏幕是开关，虽然屏幕比较负载，需要特定图片也会变多，但从自己只是识别特定屏幕（比如我自己的屏幕）是够用了的。

第二次，也就是这次，识别数字，需要近3W张图片，这个数据使其是呈现指数增长的，但都被计算机采用暴力破解方式，解决了，这很神奇。

第三次，如果是其他项目，更为复杂，所需要的训练数据，算力，等等，是几乎难以想象的。

（3）说不清，道不明感觉
自己在学习的过程中，其实一直使用AI辅助，从自己学习AI的角度来讲，感觉很多地方不懂，没有像硬件那样的，电路逻辑0和1的感觉。

## （2）其他失败的实验
失败的经验也很宝贵，

（1）
```python
import os
import struct
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import matplotlib.pyplot as plt

# 自定义MNIST数据集加载类
class MNISTDataset(Dataset):
    def __init__(self, image_file, label_file, transform=None):
        self.image_file = image_file
        self.label_file = label_file
        self.transform = transform
        self.images, self.labels = self.load_data()

    def load_data(self):
        # 读取标签文件
        with open(self.label_file, 'rb') as lbl_f:
            magic, num = struct.unpack(">II", lbl_f.read(8))
            labels = np.fromfile(lbl_f, dtype=np.uint8)

        # 读取图像文件
        with open(self.image_file, 'rb') as img_f:
            magic, num, rows, cols = struct.unpack(">IIII", img_f.read(16))
            images = np.fromfile(img_f, dtype=np.uint8).reshape(len(labels), 784)

        return images, labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx].reshape(28, 28, 1).astype(np.float32)
        label = int(self.labels[idx])

        if self.transform:
            image = self.transform(image)

        return image, label

# 数据转换：将图像转换为张量，并进行归一化
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 数据集文件路径
data_dir = "/home/wjl-linux/Desktop/minist_ai_work/MNIST"  # 替换为你的实际路径
train_images_path = os.path.join(data_dir, 'train-images.idx3-ubyte')
train_labels_path = os.path.join(data_dir, 'train-labels.idx1-ubyte')
test_images_path = os.path.join(data_dir, 't10k-images.idx3-ubyte')
test_labels_path = os.path.join(data_dir, 't10k-labels.idx1-ubyte')

# 创建数据集实例
train_dataset = MNISTDataset(train_images_path, train_labels_path, transform=transform)
test_dataset = MNISTDataset(test_images_path, test_labels_path, transform=transform)

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 检查数据加载是否正确
images, labels = next(iter(train_loader))
print(f"样本图像尺寸：{images.shape}, 样本标签：{labels[:5]}")

# 显示部分训练集图像
fig, axes = plt.subplots(1, 6, figsize=(12, 4))
for i in range(6):
    ax = axes[i]
    ax.imshow(images[i].numpy().squeeze(), cmap='gray')
    ax.set_title(f'Label: {labels[i].item()}')
plt.show()

# 定义改进的神经网络模型
class ImprovedBPnetwork(nn.Module):
    def __init__(self):
        super(ImprovedBPnetwork, self).__init__()
        self.linear1 = nn.Linear(28 * 28, 256)
        self.ReLU1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)

        self.linear2 = nn.Linear(256, 128)
        self.ReLU2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)

        self.linear3 = nn.Linear(128, 64)
        self.ReLU3 = nn.ReLU()

        self.linear4 = nn.Linear(64, 10)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # 将输入数据展平成一维
        x = self.linear1(x)
        x = self.ReLU1(x)
        x = self.dropout1(x)

        x = self.linear2(x)
        x = self.ReLU2(x)
        x = self.dropout2(x)

        x = self.linear3(x)
        x = self.ReLU3(x)

        x = self.linear4(x)
        x = self.softmax(x)
        return x

# 创建模型实例
model = ImprovedBPnetwork()

# 将模型移到 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 定义损失函数和优化器
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)  # 使用 Adam 优化器

# 训练模型
epochs = 10
for epoch in range(epochs):
    running_loss = 0.0
    model.train()  # 切换到训练模式
    for images, labels in train_loader:
        # 将数据和标签移动到 GPU
        images, labels = images.to(device), labels.to(device)

        # 清除梯度
        optimizer.zero_grad()

        # 前向传播
        output = model(images)

        # 计算损失
        loss = criterion(output, labels)

        # 反向传播
        loss.backward()

        # 更新权重
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader)}")

print("训练完成!")

# 测试模型
correct = 0
total = 0
model.eval()  # 切换到评估模式
with torch.no_grad():  # 禁用梯度计算，提升测试速度
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        output = model(images)
        _, predicted = torch.max(output, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"模型在测试集上的准确率：{100 * correct / total:.2f}%")

# 保存模型
torch.save(model.state_dict(), "mnist_improved_model.pth")
print("模型已保存!")
```

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/69c63e3acb9f491d8444f75910d90389.png)
如下图是训练过程
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/fb437a9421a5462a9c453e6a907756f8.png)![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/61fbb22953e949e79f944a8595345651.png)
以下，其实是前期尝试修改参数，进行重新训练

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/e1366f8dd07d![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/d8fdef15c52d4f719c55ed13949629ed.png)
4bdb861e6629a860896d.png)

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/d5ba1cdf2a2b4a589282102d3283df42.png)


![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/758e3943c219424898e3cb26dc7ba024.png)

```c
wjl-linux@ubuntu:~/Desktop/minist_ai_work$ vim check_torch_and_torchvision.py
wjl-linux@ubuntu:~/Desktop/minist_ai_work$ python3 check_torch_and_torchvision.py 
/home/wjl-linux/.local/lib/python3.8/site-packages/torchvision/io/image.py:13: UserWarning: Failed to load image Python extension: 
  warn(f"Failed to load image Python extension: {e}")
PyTorch 版本: 2.1.0a0+41361538.nv23.06
torchvision 版本: 0.13.0
wjl-linux@ubuntu:~/Desktop/minist_ai_work$ ^C
wjl-linux@ubuntu:~/Desktop/minist_ai_work$ pip uninstall torchvision 
Found existing installation: torchvision 0.13.0
Uninstalling torchvision-0.13.0:
  Would remove:
    /home/wjl-linux/.local/lib/python3.8/site-packages/torchvision-0.13.0.dist-info/*
    /home/wjl-linux/.local/lib/python3.8/site-packages/torchvision/*
  Would not remove (might be manually added):
    /home/wjl-linux/.local/lib/python3.8/site-packages/torchvision/datasets/_six.py
    /home/wjl-linux/.local/lib/python3.8/site-packages/torchvision/datasets/vision_old.py
Proceed (Y/n)? Y
  Successfully uninstalled torchvision-0.13.0
wjl-linux@ubuntu:~/Desktop/minist_ai_work$ python3 -m pip install --upgrade pip
Defaulting to user installation because normal site-packages is not writeable
Requirement already satisfied: pip in /home/wjl-linux/.local/lib/python3.8/site-packages (24.2)
wjl-linux@ubuntu:~/Desktop/minist_ai_work$ pip3 install --upgrade torchvision --extra-index-url https://developer.download.nvidia.com/compute/redist
Defaulting to user installation because normal site-packages is not writeable
Looking in indexes: https://pypi.org/simple, https://developer.download.nvidia.com/compute/redist
Collecting torchvision
  Downloading torchvision-0.19.1-cp38-cp38-manylinux2014_aarch64.whl.metadata (6.0 kB)
Requirement already satisfied: numpy in /home/wjl-linux/.local/lib/python3.8/site-packages (from torchvision) (1.24.4)
Collecting torch==2.4.1 (from torchvision)
  Downloading torch-2.4.1-cp38-cp38-manylinux2014_aarch64.whl.metadata (26 kB)
Requirement already satisfied: pillow!=8.3.*,>=5.3.0 in /home/wjl-linux/.local/lib/python3.8/site-packages (from torchvision) (10.4.0)
Requirement already satisfied: filelock in /home/wjl-linux/.local/lib/python3.8/site-packages (from torch==2.4.1->torchvision) (3.14.0)
Requirement already satisfied: typing-extensions>=4.8.0 in /home/wjl-linux/.local/lib/python3.8/site-packages (from torch==2.4.1->torchvision) (4.11.0)
Requirement already satisfied: sympy in /home/wjl-linux/.local/lib/python3.8/site-packages (from torch==2.4.1->torchvision) (1.12)
Requirement already satisfied: networkx in /home/wjl-linux/.local/lib/python3.8/site-packages (from torch==2.4.1->torchvision) (3.1)
Requirement already satisfied: jinja2 in /home/wjl-linux/.local/lib/python3.8/site-packages (from torch==2.4.1->torchvision) (3.1.4)
Requirement already satisfied: fsspec in /home/wjl-linux/.local/lib/python3.8/site-packages (from torch==2.4.1->torchvision) (2024.3.1)
Requirement already satisfied: MarkupSafe>=2.0 in /home/wjl-linux/.local/lib/python3.8/site-packages (from jinja2->torch==2.4.1->torchvision) (2.1.3)
Requirement already satisfied: mpmath>=0.19 in /home/wjl-linux/.local/lib/python3.8/site-packages (from sympy->torch==2.4.1->torchvision) (1.3.0)
Downloading torchvision-0.19.1-cp38-cp38-manylinux2014_aarch64.whl (14.1 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 14.1/14.1 MB 147.9 kB/s eta 0:00:00
Downloading torch-2.4.1-cp38-cp38-manylinux2014_aarch64.whl (89.7 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 89.7/89.7 MB 100.8 kB/s eta 0:00:00
Installing collected packages: torch, torchvision
  Attempting uninstall: torch
    Found existing installation: torch 2.1.0a0+41361538.nv23.6
    Uninstalling torch-2.1.0a0+41361538.nv23.6:
      Successfully uninstalled torch-2.1.0a0+41361538.nv23.6
Successfully installed torch-2.4.1 torchvision-0.19.1
wjl-linux@ubuntu:~/Desktop/minist_ai_work$ python3 check_torch_and_torchvision.py 
PyTorch 版本: 2.4.1
torchvision 版本: 0.19.1
wjl-linux@ubuntu:~/Desktop/minist_ai_work$ 

```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/c0c4969a76bd4fbd8b481417d72a2992.png)



## （3）细节部分：问题No such file or directory:"xxx"
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/c023bbb9e63a42f8b8db52d4b1d24d94.png)

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/1b90460fc22f42cc91499c51445c408e.png)

## （4）检查 TensorRT 版本
如下为在网上查到，什么是TensorRT，个人调试过程中没有用到。

TensorRT，全称为Tensor RuntimE for Deep Learning Inference，是由英伟达（NVIDIA）开发的一款高效的深度学习推理优化工具。它是一个用于加速机器学习模型（特别是卷积神经网络CNN）部署到嵌入式设备、数据中心或GPU上的库。TensorRT能够将复杂的深度学习模型转换成高性能的低级计算引擎，显著提高模型在生产环境中的运行速度和效率。通过预编译和量化技术，TensorRT减少了计算图解析的时间，并减少了内存占用，使得实时推断成为可能。


## （5）安装正确的torch 和 torchvision
```cpp
wget https://nvidia.box.com/shared/static/fm4bh3vdx3nmfqmmgz2s2pkq6c1cz8d5.whl -O torchvision-0.15.1+nv23.06-cp38-cp38-linux_aarch64.whl
pip install torchvision-0.15.1+nv23.06-cp38-cp38-linux_aarch64.whl
```


![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/6ed73b909acf4a2d8a84209cfbd99f87.png)



# 8、总结

其实在整个叙述过程中，自己还是有很多细节没有提到，但限于篇幅，认为把必要的部分写了下，希望这成为你学习AI路途中的一块砖石。



