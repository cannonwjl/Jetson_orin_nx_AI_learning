@[TOC](【学习AI-相关路程-工具使用-自我学习-jetson&收集数据-图片采集-训练前准备-基础样例 （4）】)
# 1-前言
有时候，我们想要做某是个事情的时候，其实不知道具体步骤的，常常感到无从下手，这种无力感，确实有些头疼，然后去查资料，除非有人能说得特别明白，要不然，其实也算是半摸索形式，边学边做的方式，还有时，常常质疑自己学到资料对不对，方法步骤对不对。

一般有两种情况，一种是方法步骤比较明确，大家已经趟过无数遍了。而另一种则是他人还未开发的道路，需要人们探索了。

大部分情况，是前人已经探索了，你只是需要依照这个路途走就行了。就像咱们本章，使用一些数据，进行图片更改，为训练模型准备图片。

关联文章，需要了解背景，我们之前安装好了相关工具软件，不知道同学，可以往前找找，相当有前置条件。

[【学习AI-相关路程-工具使用-自我学习-jetson&cuda&pytorch-开发工具尝试-基础样例 （3）】](https://waka-can.blog.csdn.net/article/details/138664922)

# 2-环境说明
本次环境是在jetson orin NX 上，环境信息如下。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/d74de56c00b1443e97ea9a8d617debf9.jpeg)

```shell
wjl-linux@ubuntu:~$ jetson_release
Software part of jetson-stats 4.2.9 - (c) 2024, Raffaello Bonghi
Model: NVIDIA Orin NX Developer Kit - Jetpack 5.1.2 [L4T 35.4.1]
NV Power Mode[3]: 25W
Serial Number: [XXX Show with: jetson_release -s XXX]
Hardware:
 - P-Number: p3767-0000
 - Module: NVIDIA Jetson Orin NX (16GB ram)
Platform:
 - Distribution: Ubuntu 20.04 Focal Fossa
 - Release: 5.10.120-tegra
jtop:
 - Version: 4.2.9
 - Service: Active
Libraries:
 - CUDA: 11.4.315
 - cuDNN: 8.6.0.166
 - TensorRT: 8.5.2.2
 - VPI: 2.3.9
 - Vulkan: 1.3.204
 - OpenCV: 4.5.4 - with CUDA: NO

wjl-linux@ubuntu:~$ nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2022 NVIDIA Corporation
Built on Sun_Oct_23_22:16:07_PDT_2022
Cuda compilation tools, release 11.4, V11.4.315
Build cuda_11.4.r11.4/compiler.31964100_0
wjl-linux@ubuntu:~$ python3 --version
Python 3.8.10
wjl-linux@ubuntu:~$ python3 -c "import torch; print(torch.__version__)"
2.4.0
```


# 3-操作说明
## （1）明确目标-你要做什么
（1）我们学习AI训练模型的过程中，要大概知道在什么阶段，我们之前完成软件的准备，本次是数据的准备和模型训练，本章我们目前在阶段3。


 - 阶段 1: 硬件设置
 - 阶段 2: 软件准备
 - 阶段 3: 数据准备
 - 阶段 4: 模型设计和训练
 - 阶段 5: 优化和部署
 - 阶段 6: 应用集成
 - 阶段 7: 监控和维护

（2）当然你还需要知道，你到底要做什么，盯紧你的目标，本次我们主要是使用一个模型简单识别屏幕是否开关，如果屏幕是开着的，就打印一串字符，如果是关着的，打印另一串字符。




## （2）收集数据过程
在前期收集图片其实是非常枯燥且乏味的一个过程，但是需要训练出AI，这是必不可少的部分，有时候为了提供识别率，需要大量数据，自己学习的时候，亲自弄这些数据还是非常吃力的，弄几张还好，但是量一旦超过50张，就非常痛苦了，而AI需要训练的量往往要百张起步，多则上万。

所以我们需要通过一些方式，比如扩大或者缩小，进行“数据增强”

### 1、数据输入两种方式：直接使用图片和带有文件标签。

在训练AI模型时，可以用到以下两种常见的数据输入方式：

1. 直接使用图片进行训练：
   - 方式名称：端到端训练（End-to-End Training）
   - **描述**：在这种方式中，模型直接从原始图像中学习特征，并预测输出。训练过程中，输入是图片，标签是对应的分类、回归值或其他任务的目标。模型会通过卷积神经网络（CNN）等结构从图片中提取特征，并在最终输出层进行预测。这个过程不需要额外的特征提取步骤，模型在训练过程中自动学习图像的特征。

2. 使用带有文件说明的数据进行训练**：
   - 方式名称：有监督学习（Supervised Learning）或基于标签的训练**（Label-Based Training）
   - **描述**：在这种方式中，图像数据和相应的标签通常存储在不同的文件中，或通过一种说明文件（如CSV文件或XML文件）来关联。在说明文件中，通常包含了图像的文件路径、类别标签、边界框坐标（如果是目标检测任务），甚至是图像中每个像素的标签（如果是语义分割任务）。模型通过这些标签信息来学习图像与其标签之间的映射关系。

### 2、采用缩放和其他方式，进行数据增强

将原有图片进行**扩大或缩小**后用于训练AI模型的过程被称为**数据增强（Data Augmentation）**中的一种技术，具体称为**缩放（Scaling）**。

---

#### （1）数据增强（Data Augmentation）

**数据增强**是一种通过对现有训练数据进行各种变换来生成新的训练样本的方法，旨在：

- **提高模型的泛化能力**：通过增加数据的多样性，帮助模型更好地适应不同的输入，减少过拟合的风险。
- **增强模型的鲁棒性**：使模型在面对各种变化和噪声时仍能保持良好的性能。
- **弥补数据不足**：在训练数据有限的情况下，数据增强可以有效增加训练样本的数量。

---

#### （2）缩放（Scaling）

**缩放**是数据增强技术中的一种常见方法，通过对图像进行**放大或缩小**操作，来生成具有不同尺寸比例的训练样本。

##### 1-缩放的目的和优势

- **模拟不同的视距和尺寸**：通过缩放，模型可以学习到对象在不同尺寸下的特征表现，提高在实际应用中识别不同大小对象的能力。
- **增强不变性**：缩放操作使模型对尺度变化更具鲁棒性，能够更准确地识别和分类各种尺寸的对象。
- **丰富数据多样性**：生成不同尺寸的图像，增加训练数据的多样性，有助于提高模型的泛化性能。

##### 2-实现方法

- **等比例缩放**：保持图像的宽高比例不变，对图像进行统一的放大或缩小。
- **非等比例缩放**：对图像的宽度和高度进行不同程度的缩放，改变图像的比例关系。
- **随机缩放**：在一定范围内随机选择缩放比例，增加数据的随机性和多样性。

##### 3-注意事项

- **保持图像质量**：在缩放过程中，应注意避免失真或模糊，确保缩放后的图像仍然清晰可辨。
- **边界处理**：对于放大后的图像，可能会出现超出原始尺寸的情况，需要适当裁剪或填充。
- **与其他增强方法结合**：缩放常常与其他数据增强方法（如旋转、平移、翻转、添加噪声等）结合使用，进一步提高数据多样性。

---

##### 4-其他常见的数据增强方法

除了**缩放**，还有许多其他的数据增强技术可用于提高模型性能：

1. **旋转（Rotation）**：将图像按一定角度旋转。
2. **平移（Translation）**：在水平或垂直方向上移动图像。
3. **翻转（Flipping）**：对图像进行水平或垂直翻转。
4. **裁剪（Cropping）**：截取图像的某一部分。
5. **添加噪声（Adding Noise）**：在图像中加入随机噪声。
6. **颜色抖动（Color Jittering）**：随机改变图像的亮度、对比度和饱和度。
7. **仿射变换（Affine Transformation）**：对图像进行缩放、旋转、平移和剪切等组合变换。


## （3）安装相关库pillow
`pip install pillow` 是用来安装 Python 图像处理库 Pillow 的命令。
```shell
pip install pillow
```
### 1-Pillow 简介：

- **Pillow** 是 Python 编程语言的图像处理库。它是 Python Imaging Library (PIL) 的一个分支和扩展，是处理图像文件和执行图像处理任务的常用工具。
- **功能**：
  - **打开和保存图像**：支持多种图像格式，如 JPEG、PNG、BMP、GIF、TIFF 等。
  - **图像处理**：包括裁剪、调整大小、旋转、翻转、滤镜、颜色转换等操作。
  - **图像绘制**：可以在图像上绘制图形、文本等。
  - **格式转换**：可以将图像从一种格式转换为另一种格式。

### 2-`pip install pillow` 的含义：

- **pip**：Python 包管理器，用于安装和管理 Python 包。
- **install**：pip 的安装命令，用于安装指定的包。
- **pillow**：指定要安装的包的名称。



# 4-实验过程


我们先需要收集图片，因为我们本次也是一个简单样例，使用电脑的话，相关素材也是比较容易获取，使用手机给电脑屏幕拍照，拍一些屏幕开机的图片，在拍照一些屏幕关闭的图片。然后做好标记，也就是命好名字，关闭屏幕的图片放在一个目录里，打开屏幕的放在一个目录。
### 1收集图片
如下 ，在我自己的jetson里的目录结果大致是这样的，我这里图片是自己拍照的，实际上你需要自己创建这样的过一个目录，将“off_status”和“on_status”分别装好素材，我的图片无法提供，里面有其他信息，所以如果你照着这个样例，请自己使用手机收集一些，每种大概20张吧。
```shell
.
├── off_status
│   ├── off-001.jpg
│   ├── off-002.jpg
│   ├── off-003.jpg
│   ├── off-004.jpg
│   ├── off-005.jpg
│   ├── off-006.jpg
│   ├── off-007.jpg
│   ├── off-008.jpg
│   ├── off-009.jpg
│   ├── off-010.jpg
│   ├── off-011.jpg
│   ├── off-012.jpg
│   ├── off-013.jpg
│   ├── off-014.jpg
│   ├── off-015.jpg
│   ├── off-016.jpg
│   ├── off-017.jpg
│   ├── off-018.jpg
│   └── off-019.jpg
├── off_status_outfile
├── on_status
│   ├── on-001.jpg
│   ├── on-002.jpg
│   ├── on-003.jpg
│   ├── on-004.jpg
│   ├── on-005.jpg
│   ├── on-006.jpg
│   ├── on-007.jpg
│   ├── on-008.jpg
│   ├── on-009.jpg
│   ├── on-010.jpg
│   ├── on-011.jpg
│   ├── on-012.jpg
│   ├── on-013.jpg
│   ├── on-014.jpg
│   ├── on-015.jpg
│   ├── on-016.jpg
│   ├── on-017.jpg
│   ├── on-018.jpg
│   ├── on-019.jpg
│   ├── on-020.jpg
│   ├── on-021.jpg
│   └── on-022.jpg
├── on_status_outfile
├── processed_images
│   ├── off-001.jpg
│   ├── off-002.jpg
│   ├── off-003.jpg
│   ├── off-004.jpg
│   ├── off-005.jpg
│   ├── off-006.jpg
│   ├── off-007.jpg
│   ├── off-008.jpg
│   ├── off-009.jpg
│   ├── off-010.jpg
│   ├── off-011.jpg
│   ├── off-012.jpg
│   ├── off-013.jpg
│   ├── off-014.jpg
│   ├── off-015.jpg
│   ├── off-016.jpg
│   ├── off-017.jpg
│   ├── off-018.jpg
│   ├── off-019.jpg
│   ├── resizeoff-001.jpg
│   ├── resizeoff-002.jpg
│   ├── resizeoff-003.jpg
│   ├── resizeoff-004.jpg
│   ├── resizeoff-005.jpg
│   ├── resizeoff-006.jpg
│   ├── resizeoff-007.jpg
│   ├── resizeoff-008.jpg
│   ├── resizeoff-009.jpg
│   ├── resizeoff-010.jpg
│   ├── resizeoff-011.jpg
│   ├── resizeoff-012.jpg
│   ├── resizeoff-013.jpg
│   ├── resizeoff-014.jpg
│   ├── resizeoff-015.jpg
│   ├── resizeoff-016.jpg
│   ├── resizeoff-017.jpg
│   ├── resizeoff-018.jpg
│   ├── resizeoff-019.jpg
│   ├── size1off-001.jpg
│   ├── size1off-002.jpg
│   ├── size1off-003.jpg
│   ├── size1off-004.jpg
│   ├── size1off-005.jpg
│   ├── size1off-006.jpg
│   ├── size1off-007.jpg
│   ├── size1off-008.jpg
│   ├── size1off-009.jpg
│   ├── size1off-010.jpg
│   ├── size1off-011.jpg
│   ├── size1off-012.jpg
│   ├── size1off-013.jpg
│   ├── size1off-014.jpg
│   ├── size1off-015.jpg
│   ├── size1off-016.jpg
│   ├── size1off-017.jpg
│   ├── size1off-018.jpg
│   └── size1off-019.jpg
├── py_off_status_outfile1_set_resize_parser.py
├── py_off_status_outfile1_set_resize.py
├── py_off_status_outfile.py
└── test_modify_picture_demo1
    ├── cropped_image.jpg
    ├── filtered_image.jpg
    ├── original_image.jpg
    ├── resized_image.jpg
    └── test_demo_jpg.py

6 directories, 106 files

```

### 2 使用python代码进行图片复制


确实有些就久远了，查了历史操作记录，我没记错的话，需要安装个东西，才能运行python东西 ，如上

```shell
pip install pillow
```

如下代码，我们先小试牛刀，尝试使用python脚本来复制图片，后续加载模型什么都需要python。

```python
import os
import shutil

# 原始图片文件夹和目标文件夹路径
source_dir = 'off_status'
target_dir = 'off_status_outfile'

# 确保目标文件夹存在
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

# 遍历原始图片文件夹中的图片文件列表
for filename in os.listdir(source_dir):
    # 如果是文件而不是文件夹
    if os.path.isfile(os.path.join(source_dir, filename)):
        # 构造原始图片路径和目标图片路径
        source_path = os.path.join(source_dir, filename)
        target_path = os.path.join(target_dir, filename)
        # 复制图片到目标文件夹中
        shutil.copyfile(source_path, target_path)

print("图片批量复制完成！")
```

编写好代码后，使用如下命令进行运行测试。

```shell
  570  vim test_demo_jpg.py
  571  python3 test_demo_jpg.py 
```

### 3 使用python代码进行图片缩放
能够复制图片后，那么说明你大部分python环境都有了，就可以对图片做其他操作了。
如下代码，我们将图片复制缩小一下，然后放在另一个文件夹里。
```python
import os
from PIL import Image

# 原始图片文件夹路径和目标文件夹路径
source_dir = 'off_status'
target_dir = 'processed_images'

# 确保目标文件夹存在
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

# 遍历原始图片文件夹中的图片文件列表
for filename in os.listdir(source_dir):
    # 构造原始图片路径和目标图片路径
    source_path = os.path.join(source_dir, filename)
    target_path = os.path.join(target_dir, "resize"+filename)

    # 打开原始图片
    with Image.open(source_path) as img:
        # 在这里可以添加图像处理操作，如调整大小、裁剪、滤镜等
        # 例如，调整图片大小为 224x224
        img = img.resize((224, 224))

        # 保存处理后的图片
        img.save(target_path)

print("图片处理完成！")

```
如下，处理好的图，可以看到图片大小都是变成一致的了。
这个步骤，我理解是，将图片弄成一样大小的，好方便AI用于模型训练。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/91e02e547fa6471d8d4991af1b758305.png)
如下是运行脚本过程

```shell
  578  vim py_off_status_outfile1_set_resize.py

  580  python3 py_off_status_outfile1_set_resize.py
```


### 4 尝试在脚本后面带参数
后来有尝试，希望能输出不同格式的图片，那么就需要在终端输入的时候，后面带有参数。

```c
import os
import argparse
from PIL import Image

def resize_images(source_dir, target_dir, size):
    # 确保目标文件夹存在
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # 遍历原始图片文件夹中的图片文件列表
    for filename in os.listdir(source_dir):
        # 构造原始图片路径和目标图片路径
        source_path = os.path.join(source_dir, filename)
        target_path = os.path.join(target_dir, "size1"+filename)

        # 打开原始图片
        with Image.open(source_path) as img:
            # 调整图片大小
            img_resized = img.resize(size)

            # 保存处理后的图片
            img_resized.save(target_path)

if __name__ == "__main__":
    # 创建参数解析器
    parser = argparse.ArgumentParser(description="Resize images")

    # 添加命令行参数
    parser.add_argument("source_dir", help="Path to the source directory containing images")
    parser.add_argument("target_dir", help="Path to the target directory to save resized images")
    parser.add_argument("--size", nargs=2, type=int, default=[224, 224], help="Target size of the images (width height)")

    # 解析命令行参数
    args = parser.parse_args()

    # 调用函数处理图片
    resize_images(args.source_dir, args.target_dir, tuple(args.size))

    print("图片处理完成！")
```

如下是运行脚本过程

```shell
  584  vim py_off_status_outfile1_set_resize_parser.py
  585  python3  py_off_status_outfile1_set_resize_parser.py off_status processed_images --size 300 200
```




### 5 尝试过程记录
如下为整个过程桥写命令行，其中也是犯了很多错误，但是有点就久远了，就这放在这里吧。
(1) 过程记录


```shell
  563  cd open_and_close_jgp/
  564  ls
  565  mkdir off_status_outfile
  566  mkdir on_status_outfile
  567  ls
  568  pip install pillow
  569  vim test_demo_jpg
  570  vim test_demo_jpg.py
  571  python3 test_demo_jpg.py 
  572  pwd
  573  ls
  574  ls off_status
  575  vim py_off_status_outfile.py
  576  py
  577  python3 py_off_status_outfile.py 
  578  vim py_off_status_outfile1_set_resize.py
  579  python3 py_off_status_outfile1_set_resize.py.py 
  580  python3 py_off_status_outfile1_set_resize.py
  581  vim py_off_status_outfile1_set_resize.py
  582  python3 py_off_status_outfile1_set_resize.py
  583  vim py_off_status_outfile1_set_resize.py
  584  vim py_off_status_outfile1_set_resize_parser.py
  585  python3  py_off_status_outfile1_set_resize_parser.py off_status processed_images --size 300 200

```

（2）文件夹记录
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/718fe5764a2b4be787ebe8b2acac3beb.png)



（3）早期尝试编写图片
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/829593ebd0744accbd3171e8ea32e1a9.png)
早期尝试编写代码

```python
from PIL import Image, ImageFilter

# 1. 打开原始图片
original_image = Image.open("original_image.jpg")

# 2. 调整大小
new_size = (400, 300)
resized_image = original_image.resize(new_size)

# 3. 裁剪
box = (100, 100, 300, 200)  # (left, top, right, bottom)
cropped_image = original_image.crop(box)

# 4. 应用滤镜
filtered_image = original_image.filter(ImageFilter.BLUR)

# 5. 保存生成的图片
resized_image.save("resized_image.jpg")
cropped_image.save("cropped_image.jpg")
filtered_image.save("filtered_image.jpg")

```





# 5-代码链接


### 1-目录结构截图
如下图，为自己的目录结构，如果你的目录不是按照我这样的，需要让python里的代码，找得到路径，也就是修改python文件里的路径

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/cf824e4641854b0fb46cfa6cf247f210.png)
### 2-代码链接
需要的请下载拿走
[https://download.csdn.net/download/qq_22146161/89696033](https://download.csdn.net/download/qq_22146161/89696033)



# 6-细节部分

### （1）适当的数据增强
适当的数据增强技术是指在训练过程中对训练数据进行各种变换，以增加数据的多样性和数量，从而提高模型的泛化能力和鲁棒性。这些技术在不增加实际数据数量的情况下，可以帮助模型更好地学习数据特征，==减少过拟合==。常见的数据增强技术包括：

 - 旋转：随机旋转图像一定角度范围内的角度，例如±10度。
 - 平移：随机在水平和垂直方向上平移图像。
 - 缩放：随机缩放图像，例如在一定比例范围内缩放。
 - 翻转：随机水平或垂直翻转图像。
 - 裁剪：随机裁剪图像的一部分。
 - 调整亮度：随机调整图像的亮度。
 - 调整对比度：随机调整图像的对比度。
 - 添加噪声：向图像添加随机噪声。
 - 色彩变换：随机调整图像的色彩、饱和度等。

### （2）需要准备多少张图片
需要准备多少张图片取决于多个因素，包括具体任务、模型复杂度、数据的多样性和所需的精度。以下是一些通用的指导方针：

 - 简单任务：对于简单的分类任务（例如屏幕开机与关机的识别），几百到几千张图片可能是一个好的起点。每个类别至少有100-200张图片是一个合理的起点。

 - 复杂任务：对于更复杂的任务，例如包含多个类别的分类任务，或者需要高精度的任务，通常需要更多的数据。每个类别至少有几百到几千张图片可能是必要的。

 - 数据多样性：确保数据集覆盖各种不同的情况和场景，以提高模型的泛化能力。例如，光照条件、角度、距离等因素的变化都应该在数据集中体现。

 - 模型复杂度：更复杂的模型（例如深度卷积神经网络）通常需要更多的数据来避免过拟合。如果您的模型非常深，可能需要数万张图片。

 - 数据增强：如果数据有限，可以使用数据增强技术来增加数据量。数据增强可以通过旋转、翻转、缩放、颜色变换等方法生成新的样本。

### （3）在运行Python时，输入参数如何写
在运行Python脚本时通过命令行参数来指定图片大小，可以使用Python的==argparse==模块来解析命令行参数。以下是示例代码，如何使用argparse来实现：

```python
import os
import argparse
from PIL import Image

def resize_images(source_dir, target_dir, size):
    # 确保目标文件夹存在
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # 遍历原始图片文件夹中的图片文件列表
    for filename in os.listdir(source_dir):
        # 构造原始图片路径和目标图片路径
        source_path = os.path.join(source_dir, filename)
        target_path = os.path.join(target_dir, filename)
        
        # 打开原始图片
        with Image.open(source_path) as img:
            # 调整图片大小
            img_resized = img.resize(size)
            
            # 保存处理后的图片
            img_resized.save(target_path)

if __name__ == "__main__":
    # 创建参数解析器
    parser = argparse.ArgumentParser(description="Resize images")

    # 添加命令行参数
    parser.add_argument("source_dir", help="Path to the source directory containing images")
    parser.add_argument("target_dir", help="Path to the target directory to save resized images")
    parser.add_argument("--size", nargs=2, type=int, default=[224, 224], help="Target size of the images (width height)")

    # 解析命令行参数
    args = parser.parse_args()

    # 调用函数处理图片
    resize_images(args.source_dir, args.target_dir, tuple(args.size))

    print("图片处理完成！")

```
之后可以试试带有参数的方式

```c
python3 resize_images.py off_status processed_images --size 300 200
```


### （4）批量出来图片的两张方式
目前在网上查到，如果想批量出来图片，基本有两种方法。使用相关图片处理软件和python等语言编写代码。

 - 图像处理软件：可以使用诸如Adobe Photoshop、GIMP（GNU Image Manipulation Program）等图像处理软件来手动调整和编辑图像。这些软件提供了各种图像处理工具和滤镜，可以帮助您调整图像的大小、色彩、对比度、亮度等参数，并添加特效或者滤镜以生成新的图片。

 - Python编程：如果熟悉编程，可以使用Python编程语言以及相应的图像处理库来自动化调整图片。Python中常用的图像处理库包括OpenCV、Pillow、scikit-image等。可以编写Python脚本来批量读取、调整和保存图片，实现自动化的图片处理。


### （5）在win上有软件，或者“在线的网站” 能否解决批量软件问题
批量出来图片还是挺费时费力的，那么有没批量出的的软件或者在线的网站呢？以下是上网查到的，自己并没有真正尝试过。

在Windows上有一些图像处理软件可以批量调整和编辑图片，例如：

 - Adobe Photoshop：Adobe Photoshop是一个功能强大的图像处理软件，可以用于编辑、调整和生成图片。它提供了丰富的图像处理工具和滤镜，可以满足各种图像处理需求。

 - GIMP：GIMP是一个免费的开源图像处理软件，功能类似于Adobe Photoshop。它提供了诸如调整大小、色彩、对比度等功能，可以帮助您批量编辑图片。

 - IrfanView：IrfanView是一个轻量级的图像查看和编辑软件，支持批量处理图片。它提供了简单易用的界面和各种图像处理工具，可以用于快速调整和编辑图片。

 - FastStone Image Viewer：FastStone Image Viewer是一个免费的图像查看和编辑软件，支持批量重命名、调整大小、旋转等操作，适用于快速处理大量图片。

除了本地软件，还有一些在线网站和工具可以批量调整和编辑图片，例如：

 - Pixlr：Pixlr是一个在线图像编辑工具，提供了类似于Photoshop的功能，包括调整大小、裁剪、滤镜等功能。您可以在网页上上传并编辑您的图片。

 - Canva：Canva是一个在线设计平台，提供了丰富的设计工具和模板，可以帮助您快速创建和编辑图片。它包含了各种调整大小、滤镜、文字等功能。

 - Fotor：Fotor是一个在线图像编辑工具，提供了各种调整大小、色彩、对比度等功能，同时还提供了丰富的滤镜和特效供您选择。

### （6）获取图片数据的方式
简单来说，如何得到你需要的图片，第一种，就是你自己直接那摄像头去拍摄，另外可以去网上找素材，别人拍的，如果是你需要的，可以用的话，要注意版权问题。

 - 摄像头采集: 如果有摄像头设备可用，可以使用摄像头实时采集图像。可以使用Python的OpenCV库或者其他图像处理库来捕获摄像头的实时图像数据，并保存为图像文件。

 - 屏幕截图: 如果要获取的是屏幕上的图像，可以使用屏幕截图工具来捕获屏幕上的图像。在Windows操作系统上，可以使用PrtScn键或者Windows + Shift + S组合键来进行屏幕截图，并保存为图像文件。在其他操作系统上也有类似的屏幕截图工具可用。

 - 在线图像库: 有许多在线图像库提供了大量的免费图像资源，可以在这些库中搜索并下载需要的图像。一些知名的在线图像库包括Unsplash、Pexels、Pixabay等。可以使用Python的库来编写脚本程序自动从这些在线库中下载图像。

 - 实时数据采集: 如果任务涉及到实时数据采集，例如监控摄像头、传感器数据等，可以编写脚本程序来实时获取并保存数据。这可能需要与硬件设备进行交互，具体操作取决于硬件设备和通信协议。

### （7）是否考虑图像的分辨率
在图片收集过程中，是需要考虑具体图片这个数据的，比如分辨率大小等。但也和需要任务，计算资源和图像的收集成本，等都相关。

分辨率在图像数据收集过程中是一个考虑因素。图像的分辨率直接影响着模型对图像特征的提取和识别能力，因此选择合适的分辨率可以提高模型的性能和泛化能力。

在考虑图像分辨率时，需要权衡以下几个因素：

 - 模型复杂度: 更高分辨率的图像通常意味着更多的像素和更多的细节，这可能需要更复杂的模型来处理。如果模型比较简单，可能需要降低图像分辨率以减少模型的复杂度。

 - 计算资源: 更高分辨率的图像需要更多的计算资源来处理，包括更多的内存和更长的训练时间。在嵌入式设备或者资源有限的环境下，可能需要降低图像分辨率以减少计算开销。

 - 数据多样性: 图像的分辨率应该能够捕获到您感兴趣的特征和信息。如果需要模型能够识别细微的特征或者进行精细的分类，可能需要更高分辨率的图像来提供足够的信息。

 - 数据收集成本: 更高分辨率的图像可能需要更高成本的摄像设备和存储空间来收集和存储。需要考虑预算和资源限制，以确定适合的图像分辨率。

### （8）收集到的图片，命名问题
因为我们本次，或者说本章的目的是要识别开关机，所以可以以“off”和“on”这样的字样来命名图片

 - 类别标识: 在命名中包含图片所属的类别信息是很重要的。例如，您可以在文件名中使用“on”表示开机状态，使用“off”表示关机状态。

 - 唯一标识符: 在类别标识后面，可以添加一个唯一的标识符，例如数字序号或者日期时间信息，以确保每个文件都有一个唯一的命名。这样可以避免文件名的冲突，并且方便管理。

 - 附加信息: 如果有必要，可以在文件名中添加一些附加的信息，例如拍摄时间、设备信息、拍摄者姓名等。这些信息可以帮助您更好地理解和管理数据。

 - 文件格式: 最后，不要忘记在文件名中包含文件的格式信息，例如“.jpg”或“.png”。这样可以确保您在处理数据时能够清楚地知道文件的格式。

```php
<类别标识>-<唯一标识符>.<文件格式>
```
以我本次实验为例

例如：

开机状态的图片命名：on-001.jpg, on-002.jpg, ...
关机状态的图片命名：off-001.jpg, off-002.jpg, ...

# 7-总结


