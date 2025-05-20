@[TOC](【学习AI-相关路程-工具使用-自我学习-jetson&模型训练-问题补充-无法使用GPU问题-CUDA is available: False-补充说明 （6）】)
# 1 前言
（1）说明
之前在训练模型并使用模型检测的时候，发现一直用的是cpu，没有使用gpu，尝试了一些办法，都没有成功，前几天终于调试过了吧，因此记录下来。

如下为背景，是使用模型训练和识别的过程。
[【学习AI-相关路程-工具使用-自我学习-jetson&模型训练-图片识别-使用模型检测图片-基础样例 （5）】](https://waka-can.blog.csdn.net/article/details/141856181)

（2）使用GPU和CPU

从个人的角度讲，GPU和CPU都是能使用，认为是一个好事，因为模型训练好了后，不一定在jetson orin NX上使用，对于跨其他系统的时候，如果gpu一旦用不了，使用cpu的话，就非常方便了，如果在本机上使用或者同类型的设备迁移，使用GPU也是非常理想的。
# 2 问题描述
### 1、发现问题
如下图，训练好模型后，能用以识别模型每次调用的事“CPU”。

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/54fb2a3be9bc4c1c9a2285754afb1cc6.png)
其实在代码里只是加入如下代码，如果经常看AI相关代码，如下代码应该比较熟悉的，基本是这种写法。

```shell
import torch
# 设置设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
```
然后，根据上述代码我单独写了一个python文件，想将问题分离出来。
但是发现结果是一样的。

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/fbd94339c34c4792b37d34cc97751399.png)
### 2、问题说明：CUDA is available: False

至此，出现的问题就是：CUDA is available: False

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/1e0e2427d5894948a98e0920bd4807dd.png)



# 3 问题解决思路&设备版本查询确认
## 1、问题解决思路
### 1. CUDA 和 cuDNN 是否正确安装和配置
确保 Jetson Orin NX 上已经正确安装了 CUDA 和 cuDNN。这通常通过 NVIDIA JetPack SDK 安装。JetPack 包含了 CUDA Toolkit 和 cuDNN 以及其他必要的库和工具，这个通过之前工具安装的，这里应该就不是问题点。
### 2. 正确配置环境变量
检查环境变量是否正确配置。特别是，确保 `PATH` 和 `LD_LIBRARY_PATH` 包含了 CUDA 的路径。可以通过以下命令检查和设置这些环境变量：

```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```



### 3. PyTorch 是否支持 Jetson 平台
如果你使用的是 PyTorch，需要确保安装的是为 Jetson 平台编译的 PyTorch 版本。Jetson 平台使用了特定的 ARM 架构，所以不能使用标准的 x86_64 PyTorch 版本。NVIDIA 提供了专门为 Jetson 平台编译的 PyTorch 版本，通常可以通过以下步骤安装：

```bash
sudo apt-get install python3-pip libopenblas-base libopenmpi-dev
pip3 install torch torchvision torchaudio
```

或者你可以参考 [NVIDIA PyTorch for Jetson](https://developer.nvidia.com/embedded/pytorch) 的官方指南来获取最新版本和安装指令。

### 4. 确保 GPU 驱动程序正常运行
确保 NVIDIA 驱动程序正常工作，你可以使用以下命令检查 GPU 是否被系统识别：

```bash
nvidia-smi
```

如果 GPU 没有被识别，可能需要检查驱动程序是否正确安装，或者重新安装 JetPack。

### 5. 更新并重启系统
有时候，更新系统和重启可以解决一些潜在的问题：

```bash
sudo apt-get update
sudo apt-get upgrade
sudo reboot
```

### 6. 使用不同的 PyTorch 版本
在某些情况下，PyTorch 版本可能与当前的 CUDA 版本不兼容。你可以尝试安装不同的 PyTorch 版本或 CUDA 版本，或者使用 NVIDIA 的容器化解决方案（如 NVIDIA Docker 或 NGC）来确保兼容性。

### 7.  Jetson Orin NX 上检查 GPU 状态

（1）tegra_stats:

```bash
sudo tegrastats
```
这会显示包括 CPU、GPU、内存等资源的实时使用情况。

（2）jetson_stats (jtop): jetson_stats 是一个开源工具，可以安装并使用来监控 Jetson 设备的状态。

首先，安装 jetson_stats:

```bash
sudo -H pip3 install -U jetson-stats
```
然后运行 jtop:

```bash
sudo jtop
```
这个工具提供了一个类似 nvidia-smi 的界面，可以实时查看 Jetson 设备的资源使用情况



## 2、设备版本查询确认
在解决问题过程中，发现了解自己设备版本很重要，必须找到和自己设备版本对应的软件才行，但是如何查版本，命令是什么当时自己不是很了解，于是这部分，专门将各个指令记录下来，辅助你查找的时候，快速定位，并且在之前也是说了。
### 1、设备各个版本指令记录
如下为各个版本查询指令，在前一篇文章也写了，其实还是挺重要的，帮你确认信息。
##### 1. 验证CUDA和cuDNN是否正常工作
确保CUDA和cuDNN都已经正确安装，并且可以正常使用。可以使用以下命令进行验证：


（1）运行以下命令查看CUDA版本：

```shell
nvcc --version
```
（2）执行以下命令来检查cuDNN的版本：
```shell
cat /usr/include/cudnn_version.h | grep CUDNN_MAJOR -A 2
```
##### 2. 验证PyTorch和torchvision安装
确保PyTorch和torchvision已经正确安装并与CUDA兼容。运行以下命令来检查：

(1)在Python环境中，输入以下命令：



```python
import torch
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
```
CUDA available: 应该返回 True，这表示PyTorch可以识别并使用CUDA。



（2）验证torchvision是否安装正确，在Python环境中，输入以下命令：
```python
import torchvision
print("torchvision version:", torchvision.__version__)
```

##### 3. 查看Jetson设备的系统信息

使用jetson_release命令来查看设备的详细信息：
```shell
jetson_release
```
这将显示包括JetPack版本、CUDA版本、cuDNN版本等详细信息。



##### 4. 查看python版本信息
使用如下命令，来查看python相关版本信息
```shell
python3 --version
```

### 2、设备各个版本指令记录
如下为自己实际操作。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/a9c2c8d523794bbc8ddad1c52e203536.png)

```c
wjl-linux@ubuntu:~$ nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2022 NVIDIA Corporation
Built on Sun_Oct_23_22:16:07_PDT_2022
Cuda compilation tools, release 11.4, V11.4.315
Build cuda_11.4.r11.4/compiler.31964100_0
wjl-linux@ubuntu:~$ 
wjl-linux@ubuntu:~$ 
wjl-linux@ubuntu:~$ cat /usr/include/cudnn_version.h | grep CUDNN_MAJOR -A 2
#define CUDNN_MAJOR 8
#define CUDNN_MINOR 6
#define CUDNN_PATCHLEVEL 0
--
#define CUDNN_VERSION (CUDNN_MAJOR * 1000 + CUDNN_MINOR * 100 + CUDNN_PATCHLEVEL)

/* cannot use constexpr here since this is a C-only file */
wjl-linux@ubuntu:~$ python3 
Python 3.8.10 (default, Nov 22 2023, 10:22:35) 
[GCC 9.4.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import torch
>>> print("PyTorch version:", torch.__version__)
PyTorch version: 1.12.0
>>> print("CUDA available:", torch.cuda.is_available())
CUDA available: False
>>> import torchvision
/home/wjl-linux/.local/lib/python3.8/site-packages/torchvision/io/image.py:13: UserWarning: Failed to load image Python extension: 
  warn(f"Failed to load image Python extension: {e}")
>>> print("torchvision version:", torchvision.__version__)
torchvision version: 0.13.0
>>> python3 --version
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
NameError: name 'python3' is not defined

```


其中获取了很多信息，比较重要的jetpack信息，其实是后面匹配torch版本的关键。

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/18702cf191684951a35624f1c6d2e6e0.png)

```shell
wjl-linux@ubuntu:~$ python3 --version
Python 3.8.10
wjl-linux@ubuntu:~$ 
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
wjl-linux@ubuntu:~$ torch.version.cuda
bash: torch.version.cuda: command not found


```


# 4 我的努力 

### （1）确认环境变量
如下我第一时间查看了这两个环境变量确保没有问题。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/55b723a7ef8045b4b83f714d54ba568f.png)

### （2）查看GPU状态
使用如下命令，监控 Jetson 设备的状态。

```c
sudo jtop
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/197f7dcc38744aa282d664386622f550.png)




### （3）使用指令“nvcc --version”查看信息。
如下为使用“nvcc --version”指令查看到信息。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/d9cb64b99b204fa5a63d30cfc36e888f.jpeg)
### （4）尝试安装其他版本torch和torchvision
如下为当时不知道如何找到正确版本，尝试先卸载，在网上找了一版本，进行安装。

```shell
# 先卸载可能不匹配的PyTorch版本
sudo pip3 uninstall torch torchvision

# 重新安装NVIDIA提供的与JetPack版本匹配的PyTorch
sudo apt-get install python3-pip libopenblas-base libopenmpi-dev
pip3 install --upgrade pip
sudo pip3 install torch-1.14.0+nv23.03-cp38-cp38-linux_aarch64.whl torchvision-0.15.0+nv23.03-cp38-cp38-linux_aarch64.whl
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/8e3a8e4d7a624826912ef2b1888136be.png)


但是如下图所示，报错了没有成功。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/c121fe73fe4a4974abc607508aca1f47.jpeg)
之后又尝试了了几个版本，发现都不行。

```shell
# 先卸载现有版本
sudo pip3 uninstall torch torchvision

# 通过 pip 安装 PyTorch 和 torchvision
sudo pip3 install torch==1.10.0+nv21.05.1 torchvision==0.11.1+nv21.05.1 -f https://developer.download.nvidia.com/compute/redist/jp/v45/pytorch/torch_stable.html
```

```shell
sudo pip3 install /path/to/torch-1.14.0+nv23.03-cp38-cp38-linux_aarch64.whl
sudo pip3 install /path/to/torchvision-0.15.0+nv23.03-cp38-cp38-linux_aarch64.whl
```

### （4）尝试在网站上先下载，然后安装版本torch和torchvision

在下载的时候，一个是发现下载老是失败，第二个是下载得慢，
```shell
sudo apt-get install python3-pip libopenblas-base libopenmpi-dev
pip3 install --pre torch torchvision torchaudio --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v50/pytorch/
```
于是直接打开链接，去下载。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/bab89a2115c444ac95b66fced988675b.png)
下载到本地后，使用指令安装

```c
pip3 install torch-1.13.0a0+340c4120.nv22.06-cp38-cp38-linux_aarch64.whl
```
但是发现还是不行
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/53bc4b7308af4befaaeeb31a1a1c7aa7.png)
如下图所示，虽然，CUDA is available: true, 但是会提示没有没有no module named 'torchvision'.
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/97830bcf34104c8b8713ec9555045638.png)
但是安装好了torchvision后，又变成False

```shell
wjl-linux@ubuntu:~/Desktop/get_AI_data_jpg_photo/dataset$ sudo jtop
Mismatch version jtop service: [4.2.7] and client: [4.2.9]. Please run:

sudo systemctl restart jtop.service
wjl-linux@ubuntu:~/Desktop/get_AI_data_jpg_photo/dataset$ sudo systemctl restart jtop.service
wjl-linux@ubuntu:~/Desktop/get_AI_data_jpg_photo/dataset$ sudo jtop
wjl-linux@ubuntu:~/Desktop/get_AI_data_jpg_photo/dataset$ ls
python_AI_built_models1.py          screen_status_model.pth  val
python_display_cuda_is_availbel.py  test_image
python_load_model_prediction.py     train
wjl-linux@ubuntu:~/Desktop/get_AI_data_jpg_photo/dataset$ cat python_display_cuda_is_availbel.py 
import torch
print("CUDA is available:", torch.cuda.is_available())

wjl-linux@ubuntu:~/Desktop/get_AI_data_jpg_photo/dataset$ python3 python_display_cuda_is_availbel.py 
CUDA is available: False
wjl-linux@ubuntu:~/Desktop/get_AI_data_jpg_photo/dataset$ nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2022 NVIDIA Corporation
Built on Sun_Oct_23_22:16:07_PDT_2022
Cuda compilation tools, release 11.4, V11.4.315
Build cuda_11.4.r11.4/compiler.31964100_0
wjl-linux@ubuntu:~/Desktop/get_AI_data_jpg_photo/dataset$
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/78655cf7ca6d44bd96af090ce8292794.jpeg)

### （5）反复尝试安装版本torch和torchvision

如下所示，再次尝试时，发安装完torch后，CUDA为true，但是安装完torchvision就不行
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/26c01602292a4189a5809b9ca4dcc407.jpeg)
但是会报没有这个模块错误。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/f23c63ed24ee458aba109e3c9c143fb2.jpeg)
安装完了torchvision，就又会CUDA is available: False
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/85cf8fa324df4212ae6a97148ba1326a.jpeg)
如下图所示，可以发现，torchvision安装后，会影响cuda
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/4eecaccdd5ea4b4b9419802397a494bb.jpeg)
通过以上两个步骤以及多次尝试，基本可以确定，==应该是torch和torchvision版本不匹配==


### （6）去英伟达官网找博客，搜索问题

如下图所示，在官网博客搜索，如下，有人遇到过类似问题。
博客:[https://forums.developer.nvidia.com/t/jetson-orin-nx-pytorch-2-1-2-on-jetpack-6-0-dp-torch-cuda-is-available-is-false/277189](https://forums.developer.nvidia.com/t/jetson-orin-nx-pytorch-2-1-2-on-jetpack-6-0-dp-torch-cuda-is-available-is-false/277189)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/e91c8bf8af784a429359ad692145475e.png)
但是这个博客其实没太看明白，根据博客里说，在如下界面，使用了里面指令，也没有能解决。


文档链接：[https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/ce22f54c6e3b4e049998c98131cc3bfb.png)
### （7）去Git官网搜索问题

git链接：[https://github.com/pytorch/vision](https://github.com/pytorch/vision)
其实这里提示torch和torchvision对应关系，但是不知道为啥不行。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/5a1ae839bd6c4835b4c5038a17afe600.png)



### （8）去PyTorch官网下载包
在如下网站，是pytorch网站链接：
[https://pytorch.org/get-started/previous-versions/](https://pytorch.org/get-started/previous-versions/)
但是自己没有找打与之匹配的cuda 11.4版本，一般来说更低版本应该检查兼容的。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/677921ecfb20495f81d53cd5fd75e9b3.png)


使用如下指令后，

```c
pip install torch==1.12.0+cu116 torchvision==0.13.0+cu116 torchaudio==0.12.0 -f https://download.pytorch.org/whl/torch_stable.html
```
还是报错了
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/a17fa077e31a4b0cbeda55974c78efc7.jpeg)



# 5 问题解决方式

### （1）解决问题的网站
如下最后还是在英伟达官方网站，找到匹配的版本，整个设备是英伟达，还低找英伟达支持。

这里说明下，==用jetpack包信息去匹配对应版本==。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/941b8ed6be724029be84cfae02156367.png)


链接：[https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048](https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/39f0a4f9b4d6463f828ec755edf9ee33.jpeg)
如下为下载文件内的内容，之前好像就下过，还是下载两遍，已经记不清了，在使用时候，准备时候最后一个版本时，发现必须是全名，带有“（1）”这样的字样也不能安装，只能把之前版本改名。

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/8848457b03054b7b9b2c472e0f270ea0.png)

```shell
wjl-linux@ubuntu:~/Downloads$ pip3 install torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64(1).whl
bash: syntax error near unexpected token `('
wjl-linux@ubuntu:~/Downloads$ pip3 install torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch6-1.whl
Defaulting to user installation because normal site-packages is not writeable
WARNING: Requirement 'torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch6-1.whl' looks like a filename, but the file does not exist
ERROR: torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch6-1.whl is not a valid wheel filename.
wjl-linux@ubuntu:~/Downloads$ pip4 install torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64-1.whl
bash: pip4: command not found
wjl-linux@ubuntu:~/Downloads$ pip3\ install torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64-1.whl
bash: pip3 install: command not found
wjl-linux@ubuntu:~/Downloads$ pip3 install torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64-1.whl
Defaulting to user installation because normal site-packages is not writeable
ERROR: torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64-1.whl is not a valid wheel filename.
wjl-linux@ubuntu:~/Downloads$ pip3 install torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64.whl
Defaulting to user installation because normal site-packages is not writeable
Processing ./torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64.whl
Requirement already satisfied: filelock in /home/wjl-linux/.local/lib/python3.8/site-packages (from torch==2.1.0a0+41361538.nv23.06) (3.14.0)
Requirement already satisfied: fsspec in /home/wjl-linux/.local/lib/python3.8/site-packages (from torch==2.1.0a0+41361538.nv23.06) (2024.3.1)
Requirement already satisfied: jinja2 in /home/wjl-linux/.local/lib/python3.8/site-packages (from torch==2.1.0a0+41361538.nv23.06) (3.1.4)
Requirement already satisfied: networkx in /home/wjl-linux/.local/lib/python3.8/site-packages (from torch==2.1.0a0+41361538.nv23.06) (3.1)
Requirement already satisfied: sympy in /home/wjl-linux/.local/lib/python3.8/site-packages (from torch==2.1.0a0+41361538.nv23.06) (1.12)
Requirement already satisfied: typing-extensions in /home/wjl-linux/.local/lib/python3.8/site-packages (from torch==2.1.0a0+41361538.nv23.06) (4.11.0)
Requirement already satisfied: MarkupSafe>=2.0 in /home/wjl-linux/.local/lib/python3.8/site-packages (from jinja2->torch==2.1.0a0+41361538.nv23.06) (2.1.3)
Requirement already satisfied: mpmath>=0.19 in /home/wjl-linux/.local/lib/python3.8/site-packages (from sympy->torch==2.1.0a0+41361538.nv23.06) (1.3.0)
Installing collected packages: torch
  Attempting uninstall: torch
    Found existing installation: torch 1.12.0
    Uninstalling torch-1.12.0:
      Successfully uninstalled torch-1.12.0
Successfully installed torch-2.1.0a0+41361538.nv23.6
wjl-linux@ubuntu:~/Downloads$ 


```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/b966eb6667da449ab53acaf9dc027a60.png)
### （2）测试验证
如下图片为测试一个开着屏幕的截图，
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/c9d951a93b6b42199a9e1350f4409125.jpeg)



和一个关着屏幕的截图
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/694d8f8c939e45f5ac01e0c03b64a8a0.jpeg)




### （3）测试记录
如下为测试相关记录
```shell
wjl-linux@ubuntu:~/Desktop/get_AI_data_jpg_photo/dataset$ ls
AI-screen-model_cade.zip            python_load_model_prediction.py  test_image
python_AI_built_models1.py          screen_status_model1.pth         train
python_display_cuda_is_availbel.py  screen_status_model.pth          val
wjl-linux@ubuntu:~/Desktop/get_AI_data_jpg_photo/dataset$ python3 python_display_cuda_is_availbel.py 
CUDA is available: True
wjl-linux@ubuntu:~/Desktop/get_AI_data_jpg_photo/dataset$ python3 python_load_model_prediction.py 
/home/wjl-linux/.local/lib/python3.8/site-packages/torchvision/io/image.py:13: UserWarning: Failed to load image Python extension: 
  warn(f"Failed to load image Python extension: {e}")
Using CUDA device: Orin
Screen is OFF
wjl-linux@ubuntu:~/Desktop/get_AI_data_jpg_photo/dataset$ python3 python_load_model_prediction.py 
/home/wjl-linux/.local/lib/python3.8/site-packages/torchvision/io/image.py:13: UserWarning: Failed to load image Python extension: 
  warn(f"Failed to load image Python extension: {e}")
Using CUDA device: Orin
Screen is OFF
wjl-linux@ubuntu:~/Desktop/get_AI_data_jpg_photo/dataset$ python3 python_load_model_prediction.py 
/home/wjl-linux/.local/lib/python3.8/site-packages/torchvision/io/image.py:13: UserWarning: Failed to load image Python extension: 
  warn(f"Failed to load image Python extension: {e}")
Using CUDA device: Orin
Screen is ON

```



1 ![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/fe15967dac7b4135811050c4f278cbd1.png)



# 6 细节部分

### （1）指令敲写错误
在实践过程中，有些自己的惯性思维了，一个是拼写容易错误，另一个是自以为 “-v”=“--version”,如下图，还是在网上先查好在输入。

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/d63aea19e9f84174b7e2ce2f9222d01d.png)

### （2）其他尝试 & 错误：No module named 'torch._custom_ops'
#### 1-其他尝试1
在安装torch和torchvision版本过程中，下载过多个包，以下算是一个记录吧。

```shell
https://developer.download.nvidia.com/compute/redist/jp/v50/pytorch/
```
#### 2-错误：No module named 'torch._custom_ops'
其中因为有个阶段也把torch也卸载了导致报错。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/e2d37c2f210440e2b6a30e5ef1228568.jpeg)

```shell
wjl-linux@ubuntu:~/Downloads$ python3 -c "import torch; print(torch.__version__)"
1.12.0a0+02fb0b0f.nv22.06
wjl-linux@ubuntu:~/Downloads$ python3 -c "import torchvision; print(torchvision.__version__)"
Traceback (most recent call last):
  File "<string>", line 1, in <module>
  File "/home/wjl-linux/.local/lib/python3.8/site-packages/torchvision/__init__.py", line 10, in <module>
    from torchvision import _meta_registrations, datasets, io, models, ops, transforms, utils  # usort:skip
  File "/home/wjl-linux/.local/lib/python3.8/site-packages/torchvision/_meta_registrations.py", line 4, in <module>
    import torch._custom_ops
ModuleNotFoundError: No module named 'torch._custom_ops'
```

#### 3-安装其他版本报错
如下如果安装版本不对，其实也会报错
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/9e86bf242ee84397a0cbc5fcf0010bc2.jpeg)
#### 4-通过下载的方式进安装
中间其实很多次都失败了。
不过成功下载后，还是报错了
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/8c40e173fc6343718e257c86e2cd2474.jpeg)



#### 5-这里是pytorch的一个stable网页

[https://download.pytorch.org/whl/torch_stable.html](https://download.pytorch.org/whl/torch_stable.html)

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/52e6dbd817e14700bacf826ae4791269.png)


### （3）过程记录
以下为自己常规调试过程中，所以指令记录，因为指令太多了，有时候不知道哪里像步骤作对了，哪些是无效的，所以都记录下。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/fc2e05b7a41b44338344c599007a0501.png)


```c
 696  nvcc --version 
  697  cat /usr/local/cuda/include/cudnn_version.h | grep CUDNN_MAJOR -A 2
  699  sudo pip3 uninstall torch torchvision
  700  sudo apt-get install python3-pip libopenblas-base libopenmpi-dev
  701  pip3 install --upgrade pip
  702  sudo pip3 install torch-1.14.0+nv23.03-cp38-cp38-linux_aarch64.whl torchvision-0.15.0+nv23.03-cp38-cp38-linux_aarch64.whl
  703  sudo pip3 uninstall torch torchvision
  704  sudo pip3 install torch==1.10.0+nv21.05.1 torchvision==0.11.1+nv21.05.1 -f https://developer.download.nvidia.com/compute/redist/jp/v45/pytorch/torch_stable.html
  705  ls
  706  python3 python_display_cuda_is_availbel.py 
  707  sudo pip3 install --upgrade pip
  708  python3 python_display_cuda_is_availbel.py 
  710  sudo pip3 uninstall torch torchvision
  712  sudo pip3 install torch==1.10.0+nv21.05.1 torchvision==0.11.1+nv21.05.1 -f https://developer.download.nvidia.com/compute/redist/jp/v45/pytorch/torch_stable.html
  713  dpkg-query --show nvidia-14t-core
  714  dpkg-query --show nvidia-l4t-core
  715  sudo apt-get update
  716  sudo apt-get install python3-pip libopenblas-base libopenmpi-dev
  717  sudo pip3 install torch-2.0.0+nv23.03-cp38-cp38-linux_aarch64.whl
  718  ls
  719  python3 python_display_cuda_is_availbel.py 
  720  cat python_AI_built_models1.py 
  721  ls
  722  python3 python_display_cuda_is_availbel.py 
  723  nvidia-smi
  724  reboot
  725  python3 python_load_model_prediction.py 
  726  ls
  727  history
  728  nvidia-smi
  729  sudo tegrastats
  730  nvcc --version
  731  sudo -H pip3 install -U jetson-stats
  732  sudo jtop
  733  sudo systemctl restart jtop.service
  734  sudo jtop
  735  ls
  736  cat python_display_cuda_is_availbel.py 
  737  python3 python_display_cuda_is_availbel.py 
  738  nvcc --version
  739  cat ~/.bashrc
  740  vim ~/.bashrc
  741  source ~/.bashrc
  742  vim ~/.bashrc
  743  python3 python_display_cuda_is_availbel.py 
  744  cuda -v
  745  nvcc --version
  746  cat python_display_cuda_is_availbel.py 
  747  pip3 uninstall torch torchvision torchaudio
  748  sudo apt-get install python3-pip libopenblas-base libopenmpi-dev
  749  https://developer.download.nvidia.com/compute/redist/jp/v50/pytorch/
  750  python3  python_display_cuda_is_availbel.py 
  751  ls
  752  python3 python_load_model_prediction.py 
  753  cat python_load_model_prediction.py 
  754  python3 python_load_model_prediction.py 
  755  python3 python_display_cuda_is_availbel.py 
  756  python3 python_load_model_prediction.py 
  757  cat python_load_model_prediction.py 
  758  python3 python_load_model_prediction.py 
  759  python3 python_display_cuda_is_availbel.py 
  760  python3 python_load_model_prediction.py 
  761  python3 python_load_model_prediction.py \
  762  python3 python_display_cuda_is_availbel.py 
  763  pip3 uninstall torch torchvision torchaudio
  764  pip3 install --pre torch torchvision torchaudio --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v51/pytorch/
  765  python3 python_display_cuda_is_availbel.py 
  766  python3 python_load_model_prediction.py \pip3 uninstall torch torchvision torchaudio
  767  python3 python_load_model_prediction.py 
  768  python3 python_display_cuda_is_availbel.py 
  769  export TORCH_INSTALL=https://developer.download.nvidia.cn/compute/redist/jp/v511/pytorch/torch-2.0.0+nv23.05-cp38-cp38-linux_aarch64.whl
  770  python3 python_display_cuda_is_availbel.py 
  771  python3 python_load_model_prediction.py 
  772  history
  773  python3 python_display_cuda_is_availbel.py 
  774  python3 python_load_model_prediction.py 
  775  python3 python_display_cuda_is_availbel.py 
  776  git clonehttps://github.com/pytorch/visioncd vision
  777  git clone https://github.com/pytorch/visioncd vision
  778  git clone https://github.com/pytorch/vision
  779  python3 python_display_cuda_is_availbel.py 
  780  python3 python_load_model_prediction.py 
  781  python3 python_display_cuda_is_availbel.py 
  782  python3 python_load_model_prediction.py 
  783  python3 python_display_cuda_is_availbel.py 
  784  pip install torch-1.13.0a0+340c4120.nv22.06-cp38-cp38-linux_aarch64.whl 
  785  pip3 install torchvision
  786  ls
  787  history 
  788  pip3 uninstall torchvision
  789  pip install torch-1.13.0a0+340c4120.nv22.06-cp38-cp38-linux_aarch64.whl 
  790  pip3 install --pre torch torchvision torchaudio --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v50/pytorch/
  791  pip install torch-1.13.0a0+340c4120.nv22.06-cp38-cp38-linux_aarch64.whl 
  792  python3 -c "import torch; print(torch.__version__)"
  793  python3 -c "import torchvision; print(torchvision.__version__)"
  794  ls
  795  pip3 install tensorflow-2.9.1+nv22.06-cp38-cp38-linux_aarch64.whl 
  796  python -m pip install --upgrade pip
  797  ls
  798  pip3 install torch-2.0.0+nv23.05-cp38-cp38-linux_aarch64.whl 
  799  history 
  800  pip3 install --pre  torchvision  --extra-index-urlhttps://developer.download.nvidia.cn/compute/redist/jp/v511/pytorch/
  801  pip3 install   torchvision  --extra-index-urlhttps://developer.download.nvidia.cn/compute/redist/jp/v511/pytorch/
  802  pip3 install torchvision --extra-index-url https://developer.download.nvidia.cn/compute/redist/jp/v511/pytorch/
  803  sudo apt-get -y update; 
  804  export TORCH_INSTALL=https://developer.download.nvidia.cn/compute/redist/jp/v60dp/pytorch/torch-2.2.0a0+6a974be.nv23.11-cp310-cp310-linux_aarch64.whl
  805  head -n 1 /etc/nv_tegra_release
  806  pip3 install --pre torch torchvision torchaudio --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v50/pytorch/
  807  pip3 install torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64.whl 
  808  sudo apt-get -y install python3-pip libopenblas-dev;
  809  export TORCH_INSTALL=https://developer.download.nvidia.cn/compute/redist/jp/v511/pytorch/torch-2.0.0+nv23.05-cp38-cp38-linux_aarch64.whl
  810  python3 -m pip install --upgrade pip; python3 -m pip install numpy==’1.26.1’; python3 -m pip install --no-cache $TORCH_INSTALL
  811  pip3 install --no-cache https://developer.download.nvidia.com/compute/redist/jp/v51/pytorch/
  812  pip3 install torch  https://developer.download.nvidia.com/compute/redist/jp/v51/pytorch/
  813  ls
  814  pip3 install torch-2.0.0a0+8aa34602.nv23.03-cp38-cp38-linux_aarch64.whl 
  815  pip3 uninstall torchvision 
  816  pip3 install torchvision==0.16.0
  817  pip3 uninstall torch torchvision torchaudio
  818  git clone https://github.com/pytorch/vision
  819  pip3 install --pre torch torchvision torchaudio --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v50/pytorch/
  820  ls
  821  pwd
  822  cat ~/.bashrc
  823  pwd
  824  vim ~/.bashrc
  825  source ~/.bash\
  826  source ~/.bash
  827  source ~/.bashrc
  828  python3 python_display_cuda_is_availbel.py 
  829  robot
  830  reboot
  831  pwd
  832  history
  833  tree
  834  tree
  835  nvcc --version
  836  python3 --version
  837  python3 -c "import torch; print(torch.__version__)"
  838  pip3 list
  839  cat /usr/include/cudnn_version.h | grep CUDNN_MAJOR -A 2
  840  python3
  841  jetson_relase
  842  jetson_release
  843  pip install torch==1.12.0+cu116 torchvision==0.13.0+cu116 torchaudio==0.12.0 -f https://download.pytorch.org/whl/torch_stable.html
  844  pip install torch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 -f https://download.pytorch.org/whl/torch_stable.html
  845  python3 python_AI_built_models1.py 
  846  ls
  847  python3 python_load_model_prediction.py 
  848  clear
  849  python3 python_load_model_prediction.py 
  850  sudo jtop
  851  pip3 install torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64(1).whl
  852  pip3 install torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch6-1.whl
  853  pip4 install torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64-1.whl
  854  pip3\ install torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64-1.whl
  855  pip3 install torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64-1.whl
  856  pip3 install torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64.whl
  857  ls
  858  python3 python_display_cuda_is_availbel.py 
  859  python3 python_load_model_prediction.py 
  860  python3 --version
  861  jetson_release
  862  torch.version.cuda
  863  nvcc --version
  864  cat /usr/include/cudnn_version.h | grep CUDNN_MAJOR -A 2
  865  python3 

```

 
# 7 总结
好长时间了，终于解决了吧。

