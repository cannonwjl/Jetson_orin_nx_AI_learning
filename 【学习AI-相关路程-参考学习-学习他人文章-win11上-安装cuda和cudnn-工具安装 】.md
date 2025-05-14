@[TOC](【学习AI-相关路程-参考学习-学习他人文章-win11上-安装cuda和cudnn-工具安装 】)

# 1、 前言

之前在Linux下也就是Ubuntu 20.04下，安装过cuda环境。
[【学习AI-相关路程-工具使用-自我学习-NVIDIA-cuda-工具安装 （1）】](https://blog.csdn.net/qq_22146161/article/details/138182509?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522172363230316800172567047%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fblog.%2522%257D&request_id=172363230316800172567047&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~blog~first_rank_ecpm_v1~rank_v31_ecpm-1-138182509-null-null.nonecase&utm_term=cuda&spm=1018.2226.3001.4450)
同时也在jetson orin NX上安装了，如下文章

[【学习AI-相关路程-工具使用-自我学习-jetson&cuda&pytorch-开发工具尝试-基础样例 （3）】](https://blog.csdn.net/qq_22146161/article/details/138664922?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522172363230316800172567047%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fblog.%2522%257D&request_id=172363230316800172567047&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~blog~first_rank_ecpm_v1~rank_v31_ecpm-2-138664922-null-null.nonecase&utm_term=cuda&spm=1018.2226.3001.4450)
# 2、 概念说明
## 1-使用软件时，先想明白为什么用。
凡事要想明白，你为何这样做，有什么目的，自己带着主观能动性，要明白自己为啥做这样事情，有时候可能更重要，我们在win上安装cuda，是想要借用它并行出来能力，往小了说，我们学习AI相关知识，正好cuda平台可以用，这是一个工具，往大了说，是要完成相关工作。
## 2-了解cuda
 CUDA（Compute Unified Device Architecture）是由NVIDIA开发的一种并行计算平台和编程模型，允许开发者使用GPU（图形处理单元）来加速计算任务。GPU原本是为图形渲染设计的，但由于其高度并行的架构，也非常适合处理大量相似的数据任务。CUDA的主要作用包括以下几个方面：
1. 加速计算密集型任务
CUDA可以将计算密集型任务分配到GPU上进行处理，而不是传统地使用CPU。GPU拥有大量的计算核心，能够同时执行成千上万的线程，从而显著加快处理速度。这在科学计算、图像处理、深度学习、模拟仿真等领域非常有用。
2. 并行处理
CUDA使得开发者能够利用GPU的并行计算能力来处理大量并行任务。比如在深度学习中，训练神经网络需要大量的矩阵运算，这些运算非常适合在GPU上并行处理，从而加快训练速度。
3. 编程灵活性
CUDA提供了一种扩展的C语言编程环境，允许开发者编写能够在GPU上运行的并行代码。通过CUDA，开发者可以更灵活地控制GPU的资源，优化代码性能。
4. 广泛应用
CUDA已经成为深度学习和AI领域的事实标准。许多流行的深度学习框架（如TensorFlow、PyTorch等）都提供了对CUDA的支持，使得在这些框架上运行的模型能够充分利用GPU加速。
5. 支持大规模并行运算
CUDA能够支持大规模并行运算，这对于处理大数据集和复杂的仿真非常重要。例如，在高性能计算（HPC）中，CUDA常用于求解科学和工程问题，如气候模拟、分子动力学等。
6. 优化现有应用
现有的应用程序可以通过CUDA进行优化，将部分计算密集型代码移植到GPU上，从而提高性能。例如，视频处理软件可以通过CUDA加速视频解码和编码过程。



## 3-了解cuDNN
简单说这是一个加速库，在其他文章里也说过。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/e79b09e50ea74f86aa4f9efa8d6b73cf.png)
cuDNN（CUDA Deep Neural Network library）是由NVIDIA开发的一个用于深度学习的GPU加速库。它专门优化了卷积神经网络（CNN）的性能，并提供了高度优化的GPU操作以支持各种深度学习框架。以下是cuDNN的主要特点和作用：
1. 高效的深度学习运算
cuDNN 提供了一组高度优化的低级别函数，专门用于加速深度学习模型中的常见操作，如卷积、池化、归一化和激活等。它对这些操作进行了专门优化，特别是在大规模的神经网络训练和推理中，可以显著提高性能。
2. 深度学习框架的支持
cuDNN 被广泛集成到主流的深度学习框架中，如 TensorFlow、PyTorch、Caffe、MXNet 等。这意味着当您使用这些框架时，如果系统中安装了 cuDNN，框架会自动调用 cuDNN 提供的优化函数，从而利用 GPU 来加速深度学习任务。
3. 跨平台支持
cuDNN 可以在不同的操作系统上运行，包括 Windows、Linux 和 macOS。它与 NVIDIA 的 CUDA 平台紧密集成，因此可以利用 NVIDIA GPU 的并行计算能力进行加速。
4. 支持多种网络类型
cuDNN 不仅支持卷积神经网络，还支持其他类型的网络如循环神经网络（RNN）、长短时记忆网络（LSTM）等。它提供了支持这些网络所需的基本构件，并进行了高度优化。
5. 自动化内存管理
cuDNN 自动管理 GPU 内存的分配和优化，开发者不需要手动处理这些低级细节。它能够根据不同的硬件架构动态调整内存使用，以达到最佳的性能。
6. 性能优化
cuDNN 的每个版本都会根据最新的 NVIDIA GPU 进行优化。它通过利用 GPU 的最新架构特性，如 Tensor Cores 等，来进一步提高深度学习任务的执行速度。
7. 精度和性能的平衡
cuDNN 支持多种精度（如 FP32、FP16 等），允许开发者在不同的精度模式下平衡性能和模型精度。特别是在训练和推理时，使用混合精度（如 FP16）可以显著提高计算速度而不显著影响模型的精度。
8. 广泛应用
cuDNN 被广泛应用于许多需要高性能计算的深度学习应用中，如图像分类、目标检测、自然语言处理和生成对抗网络（GANs）等。
总之，cuDNN 是一个专门为深度学习任务设计的 GPU 加速库，通过优化神经网络的常见操作，为深度学习模型提供高效的计算支持。它是深度学习开发者和研究人员在训练和部署大规模深度学习模型时的重要工具。

## 4-整个流程
在安装软件的时候，其实有些东西我们默认不管的，可以说三个系统下

 - jetson orin NX
 - Ubuntu 20.04
 - win11
 
 
 虽然涉及整个流程步骤不完全一致，但是大体上要做的事情是几乎差不多的。
 
 - 系统要求
 - 硬件要求
 - 安装英伟达驱动
 - 安装cuda
 - 安装cuDNN
 - 配置环境变量
 - 验证安装
 - 软件依赖和开发软件
 
 其中前三项，一般win下买英伟达显卡，都是为了打游戏的，基本都是适配的。即使不说，大家自己也早已熟悉了。
 
 至于最后的软件依赖进行编程等等，本次先不说了，不过根据自己了解，需要使用Anaconda这种工具，正在学习中。。。

# 3、参考文章-致谢
首先还是非常感谢博主的文章，有个参照，并且一尝试，就直接可以了。
参考链接：[https://blog.csdn.net/YYDS_WV/article/details/137825313](https://blog.csdn.net/YYDS_WV/article/details/137825313)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/60b8258f6a80461080bb123d0f714e6c.png)

# 4、实验步骤
## 1、安装cuda
### 1-查看电脑适合的cuda
（1）安装好了英伟达驱动后，在终端中，使用如下指令，这步目的是为了知道自己电脑可以安装什么版本的cuda。

```shell
nvidia-smi
```
如下自己电脑显示的。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/746754adb83d4499bf42eb3ad547b321.png)
当然你说不会使用终端，那么使用图像界面工具也是可以的。

（2）在搜索中输入“NVIDIA Control Panel”，
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/0c6e6ae926044e4ba077395cf63eb092.png)

```shell
NVIDIA Control Panel
```
基本会看到如上软件。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/2cd19bae078a451cac2894df357759d2.png)
然后在帮助内的==“系统信息”==里找到组件，可以查看
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/8b4b520f75f5439498810f1c06a98aba.png)


（3）通过上述两种方式，确定cuda版本。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/6fdcd6becdee462ab7783b3ec2cb62a2.png)
如上图，就可以确定是12.2.1了

### 2-去英伟达官网下载cuda
（1）确定版本后，我们去官网，当然，需要先注册账号，你要是喜欢打游戏，那个账号是一样的，就不需要注册了，用那个账号是通用的。
官网链接：[https://developer.nvidia.com/cuda-toolkit-archive](https://developer.nvidia.com/cuda-toolkit-archive)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/aab99d22e8b843bbad48cd1ad9e8196d.png)
如下图，选择自己之前确认的cuda版本。

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/cb4a96c4067949208bf2254bdaffc9b5.png)

（2）选择自己对应的版本，我这里因为是12.2cuda，所以如下选择。我们直接选择exe，直接下载下来，进行安装。

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/eeb1bebb27aa4d7db038e504e32e2243.png)
可以看到大概是3G左右，点击“Download(3.0GB)”
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/74fa048bf22242b7b61615abdc08a594.png)
（3）下载完毕的包如下
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/b8944099172e40028d36661f367b17ba.png)
### 3-双击安装
（1）安装最好默认，据参考博主说，安装其他目录失败了，那么最好默认安装，当然如果选择其他目录，也可以尝试下，看评论说有成功的。

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/a7ffc490aa46423faf56367ed2f1d002.png)
（2）检查系统兼容性
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/ea6691de4d7a4e5bbf0d7458794ff8fa.png)
（3）和参考博文一样，选择自定义安装
![h](https://i-blog.csdnimg.cn/direct/efd9c66fb6b8402494d9b91816a38741.png)
（5）安装的组件，cuda是必装的。其他可以了解下。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/eaddeb967b4c48b9b43f247d5f12b164.png)
（6）安装位置，最好留意下，后续要配置环境变量
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/1695801caa904648ad016fe6991bca15.png)
（7）安装过程
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/6a7a1ceec11b433dbc84863af04d5fa6.png)
（8）安装完成
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/d4f10cfdfd9f474abc3b5492078a9e4f.png)
### 4-验证cuda
安装完成后，使用如下命令进行验证。

```shell
nvcc -V
```

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/66758ab5739d49fe8edc56195b6374c4.png)


### 5-确认cuda环境变量
（1）在搜索中输入“编辑环境变量”，进行相关环境变量编辑，如下图所示。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/68d7394f294c498bbe64121bd723db09.png)
（2）确认已有的环境变量，如下图为软件自动安装后，有了的cuda环境变量。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/aaf0c36046674a7993aeea69a209494f.png)


## 2、安装cuDNN
那么我们接下来下载并复制cuDNN。
### 1-去英伟达官网寻找cuDNN
链接：[https://developer.nvidia.com/rdp/cudnn-archive](https://developer.nvidia.com/rdp/cudnn-archive)
如下图，我们寻找一个相配的版本，如果后续不可用，我们可能需要再回来找相关应该版本。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/1bed4d2ec1804da3953327bc184e3e2c.png)

### 2-去英伟达官网下载cuDNN
我们直接选择对应的版本的zip。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/84e686a247504dbabbbd8b26bc989bc7.png)
### 3-复制到对应目录下
我们下载这个，主要是复制到对应目录下。

```file
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2\lib\x64
```
如下图，将解压的文件复制这个目录下。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/196349a103224c55b28f1cc165421621.png)


## 3、配置环境变量
之后就是配置环境变量了，
### 1-打开环境变量
（1）在搜索中输入“编辑环境变量”，进行相关环境变量编辑，如下图所示。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/68d7394f294c498bbe64121bd723db09.png)
（2）确认已有的环境变量，如下图为软件自动安装后，有了的cuda环境变量。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/aaf0c36046674a7993aeea69a209494f.png)
如上，是我们刚才确认的，然后新建环境变量。
### 2-新建环境变量-相关说明
==这里我自己应该犯了一个错误，我不确定是不是应该编辑Path，还是说，直接新建。不过最后过了，如果到时候有问题在改吧。==
我这里直接新建了，个人觉得也是可以的，添加进来，就好，自己用win11也是不长时间，所以看到博主的文章就很奇怪，直接新建了，其实应该是编辑，Path，如下图那样。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/8fe2c1e6b372425daaa6f6e8ac8fb2fb.png)
### 3-新建环境变量-我自己尝试
下图直接点击“新建”
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/27eb257e92f04cf287fa228e1f5b59b1.png)
就会出如下。

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/0fc60a660f2d48ef81f4f8ef30ad85aa.png)
起个名字之后，分别编辑另几个。


![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/c86fdc5702c1424086a260dc4b0763b4.png)


## 4、验证安装结果
（1）在如下目录下，使用终端运行两个exe文件，

```shell
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2\extras\demo_suite
```

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/ceaa3f8ab97b44e6aa52f5bb5e68fadd.png)
如上图所示，右键点开终端

或者

搜索中输入CMD,然后使用如下目录进入相关目录

```shell
cd /d C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2\extras\demo_suite
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/27c3e73e1d66487bb3da53ec1a4e3bb2.png)
（2）进入目录后，运行exe文件，如下为运行.\deviceQuery.exe

```shell
.\deviceQuery.exe
```

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/42891b0c4ea54957a41d696cf5ac5cd6.png)
如下为运行软件 .\bandwidthTest.exe
```shell
 .\bandwidthTest.exe
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/754b0572bd5e460fa1453464babcad43.png)
最后都会有什么pass字样为成功。
```shell
Windows PowerShell
版权所有（C） Microsoft Corporation。保留所有权利。

安装最新的 PowerShell，了解新功能和改进！https://aka.ms/PSWindows

PS C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2\extras\demo_suite> .\bandwidthTest.exe
[CUDA Bandwidth Test] - Starting...
Running on...

Device 0: NVIDIA GeForce GTX 1650 Ti with Max-Q Design
Quick Mode

Host to Device Bandwidth, 1 Device(s)
PINNED Memory Transfers
   Transfer Size (Bytes)        Bandwidth(MB/s)
   33554432                     12663.0

Device to Host Bandwidth, 1 Device(s)
PINNED Memory Transfers
   Transfer Size (Bytes)        Bandwidth(MB/s)
   33554432                     12066.7

Device to Device Bandwidth, 1 Device(s)
PINNED Memory Transfers
   Transfer Size (Bytes)        Bandwidth(MB/s)
   33554432                     31088.3

Result = PASS

NOTE: The CUDA Samples are not meant for performance measurements. Results may vary when GPU Boost is enabled.
PS C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2\extras\demo_suite> .\deviceQuery.exe
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2\extras\demo_suite\deviceQuery.exe Starting...

CUDA Device Query (Runtime API) version (CUDART static linking)

Detected 1 CUDA Capable device(s)

Device 0: "NVIDIA GeForce GTX 1650 Ti with Max-Q Design"
  CUDA Driver Version / Runtime Version          12.2 / 12.2
  CUDA Capability Major/Minor version number:    7.5
  Total amount of global memory:                 4096 MBytes (4294639616 bytes)
  (16) Multiprocessors, ( 64) CUDA Cores/MP:     1024 CUDA Cores
  GPU Max Clock rate:                            1200 MHz (1.20 GHz)
  Memory Clock rate:                             5001 Mhz
  Memory Bus Width:                              128-bit
  L2 Cache Size:                                 1048576 bytes
  Maximum Texture Dimension Size (x,y,z)         1D=(131072), 2D=(131072, 65536), 3D=(16384, 16384, 16384)
  Maximum Layered 1D Texture Size, (num) layers  1D=(32768), 2048 layers
  Maximum Layered 2D Texture Size, (num) layers  2D=(32768, 32768), 2048 layers
  Total amount of constant memory:               zu bytes
  Total amount of shared memory per block:       zu bytes
  Total number of registers available per block: 65536
  Warp size:                                     32
  Maximum number of threads per multiprocessor:  1024
  Maximum number of threads per block:           1024
  Max dimension size of a thread block (x,y,z): (1024, 1024, 64)
  Max dimension size of a grid size    (x,y,z): (2147483647, 65535, 65535)
  Maximum memory pitch:                          zu bytes
  Texture alignment:                             zu bytes
  Concurrent copy and kernel execution:          Yes with 2 copy engine(s)
  Run time limit on kernels:                     Yes
  Integrated GPU sharing Host Memory:            No
  Support host page-locked memory mapping:       Yes
  Alignment requirement for Surfaces:            Yes
  Device has ECC support:                        Disabled
  CUDA Device Driver Mode (TCC or WDDM):         WDDM (Windows Display Driver Model)
  Device supports Unified Addressing (UVA):      Yes
  Device supports Compute Preemption:            Yes
  Supports Cooperative Kernel Launch:            Yes
  Supports MultiDevice Co-op Kernel Launch:      No
  Device PCI Domain ID / Bus ID / location ID:   0 / 1 / 0
  Compute Mode:
     < Default (multiple host threads can use ::cudaSetDevice() with device simultaneously) >

deviceQuery, CUDA Driver = CUDART, CUDA Driver Version = 12.2, CUDA Runtime Version = 12.2, NumDevs = 1, Device0 = NVIDIA GeForce GTX 1650 Ti with Max-Q Design
Result = PASS
PS C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2\extras\demo_suite>
```

# 5、总结
还是感谢参考文章，如此一来在三个环境下的cuda我们就都有，之后就是在win上运行相关AI的东西了。

