@[TOC](问题：OSError: [Errno 22] Invalid argument: 'xx:xx'-解决方式-在window下-调用python-转义字符相关-记录)
# 1、背景
在自己学习mnist手写体识别的时候，准备在window下实现一下，但是在运行python文件的时候，遇到如下报错。上网查过后，最后解决了，觉得有必要记录下来。

# 2、问题描述：OSError: [Errno 22] Invalid argument: 'xx:xx'
如下，是我本次遇到的问题，这个问题虽然是在运行mnist分类python脚本，==但是本质上，这其实是属于python找不到目标文件的问题==。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/635554f3cf7c414fba97ee0614c998ad.png)

经常在不同环境运行脚本，或者项目目录的人，其实很有可能遇到这样的问题，一般很容易想到使用转义字符。

这个问题是由于路径字符串中的反斜杠 `\` 导致的。在 Windows 中，反斜杠是路径分隔符，但在 Python 中需要进行转义，因此通常使用双反斜杠 `\\` 或者前面加上 `r` 来表示原始字符串。可以按照以下方法做，也是三种方式供你选择：



#  3、解决方式1：转义字符


1. 修改路径字符串：
简单说在路径前加入“\”标准转义，实际路径请以自己路径为准。
   ```python
   image_path = "D:\\py_work\\mnist_ai_bp\\AI_test_mnist_demo_BP\\test_image2\\test_iamge_ai\\testNum8.jpg"
   ```

  



#  4、解决方式2：正斜杠
2. 另一种方法是使用正斜杠 `/`，Python 也支持它来作为路径分隔符：
自己在实际使用中，想到既然转义不行，是否也能识别相关正斜杠，也看到有说这样的方式的。
   ```python
   image_path = "D:/py_work/mnist_ai_bp/AI_test_mnist_demo_BP/test_image2/test_iamge_ai/testNum8.jpg"
   ```
#  5、解决方式3：前面加上 `r` 
 3.  使用原始字符串：就是window路径治具复制，但是在路径前加入“r”,也是一种方式。

   ```python
   image_path = r"D:\py_work\mnist_ai_bp\AI_test_mnist_demo_BP\test_image2\test_iamge_ai\testNum8.jpg"
   ```
# 6、我的努力过程

## （1）参考
自己也查了网上相关链接，但是没成，按文章应该是一种解决方法，其实对应这里第一种解决方式。

[python中invalid argument_Python文件报错OSError：\[Errno 22\] Invalid argument处理](https://blog.csdn.net/weixin_39746869/article/details/109924418?ops_request_misc=%257B%2522request%255Fid%2522%253A%25225A1F500D-8669-448C-AFC3-D0D75CE3725A%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fblog.%2522%257D&request_id=5A1F500D-8669-448C-AFC3-D0D75CE3725A&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~blog~first_rank_ecpm_v1~rank_v31_ecpm-2-109924418-null-null.nonecase&utm_term=OSError:%20%5BErrno%2022%5D%20Invalid%20argument:%20D:%5C%5Cpy_work%5C%5Cmnist_ai_bp%5C%5CAI_test_mnist_demo_BP%5Ctest_image2%5Ctest_iamge_ai%5CtestNum8.jpg&spm=1018.2226.3001.4450)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/6be179f50a744b70bbaae256268a1d73.png)


## （2）我的尝试
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/3f3776b4d2874e008fad162c198f62c6.png)
 - 1、如上图所示，其实目标文件放在和python脚本文件一样的目录，==也是可以的==，但是这不是我们所追求的。
 - 2、其中中间也尝试了很多种，但是都没有成功。

 - 3、其中最后一种，是绝对路径，就是在前面加“r”解决的。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/b1abfac15fe644ae9e7f41823497add3.png)
# 7、总结

一点点积累吧
