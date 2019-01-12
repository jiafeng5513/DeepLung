1. 查询cuda版本:
    ```bash
    cat /usr/local/cuda/version.txt
    cat /usr/local/cuda/include/cudnn.h | grep CUDNN_MAJOR -A 2
    ```
    CUDA Version 9.0.176
    CUDNN 7
2. 安装anaconda2
    官网下载安装
3. 安装CUDNN
    https://developer.nvidia.com/rdp/cudnn-download
    下载cudnn library for linux,是一个压缩包
    把压缩包里面的文件复制到cuda安装目录的对应文件夹里面
4. simpleitk 1.2.0
5. pytorch 1.0.0
6. torchvision 0.2.1
7. pydicom 1.2.1
8. mahotas 1.4.5
9. nvidia-ml-py 375.53.1
