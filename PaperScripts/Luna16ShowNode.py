import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image
import pydicom
import pydicom as dicom
"""
绘制结节位置框
"""
def normalize_hu(image):
	#将输入图像的像素值(-4000 ~ 4000)归一化到0~1之间
    MIN_BOUND = -1000.0
    MAX_BOUND = 400.0
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image > 1] = 1.
    image[image < 0] = 0.
    return image

def pc(img,pt,r):
    u,v=img.shape[:2]
    cv2.circle(img,(pt[0],pt[1]),r,(255,255,255),2)
    return img

def show_nodules(ct_scan, nodules,Origin,Spacing,radius=20, pad=2, max_show_num=4):
    # radius是正方形边长一半，pad是边的宽度,max_show_num最大展示数
    show_index = []
    for idx in range(nodules.shape[0]):# lable是一个nx4维的数组，n是肺结节数目，4代表x,y,z,以及直径
        if idx < max_show_num:
            if abs(nodules[idx, 0]) + abs(nodules[idx, 1]) + abs(nodules[idx, 2]) + abs(nodules[idx, 3]) == 0:
                continue
            # 转换坐标
            x = int((nodules[idx, 0]-Origin[0])/SP[0])
            y = int((nodules[idx, 1]-Origin[1])/SP[1])
            z = int((nodules[idx, 2]-Origin[2])/SP[2])
            # 转换半径(注意数据集提供的是单位为mm的直径)
            radius=int(nodules[idx, 3]/SP[0]/2)
            print(x, y, z,radius)  # 坐标转换完成

            data = ct_scan[z]
            if z in show_index:  # 检查是否有结节在同一张切片，如果有，只显示一张
                continue

            show_index.append(z)
            plt.figure(idx)

            data2 = pc(data, [x, y], radius + 5)
            plt.imshow(data2,cmap='gray')#,
    plt.show()


if __name__ == '__main__':
    filename = 'H:/LUNA16/subsets/subset4/1.3.6.1.4.1.14519.5.2.1.6279.6001.107351566259572521472765997306.mhd'
    itkimage = sitk.ReadImage(filename)  # 读取.mhd文件
    OR = itkimage.GetOrigin()
    print(OR)
    SP = itkimage.GetSpacing()
    print(SP)
    numpyImage = sitk.GetArrayFromImage(itkimage)  # 获取数据，自动从同名的.raw文件读取
    b = np.array([[122.07891990,-175.8341133,-193.8795594,5.084404982],
[101.93227340,-178.4539814,-222.7930698,6.644741170],
[-46.78372860,-66.97369400,-207.4908964,6.214452054],
[-69.12656791,-81.56714288,-189.8066129,8.648009287],
[-118.0728115,-147.1544803,-186.0432849,7.185341553],
[82.229180830,-82.37619066,-177.8976950,6.140896043]])
    show_nodules(numpyImage, b, OR, SP)

'''
subset2
1.3.6.1.4.1.14519.5.2.1.6279.6001.100621383016233746780170740405	

([[-24.0138242,192.1024053,-391.0812764,8.143261683],
[2.441546798,172.4648812,-405.4937318,18.54514997],
[90.93171321,149.0272657,-426.5447146,18.20857028],
[89.54076865,196.4051593,-515.0733216,16.38127631],
[81.50964574,54.95721860,-150.3464233,10.36232088]])

subset1
1.3.6.1.4.1.14519.5.2.1.6279.6001.173106154739244262091404659845
([[-115.2874457, 24.16102581, -124.619925, 10.88839157],
[-113.1930507, -1.264504521, -138.6984478, 16.39699158],
[72.77454834, 37.27831567, -118.3077904, 8.548347161]])

subset4
1.3.6.1.4.1.14519.5.2.1.6279.6001.107351566259572521472765997306	
([[122.07891990,-175.8341133,-193.8795594,5.084404982],	
[100.93227340,-179.4539814,-222.7930698,5.644741170],	
[-46.78372860,-66.97369400,-207.4908964,4.214452054],	
[-69.12656791,-80.56714288,-189.8066129,8.648009287],	
[-108.0728115,-147.1544803,-186.0432849,7.285341553],	
[82.229180830,-82.37619066,-177.8976950,6.140896043]])
'''