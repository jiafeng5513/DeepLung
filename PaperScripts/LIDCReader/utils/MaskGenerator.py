"""
用于生成标签图
1.颜色定义:体外/体内非肺/肺/结节
2.步骤:体外上色
    2.1.体内肺部区域上色
    2.2.体内非肺部区域上色
    2.3.结节上色
3.
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pydicom
import pydicom as dicom
from scipy import ndimage as ndi
from skimage.filters import roberts
from skimage.measure import label, regionprops
from skimage.morphology import disk, binary_erosion, binary_closing
from skimage.segmentation import clear_border

# 肺分割,第一个参数是一张CT,第二个参数决定是否绘制出处理过程
"""
生成mask并用mask找出原始肺部CT图片中的肺部数据,讲其他部分全部调0
    参数1:pydicom读出来的dcm文件
    参数2:是否绘制过程
"""

"""
使用类似的方法生成着色Label图片
"""
def GenerateLabel(img,rois,plot=False):
    if plot == True:
        f, plots = plt.subplots(3, 4)
    '''
    Step 1: 以604(HU=400)为分界点二值化
    '''
    binary = img < 604
    if plot == True:
        plt.subplot(3,4,1)
        plt.axis('off')
        plt.title(u"二值化", fontproperties='SimHei')
        plt.imshow(binary, cmap=plt.cm.bone)
    '''
    Step 2: 移除与边界相连的部分
    '''
    cleared = clear_border(binary)
    if plot == True:
        plt.subplot(3, 4, 2)
        plt.axis('off')
        plt.title(u"移除边界", fontproperties='SimHei')
        plt.imshow(cleared, cmap=plt.cm.bone)
    '''
    Step 3: 标记连通区域
    '''
    label_image = label(cleared)
    if plot == True:
        plt.subplot(3, 4, 3)
        plt.axis('off')
        plt.title(u"标记联通区域", fontproperties='SimHei')
        plt.imshow(label_image, cmap=plt.cm.bone)
    '''
    Step 4: 只保留两个最大的连通区域
    '''
    areas = [r.area for r in regionprops(label_image)]
    areas.sort()
    if len(areas) > 2:
        for region in regionprops(label_image):
            if region.area < areas[-2]:
                for coordinates in region.coords:
                    label_image[coordinates[0], coordinates[1]] = 0
    binary = label_image > 0
    if plot == True:
        plt.subplot(3, 4, 4)
        plt.axis('off')
        plt.title(u"保留最大的两个区域", fontproperties='SimHei')
        plt.imshow(binary, cmap=plt.cm.bone)
    '''
    Step 5: 半径为2的腐蚀操作,分离附着于血管的肺结节
    '''
    selem = disk(2)
    binary = binary_erosion(binary, selem)
    if plot == True:
        plt.subplot(3, 4, 5)
        plt.axis('off')
        plt.title(u"腐蚀", fontproperties='SimHei')
        plt.imshow(binary, cmap=plt.cm.bone)
    '''
    Step 6: 半径为10的闭操作,合并粘在肺壁上的结节
    '''
    selem = disk(10)
    binary = binary_closing(binary, selem)
    if plot == True:
        plt.subplot(3, 4, 6)
        plt.axis('off')
        plt.title(u"闭", fontproperties='SimHei')
        plt.imshow(binary, cmap=plt.cm.bone)
    '''
    Step 7: 填充小洞
    '''
    edges = roberts(binary)#边缘检测,Roberts算子,也可以使用sobel算子
    binary = ndi.binary_fill_holes(edges)#空洞填充
    if plot == True:
        plt.subplot(3, 4, 7)
        plt.axis('off')
        plt.title(u"填充小洞", fontproperties='SimHei')
        plt.imshow(binary, cmap=plt.cm.bone)

    "此时binnary就是最终的掩码了"
    '''
    7.1 非肺部区域绿色,肺部区域蓝色
    '''
    # 拷贝一个binnary
    ColorMask = np.zeros((binary.shape[0],binary.shape[1],3), np.uint8)

    for i in range(ColorMask.shape[0]):
        for j in range(ColorMask.shape[1]):
            if binary[i,j] == 0:
                ColorMask[i,j]=(0,255,0)
            if binary[i,j] == 1:
                ColorMask[i, j] = (0, 0, 255)

    if plot == True:
        plt.subplot(3, 4, 9)
        plt.axis('off')
        plt.title(u"上色", fontproperties='SimHei')
        plt.imshow(ColorMask)

    '''
    7.2 ROI描点,连接成封闭区域,填充
    '''
    if len(rois)>=1:
        for roi in rois:
            cv2.fillPoly(ColorMask, [roi], (255, 0, 0))

    # cv2.polylines(ColorMask, [pts], True, (255, 0, 0), 2)
    # cv2.fillPoly(ColorMask, [pts], (255, 0, 0))
    if plot == True:
        plt.subplot(3, 4, 10)
        plt.axis('off')
        plt.title(u"ROI勾画", fontproperties='SimHei')
        plt.imshow(ColorMask)
    '''
    Step 8: 使用掩码提取原始图像中的肺区域
    '''
    get_high_vals = binary == 0
    img[get_high_vals] = 0
    if plot == True:
        plt.subplot(3, 4, 8)
        plt.axis('off')
        plt.title(u"使用掩码提取原始数据", fontproperties='SimHei')
        plt.imshow(img, cmap=plt.cm.bone)

    return img,ColorMask





if __name__ == '__main__':

    #url = "../Demo/000041.dcm"  # 有结节,而且结节与肺的外部相通
    #url = "../Demo/000047.dcm"  # 没有结节
    url = "../Demo/000080.dcm"  # 有结节,结节与肺的外部不相同
    pix = dicom.read_file(url).pixel_array
    print(pydicom.read_file(url).SOPInstanceUID)

    pts = np.array([[312,346],[311,347],[311,348],[311,349],[310,349],[309,349],[308,350],[308,351],[308,352],[307,353],
                     [306,354],[306,355],[306,356],[306,357],[306,358],[305,358],[304,359],[303,360],[303,361],[302,361],
                     [301,361],[300,362],[299,363],[299,364],[299,365],[299,366],[299,367],[299,368],[300,369],[300,370],
                     [300,371],[299,372],[298,372],[297,373],[298,374],[298,375],[299,376],[300,376],[300,377],[300,378],
                     [300,379],[301,380],[301,381],[302,382],[303,383],[304,383],[305,383],[306,382],[306,381],[307,381],
                     [308,381],[309,381],[310,381],[311,381],[312,381],[313,381],[314,382],[315,382],[315,383],[316,384],
                     [317,385],[318,386],[319,386],[320,387],[321,386],[322,385],[322,384],[322,383],[323,382],[324,382],
                     [325,381],[325,380],[325,379],[324,378],[324,377],[324,376],[324,375],[324,376],[325,377],[326,378],
                     [327,379],[328,379],[329,378],[329,377],[329,376],[329,375],[330,374],[331,374],[332,374],[333,373],
                     [334,372],[334,371],[334,370],[334,369],[334,368],[334,367],[333,366],[333,365],[333,364],[333,363],
                     [332,362],[331,362],[330,362],[329,362],[328,362],[328,361],[329,360],[329,359],[328,358],[328,357],
                     [328,356],[328,355],[327,354],[326,353],[325,353],[324,353],[323,353],[322,353],[321,353],[320,353],
                     [319,352],[319,351],[318,350],[317,350],[317,349],[316,348],[315,347],[314,347],[313,347],[312,346]],
                   dtype=np.int32)
    ROIS=[]
    ROIS.append(pts)


    GenerateLabel(pix, ROIS,True)
    plt.show()

