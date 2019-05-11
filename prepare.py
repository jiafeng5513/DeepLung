#coding:utf-8
import os
from config_training import config
import numpy as np
from scipy.ndimage.interpolation import zoom
import SimpleITK as sitk
from scipy.ndimage.morphology import binary_dilation, generate_binary_structure
from skimage.morphology import convex_hull_image
import pandas
from multiprocessing import Pool
from functools import partial
import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pydicom
import pydicom as dicom
from scipy import ndimage as ndi
from skimage.filters import roberts
from skimage.measure import label, regionprops
from skimage.morphology import disk, binary_erosion, binary_closing
from skimage.segmentation import clear_border

zhfont = mpl.font_manager.FontProperties(fname='/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf')

#重采样,调整像素间隔到一致
def resample(imgs, spacing, new_spacing, order=2):
    """
    重采样,调整像素间隔到一致
    :param imgs: 输入图片
    :param spacing: 原来的像素物理尺寸
    :param new_spacing: 新的像素物理尺寸
    :param order: 样条插值参数[0,5]
    :return:
    """
    if len(imgs.shape) == 3:
        new_shape = np.round(imgs.shape * spacing / new_spacing)
        true_spacing = spacing * imgs.shape / new_shape
        resize_factor = new_shape / imgs.shape
        imgs = zoom(imgs, resize_factor, mode='nearest', order=order)
        return imgs, true_spacing
    elif len(imgs.shape) == 4:
        n = imgs.shape[-1]
        newimg = []
        for i in range(n):
            slice = imgs[:, :, :, i]
            newslice, true_spacing = resample(slice, spacing, new_spacing)
            newimg.append(newslice)
        newimg = np.transpose(np.array(newimg), [1, 2, 3, 0])
        return newimg, true_spacing
    else:
        raise ValueError('wrong shape')

#世界坐标转换为体素坐标
def worldToVoxelCoord(worldCoord, origin, spacing):
    """
    世界坐标转换为体素坐标
    :param worldCoord: 世界坐标(单位mm)
    :param origin: 原点
    :param spacing: 像素的物理尺寸(每像素是多少毫米)
    :return:
    """
    stretchedVoxelCoord = np.absolute(worldCoord - origin)
    voxelCoord = stretchedVoxelCoord / spacing
    return voxelCoord

# 打开itk图片
def load_itk_image(filename):
    """
    itk打开CT图片
    :param filename:
    :return:图片,坐标原点,像素的物理尺寸,是否翻转
    """
    with open(filename) as f:
        contents = f.readlines()
        line = [k for k in contents if k.startswith('TransformMatrix')][0]
        transformM = np.array(line.split(' = ')[1].split(' ')).astype('float')
        transformM = np.round(transformM)
        if np.any(transformM != np.array([1, 0, 0, 0, 1, 0, 0, 0, 1])):
            isflip = True
        else:
            isflip = False

    itkimage = sitk.ReadImage(filename)
    numpyImage = sitk.GetArrayFromImage(itkimage)

    numpyOrigin = np.array(list(reversed(itkimage.GetOrigin())))
    numpySpacing = np.array(list(reversed(itkimage.GetSpacing())))

    return numpyImage, numpyOrigin, numpySpacing, isflip

# 对掩码进行膨胀处理
def process_mask(mask):
    convex_mask = np.copy(mask)
    for i_layer in range(convex_mask.shape[0]):
        mask1 = np.ascontiguousarray(mask[i_layer])
        if np.sum(mask1) > 0:
            mask2 = convex_hull_image(mask1)
            if np.sum(mask2) > 1.5 * np.sum(mask1):
                mask2 = mask1
        else:
            mask2 = mask1
        convex_mask[i_layer] = mask2
    struct = generate_binary_structure(3, 1)
    dilatedMask = binary_dilation(convex_mask, structure=struct, iterations=10)
    return dilatedMask

# 归一化到[0,255]
def lumTrans(img):
    lungwin = np.array([-1200., 600.])
    newimg = (img - lungwin[0]) / (lungwin[1] - lungwin[0])
    newimg[newimg < 0] = 0
    newimg[newimg > 1] = 1
    newimg = (newimg * 255).astype('uint8')
    return newimg


def savenpy_luna(id, annos, filelist, luna_segment, luna_data, savepath):
    """
    处理一个子集
    :param id: 当前子集中的CT切片数量
    :param annos: annotations.csv中的数据，真肺结节的xyzd
    :param filelist: 当前子集中的所有mhd文件
    :param luna_segment: 分割mask
    :param luna_data: 当前子集所在路径
    :param savepath: 保存路径
    :return:
    """
    islabel = True
    isClean = True
    resolution = np.array([1, 1, 1])
    #     resolution = np.array([2,2,2])
    name = filelist[id]
    # 加载原始CT
    sliceim, origin, spacing, isflip = load_itk_image(os.path.join(luna_data, name + '.mhd'))
    # 加载分割mask
    Mask, origin, spacing, isflip = load_itk_image(os.path.join(luna_segment, name + '.mhd'))
    if isflip:
        Mask = Mask[:, ::-1, ::-1]
    newshape = np.round(np.array(Mask.shape) * spacing / resolution).astype('int')
    m1 = Mask == 3
    m2 = Mask == 4
    Mask = m1 + m2

    xx, yy, zz = np.where(Mask)
    box = np.array([[np.min(xx), np.max(xx)], [np.min(yy), np.max(yy)], [np.min(zz), np.max(zz)]])
    box = box * np.expand_dims(spacing, 1) / np.expand_dims(resolution, 1)
    box = np.floor(box).astype('int')
    margin = 5
    extendbox = np.vstack(
        [np.max([[0, 0, 0], box[:, 0] - margin], 0), np.min([newshape, box[:, 1] + 2 * margin], axis=0).T]).T

    this_annos = np.copy(annos[annos[:, 0] == (name)])

    if isClean:
        convex_mask = m1
        dm1 = process_mask(m1)
        dm2 = process_mask(m2)
        dilatedMask = dm1 + dm2
        Mask = m1 + m2

        extramask = dilatedMask ^ Mask
        bone_thresh = 210
        pad_value = 170

        if isflip:
            sliceim = sliceim[:, ::-1, ::-1]
            print('flip!')
        sliceim = lumTrans(sliceim)
        sliceim = sliceim * dilatedMask + pad_value * (1 - dilatedMask).astype('uint8')
        bones = (sliceim * extramask) > bone_thresh
        sliceim[bones] = pad_value

        sliceim1, _ = resample(sliceim, spacing, resolution, order=1)
        sliceim2 = sliceim1[extendbox[0, 0]:extendbox[0, 1],
                   extendbox[1, 0]:extendbox[1, 1],
                   extendbox[2, 0]:extendbox[2, 1]]
        sliceim = sliceim2[np.newaxis, ...]

        np.save(os.path.join(savepath, name + '_clean.npy'), sliceim)
        np.save(os.path.join(savepath, name + '_spacing.npy'), spacing)
        np.save(os.path.join(savepath, name + '_extendbox.npy'), extendbox)
        np.save(os.path.join(savepath, name + '_origin.npy'), origin)
        np.save(os.path.join(savepath, name + '_mask.npy'), Mask)

    if islabel:
        this_annos = np.copy(annos[annos[:, 0] == (name)])
        label = []
        if len(this_annos) > 0:

            for c in this_annos:
                pos = worldToVoxelCoord(c[1:4][::-1], origin=origin, spacing=spacing)
                if isflip:
                    pos[1:] = Mask.shape[1:3] - pos[1:]
                label.append(np.concatenate([pos, [c[4] / spacing[1]]]))

        label = np.array(label)
        if len(label) == 0:
            label2 = np.array([[0, 0, 0, 0]])
        else:
            label2 = np.copy(label).T
            label2[:3] = label2[:3] * np.expand_dims(spacing, 1) / np.expand_dims(resolution, 1)
            label2[3] = label2[3] * spacing[1] / resolution[1]
            label2[:3] = label2[:3] - np.expand_dims(extendbox[:, 0], 1)
            label2 = label2[:4].T
        np.save(os.path.join(savepath, name + '_label.npy'), label2)

    print(name)


def preprocess_luna():
    luna_segment = config['luna_segment']
    savepath = config['preprocess_result_path']
    luna_data = config['luna_data']
    luna_label = config['luna_label']
    finished_flag = '.flag_preprocessluna'
    print('starting preprocessing luna')
    if not os.path.exists(finished_flag):
        annos = np.array(pandas.read_csv(luna_label))
        pool = Pool()
        if not os.path.exists(savepath):
            os.mkdir(savepath)
        for setidx in xrange(10):
            print 'process subset', setidx
            filelist = [f.split('.mhd')[0] for f in os.listdir(luna_data+'subset'+str(setidx)) if f.endswith('.mhd') ]
            if not os.path.exists(savepath+'subset'+str(setidx)):
                os.mkdir(savepath+'subset'+str(setidx))
            # 构造调用savenpy_luna函数的partial对象
            partial_savenpy_luna = partial(savenpy_luna, annos=annos, filelist=filelist,
                                       luna_segment=luna_segment, luna_data=luna_data+'subset'+str(setidx)+'/', 
                                       savepath=savepath+'subset'+str(setidx)+'/')
            N = len(filelist)
            #savenpy(1)
            _=pool.map(partial_savenpy_luna,range(N))
        pool.close()
        pool.join()
    print('end preprocessing luna')
    f= open(finished_flag,"w+")


def TestPreprocess(img,plot=False):
    if plot == True:
        f, plots = plt.subplots(2, 4)
    '''
    Step 1: 以604(HU=400)为分界点二值化
    '''
    binary = img < 604
    if plot == True:
        plt.subplot(2,4,1)
        plt.axis('off')
        plt.title(u"二值化", fontproperties=zhfont)
        plt.imshow(binary, cmap=plt.cm.bone)
    '''
    Step 2: 移除与边界相连的部分
    '''
    cleared = clear_border(binary)
    if plot == True:
        plt.subplot(2, 4, 2)
        plt.axis('off')
        plt.title(u"移除边界", fontproperties=zhfont)
        plt.imshow(cleared, cmap=plt.cm.bone)
    '''
    Step 3: 标记连通区域
    '''
    label_image = label(cleared)
    if plot == True:
        plt.subplot(2, 4, 3)
        plt.axis('off')
        plt.title(u"标记联通区域", fontproperties=zhfont)
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
        plt.subplot(2, 4, 4)
        plt.axis('off')
        plt.title(u"保留最大的两个区域", fontproperties=zhfont)
        plt.imshow(binary, cmap=plt.cm.bone)
    '''
    Step 5: 半径为2的腐蚀操作,分离附着于血管的肺结节
    '''
    selem = disk(2)
    binary = binary_erosion(binary, selem)
    if plot == True:
        plt.subplot(2, 4, 5)
        plt.axis('off')
        plt.title(u"腐蚀", fontproperties=zhfont)
        plt.imshow(binary, cmap=plt.cm.bone)
    '''
    Step 6: 半径为10的闭操作,合并粘在肺壁上的结节
    '''
    selem = disk(10)
    binary = binary_closing(binary, selem)
    if plot == True:
        plt.subplot(2, 4, 6)
        plt.axis('off')
        plt.title(u"闭", fontproperties=zhfont)
        plt.imshow(binary, cmap=plt.cm.bone)
    '''
    Step 7: 填充小洞
    '''
    edges = roberts(binary)#边缘检测,Roberts算子,也可以使用sobel算子
    binary = ndi.binary_fill_holes(edges)#空洞填充
    if plot == True:
        plt.subplot(2, 4, 7)
        plt.axis('off')
        plt.title(u"填充小洞", fontproperties=zhfont)
        plt.imshow(binary, cmap=plt.cm.bone)

    "此时binnary就是最终的掩码了"
    # '''
    # 7.1 非肺部区域绿色,肺部区域蓝色
    # '''
    # # 拷贝一个binnary
    # ColorMask = np.zeros((binary.shape[0],binary.shape[1],3), np.uint8)
    #
    # for i in range(ColorMask.shape[0]):
    #     for j in range(ColorMask.shape[1]):
    #         if binary[i,j] == 0:
    #             ColorMask[i,j]=(0,255,0)
    #         if binary[i,j] == 1:
    #             ColorMask[i, j] = (0, 0, 255)
    #
    # if plot == True:
    #     plt.subplot(3, 4, 9)
    #     plt.axis('off')
    #     plt.title(u"上色", fontproperties=zhfont)
    #     plt.imshow(ColorMask)
    #
    # '''
    # 7.2 ROI描点,连接成封闭区域,填充
    # '''
    # if len(rois)>=1:
    #     for roi in rois:
    #         cv2.fillPoly(ColorMask, [roi], (255, 0, 0))
    #
    # # cv2.polylines(ColorMask, [pts], True, (255, 0, 0), 2)
    # # cv2.fillPoly(ColorMask, [pts], (255, 0, 0))
    # if plot == True:
    #     plt.subplot(3, 4, 10)
    #     plt.axis('off')
    #     plt.title(u"ROI勾画", fontproperties=zhfont)
    #     plt.imshow(ColorMask)
    '''
    Step 8: 使用掩码提取原始图像中的肺区域
    '''
    get_high_vals = binary == 0
    img[get_high_vals] = 0
    if plot == True:
        plt.subplot(2, 4, 8)
        plt.axis('off')
        plt.title(u"使用掩码提取原始数据", fontproperties=zhfont)
        plt.imshow(img, cmap=plt.cm.bone)

    return img


if __name__=='__main__':
    preprocess_luna()
