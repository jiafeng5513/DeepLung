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


if __name__=='__main__':
    preprocess_luna()
