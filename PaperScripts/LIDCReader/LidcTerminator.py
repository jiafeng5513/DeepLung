"""
终结这个讨厌的问题

1. 遍历patients目录,对于每个patient:
    1.1 遍历study目录,对于每个study:
        1.1.1 遍历series目录,对于每个series:
            1.1.1.1 遍历所有文件
            1.1.1.2 找出xml文件,把dcm文件名存到一个列表里面
            1.1.1.3 把dcm文件排序,把排序结果存在一个文件中,保存在当前目录下
            1.1.1.4 构造xml对象
            1.1.1.5 遍历dcm文件列表,对于每个dcm文件:
                1.1.1.5.1 到xmlobj中查询,该dcm文件有没有被标注,如果有,构造一个数组,存储所有对这张图标注的ROI
                1.1.1.5.2 生成该dcm对应的mask和label,存储在输出目录中

"""
import os
import sys
import cv2 as cv
import numpy as np
import pydicom
from PIL import Image
#from progressbar import *
from Model.XmlData import XmlLabelForCT, XmlLabelSlim
from utils.MaskGenerator import GenerateLabel

"""
常量定义
"""
# 输出目录
OutPutDIR = "G:/TCIA_LIDC-IDRI/LIDC-Sample-Output"

# ROI阈值,取值是0或1,取0的时候输出全部数据的label和分割示意,不管是否带有结节
# 取1的时候输出带有结节的label和分割示意
kv=0

"""
为DcmList按照SliceLocation排序 
"""
def SortDcmListBySliceLocation(dcmlist,XmlFilePath):
    print("按照SliceLocation字段对dcm文件列表进行排序,待排序列表长度为:" + str(len(dcmlist)))
    flag = 1
    for index in range(len(dcmlist) - 1, 0, -1):
        if flag:
            flag = 0
            for two_index in range(index):
                ds1 = pydicom.read_file(dcmlist[two_index]).SliceLocation
                ds2 = pydicom.read_file(dcmlist[two_index + 1]).SliceLocation
                if ds1 < ds2:  # 排序依据是SliceLocation从大到小(基本全是负数)
                    dcmlist[two_index], dcmlist[two_index + 1] = dcmlist[two_index + 1], dcmlist[two_index]
                    flag = 1
        else:
            break
    # 在此处保存排序文件
    SeriesAbsPath = os.path.dirname(dcmlist[0])
    SortFile = open(SeriesAbsPath + "\\DcmList.txt", 'w')  # 创建排序缓存文件

    SortFile.writelines(XmlFilePath.replace("\\", "/") + "\n")  # 先写入xml的路径
    for word in dcmlist:  # 按照解破学顺序把dcm文件路径依次写入txt
        SortFile.writelines(word.replace("\\", "/") + "\n")
    SortFile.close()
    return dcmlist


"""
处理函数
"""
def Terminator(RootDir):
    print ("开始处理...")
    if os.path.exists(RootDir):  # 判断路径是否存在
        patients = os.listdir(RootDir)  # patients
        patientNum = 1
        for patient in patients:
            print("正在处理第{:4d}个patient,还剩下{:4d}个patient".format(patientNum,len(patients)-patientNum))
            patientNum = patientNum+1
            patient_abs_path = os.path.join(RootDir, patient)
            studies = os.listdir(patient_abs_path)
            for study in studies:
                study_abs_path = os.path.join(patient_abs_path, study)
                serieses = os.listdir(study_abs_path)
                for series in serieses:
                    isCTSeries =False  # 当前序列是否是CT序列
                    DcmFileList=[]     # DCM文件列表
                    XmlFilePath=''     # xml文件路径
                    XmlObj = None
                    series_abs_path = os.path.join(study_abs_path, series)
                    itemNames = os.listdir(series_abs_path)  # 获取该目录下的所有文件
                    for lookingFordcm in itemNames:
                        if "dcm" in lookingFordcm:  # 发现dcm文件
                            dcmAbsPath = os.path.join(series_abs_path, lookingFordcm)  # 得到这个DCM文件的绝对路径
                            if pydicom.read_file(dcmAbsPath).Modality == "CT":  # 识别是否为CT序列
                                isCTSeries = True
                            break
                    if isCTSeries == True:
                        # 是CT序列
                        # 判断是否存在排序缓存文件
                        if os.path.exists(series_abs_path + "/DcmList.txt"):
                            # 2.1.1 找到了排序缓存
                            SortFile = open(series_abs_path + "/DcmList.txt", 'r')
                            lines = SortFile.readlines()  # 读取全部缓存内容
                            XmlFilePath = lines[0].replace('\n', '')  # 取出缓存第一行中的xml文件地址
                            XmlObj = XmlLabelForCT(XmlFilePath)  # 加载CT-xml
                            for i in range(1, len(lines)):  # 依次取出缓存中排好序的dcm文件地址
                                DcmFileList.append(lines[i].replace('\n', ''))
                        else:
                            # 2.1.2 没找到排序缓存
                            print("没有排序缓存,准备排序")
                            for itemName in itemNames:
                                itemAbsPath = os.path.join(series_abs_path, itemName)  # 获取每个item的绝对路径
                                if "dcm" in itemName:  # 发现dcm文件
                                    DcmFileList.append(itemAbsPath)
                                elif "xml" in itemName:  # 发现xml标签文件
                                    XmlFilePath = itemAbsPath
                            XmlObj = XmlLabelForCT(XmlFilePath)  # 加载CT-xml
                            # 把dcm文件的绝对路径按照解刨学顺序排列,并创建排序缓存文件
                            SortDcmListBySliceLocation(DcmFileList,XmlFilePath)
                        '进入耗时工作,打进度条'
                        widgets = ['Progress: ', Percentage(), ' ', Bar('#'), ' ', Timer(), ' ', ETA(), ' ',
                                   FileTransferSpeed()]
                        pbar = ProgressBar(widgets=widgets, maxval=len(DcmFileList)).start()
                        i=0
                        for dcmFile in DcmFileList:
                            #print (XmlObj.getRoiListBySOP_UID(pydicom.read_file(dcmFile).SOPInstanceUID))
                            ROIList=[]
                            for roi in XmlObj.getRoiListBySOP_UID(pydicom.read_file(dcmFile).SOPInstanceUID):
                                ROIList.append(np.array(roi,dtype=np.int32))

                            if len(ROIList)>=kv:
                                pix = pydicom.read_file(dcmFile).pixel_array
                                img, ColorMask=GenerateLabel(pix, ROIList)
                                SaveTo = '%s/%s/%s/%s' % (OutPutDIR,patient,study,series)
                                fileFirstName = os.path.basename(dcmFile)[0:6]# 获取文件名
                                if not os.path.exists(SaveTo):
                                    os.makedirs(SaveTo)

                                cv.imwrite(os.path.join(SaveTo, fileFirstName + "-lung.png"), img)
                                cv.imwrite(os.path.join(SaveTo, fileFirstName + "-label.png"), ColorMask)
                            i=i+1
                            pbar.update(i)
                            # 到xmlobj中查询,该dcm文件有没有被标注,如果有,构造一个数组,存储所有对这张图标注的ROI
                            # 生成该dcm对应的mask和label,存储在输出目录中
                        pbar.finish()
                    else:
                        # 不是CT序列
                        for lookingInnonCT in itemNames:
                            nonCTFile = os.path.join(series_abs_path, lookingInnonCT)
                            if "xml" in lookingInnonCT:  # 发现xml文件
                                XmlFilePath = nonCTFile  # 得到这个xml文件的绝对路径
                            elif "dcm" in lookingInnonCT:  # 发现dcm文件
                                DcmFileList.append(nonCTFile)
                        if XmlFilePath == "":
                            print("没有xml!")
                        else:
                            XmlObj = XmlLabelSlim(XmlFilePath)


if __name__ == '__main__':
    Terminator("G:/TCIA_LIDC-IDRI/LIDC-IDRI-Sample")