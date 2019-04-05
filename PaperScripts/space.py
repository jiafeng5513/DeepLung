"""
找出所有的像素间隔
"""
import os
import numpy as np
import pydicom
import SimpleITK as sitk

"""
处理函数
"""
def Terminator(RootDir):
    print ("开始处理...")
    file = open('spacingCT.txt', 'w')
    file2 = open('spacingNotCT.txt', 'w')
    if os.path.exists(RootDir):  # 判断路径是否存在
        patients = os.listdir(RootDir)  # patients
        patientNum = 1
        for patient in patients:
            #print("正在处理第{:4d}个patient,还剩下{:4d}个patient".format(patientNum,len(patients)-patientNum))
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
                        if os.path.exists(os.path.join(series_abs_path,"000006.dcm")):
                            itkimage=sitk.ReadImage(os.path.join(series_abs_path,"000006.dcm"))
                            npspacing=np.array(list(reversed(itkimage.GetSpacing())))
                            print(npspacing[1])
                            file.write(str(npspacing[1]))
                            file.write('\n')
                    # else:
                    #     tempfilename=''
                    #     if os.path.exists(os.path.join(series_abs_path,"000001.dcm")):
                    #         tempfilename = os.path.join(series_abs_path,"000001.dcm")
                    #     elif os.path.exists(os.path.join(series_abs_path,"000000.dcm")):
                    #         tempfilename = os.path.join(series_abs_path,"000000.dcm")
                    #
                    #     itkimage = sitk.ReadImage(tempfilename)
                    #     npspacing = np.array(list(reversed(itkimage.GetSpacing())))
                    #     print(npspacing)
                    #     file2.write(str(npspacing))
                    #     file2.write('\n')



if __name__ == '__main__':
    Terminator("G:/TCIA_LIDC-IDRI/LIDC-IDRI")