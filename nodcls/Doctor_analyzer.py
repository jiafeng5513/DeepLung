# coding=utf-8
import pandas as pd
import os
import pydicom
from bs4 import BeautifulSoup
import numpy as np


class Point():
    def __init__(self, x, y):
        self.X = x
        self.Y = y

"ROI"
class Roi():
    def __init__(self):
        self.imageZposition = ""
        self.imageSOP_UID = ""
        self.inclusion = ""
        self.edgeMap = []

class readingSession():
    def __init__(self):
        self.ID = ""  # 医生ID
        self.NoduleList = []  # 结节列表
        self.nonNoduleList = []  # 非结节列表

class Nodule():
    def __init__(self):
        self.noduleID = ""
        self.LargerThan3mm = True
        self.subtlety = 0
        self.internalStructure = 0
        self.calcification = 0
        self.sphericity = 0
        self.margin = 0
        self.lobulation = 0
        self.spiculation = 0
        self.texture = 0
        self.malignancy = 0
        self.RoiList = []


"非结节节点"
class nonNodule():
    def __init__(self):
        self.nonNoduleID = ""
        self.imageZposition = ""
        self.imageSOP_UID = ""
        self.inclusion = ""
        self.locus = Point(0, 0)

class XmlLabelForCT():
    def __init__(self, dir):
        with open(dir, 'r') as xml_file:
            markup = xml_file.read()
        xml = BeautifulSoup(markup, features="xml")
        self.SeriesInstanceUID = xml.LidcReadMessage.ResponseHeader.SeriesInstanceUid.text
        self.StudyInstanceUID = xml.LidcReadMessage.ResponseHeader.StudyInstanceUID.text
        "找到所有的readingSession,理论上应有1~4个"
        reading_sessions = xml.LidcReadMessage.find_all("readingSession")
        self.readingSessionList = []
        for reading_session in reading_sessions:
            _readingSession = readingSession()
            _readingSession.ID = reading_session.servicingRadiologistID.text
            unblindedReadNodules = reading_session.find_all("unblindedReadNodule")
            for unBlindedNodule in unblindedReadNodules:
                _nodule = Nodule()
                _nodule.noduleID = unBlindedNodule.noduleID.text
                characteristicsList = unBlindedNodule.find_all("characteristics")
                if len(characteristicsList) == 0:
                    _nodule.LargerThan3mm = False
                else:
                    _nodule.LargerThan3mm = True
                    characteristics = characteristicsList[0]
                    _nodule.subtlety = int(characteristics.subtlety.text)
                    _nodule.internalStructure = int(characteristics.internalStructure.text)
                    _nodule.calcification = int(characteristics.calcification.text)
                    _nodule.sphericity = int(characteristics.sphericity.text)
                    _nodule.margin = int(characteristics.margin.text)
                    _nodule.lobulation = int(characteristics.lobulation.text)
                    _nodule.spiculation = int(characteristics.spiculation.text)
                    _nodule.texture = int(characteristics.texture.text)
                    _nodule.malignancy = int(characteristics.malignancy.text)
                roiList = unBlindedNodule.find_all("roi")
                for roi in roiList:
                    _roi = Roi()
                    _roi.imageZposition = roi.imageZposition.text
                    _roi.imageSOP_UID = roi.imageSOP_UID.text
                    _roi.inclusion = roi.inclusion.text
                    edgeMapList = roi.find_all("edgeMap")
                    for edgeMap in edgeMapList:
                        # _Point = Point(int(edgeMap.xCoord.text), int(edgeMap.yCoord.text))
                        _roi.edgeMap.append([int(edgeMap.xCoord.text), int(edgeMap.yCoord.text)])
                    _nodule.RoiList.append(_roi)
                _readingSession.NoduleList.append(_nodule)
            nonNodules = reading_session.find_all("nonNodule")
            for nodule in nonNodules:
                _nonNode = nonNodule()
                _nonNode.nonNoduleID = nodule.nonNoduleID.text
                _nonNode.imageZposition = nodule.imageZposition.text
                _nonNode.imageSOP_UID = nodule.imageSOP_UID.text
                _nonNode.locus.X = int(nodule.locus.xCoord.text)
                _nonNode.locus.Y = int(nodule.locus.yCoord.text)
                _readingSession.nonNoduleList.append(_nonNode)
            self.readingSessionList.append(_readingSession)

    def getRoiListBySOP_UID(self,SOP_UID):
        ROIList=[]
        for _readingSession in self.readingSessionList:
            for Nodule in _readingSession.NoduleList:
                for roi in Nodule.RoiList:
                    if roi.imageSOP_UID == SOP_UID:
                        ROIList.append(roi.edgeMap)
        return ROIList

def getDoctor_cls_from_LIDC(RootDir):
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
                        for itemName in itemNames:
                            itemAbsPath = os.path.join(series_abs_path, itemName)  # 获取每个item的绝对路径
                            if "dcm" in itemName:  # 发现dcm文件
                                DcmFileList.append(itemAbsPath)
                            elif "xml" in itemName:  # 发现xml标签文件
                                XmlFilePath = itemAbsPath
                        XmlObj = XmlLabelForCT(XmlFilePath)  # 加载CT-xml

global doctor_count
def getLine(csv,index):
    global doctor_count
    record = csv.ix[index,:]
    malignant=record['malignant']
    if not np.isnan(record['A']):
        A= int(record['A'])
        doctor_count[0] +=1
    else:
        A= 3

    if not np.isnan(record['B']):
        B= int(record['B'])
        doctor_count[1] += 1
    else:
        B = 3

    if not np.isnan(record['C']):
        C= int(record['C'])
        doctor_count[2] += 1
    else:
        C= 3

    if not np.isnan(record['D']):
        D= int(record['D'])
        doctor_count[3] += 1
    else:
        D = 3

    return malignant,A,B,C,D

def DoctorAnalyzer(doctor_csv):
    doctor_csv=pd.read_csv(doctor_csv,error_bad_lines=False)
    num = int(doctor_csv.describe().ix[0,0])
    doctor_true_count=[0,0,0,0]
    doctor_value=[0,0,0,0]
    for index in range(num):
        malignant,doctor_value[0],doctor_value[1],doctor_value[2],doctor_value[3]=getLine(doctor_csv,index)
        if malignant==0:
            for i in range(4):
                if doctor_value[i]<3:
                    doctor_true_count[i] +=1
        else:
            for i in range(4):
                if doctor_value[i]>3:
                    doctor_true_count[i] +=1
    print(doctor_true_count)
    global doctor_count
    print(doctor_count)
    print(num)
    print('A:%.2f,B:%.2f,C:%.2f,D:%.2f,' % (doctor_true_count[0] / num,doctor_true_count[1] / num,
                                            doctor_true_count[2] / num,doctor_true_count[3] / num,))

if __name__ == '__main__':
    global doctor_count
    doctor_count = [0, 0, 0, 0]
    DoctorAnalyzer('./data/annotationdetclssgm_doctor.csv')