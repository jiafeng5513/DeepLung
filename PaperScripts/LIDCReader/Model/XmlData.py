from bs4 import BeautifulSoup
import numpy as np
"""
两个职责:
    1.从绝对路径加载xml文件并阅读.
    2.分拣信息
当序列为CT序列时,进行全部信息的分拣;
当序列时非CT序列时,只分拣简要信息
    之所以需要分析非CT序列,是因为patientID,seriesid,studyid等信息必须从xml图像中直接或间接获取
"""
"坐标"


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


"结节节点"


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


"每个reading_sessions中存储一个医生的全部标注信息"


class readingSession():
    def __init__(self):
        self.ID = ""  # 医生ID
        self.NoduleList = []  # 结节列表
        self.nonNoduleList = []  # 非结节列表


"xml标注文件ForCT"


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



" 精简版XML文件对象,用于描述DX和CR扫描序列的xml标记文档 "


class XmlLabelSlim():
    def __init__(self, dir):
        with open(dir, 'r') as xml_file:
            markup = xml_file.read()
        xml = BeautifulSoup(markup, features="xml")
        #print("构造非CT序列的xml文件对象")
        self.SeriesInstanceUID = xml.IdriReadMessage.ResponseHeader.SeriesInstanceUID.text
        self.StudyInstanceUID = xml.IdriReadMessage.ResponseHeader.StudyInstanceUID.text


if __name__ == '__main__':
    # xml=XmlLabel('F:/TCIA_LIDC-IDRI/LIDC-IDRI/LIDC-IDRI-0001/1.3.6.1.4.1.14519.5.2.1.6279.6001.298806137288633453246975630178/1.3.6.1.4.1.14519.5.2.1.6279.6001.179049373636438705059720603192/069.xml')
    xml = XmlLabelForCT("G:/TCIA_LIDC-IDRI/LIDC-IDRI/LIDC-IDRI-0002/1.3.6.1.4.1.14519.5.2.1.6279.6001.490157381160200744295382098329/1.3.6.1.4.1.14519.5.2.1.6279.6001.619372068417051974713149104919/071.xml")
    print("Num Of readingSession:" + str(len(xml.readingSessionList)))
    print("Num of Node :" + str(len(xml.readingSessionList[0].NoduleList)))
    #print("Num of ROI :" + str(len(xml.readingSessionList[0].NoduleList[0].RoiList)))

    print("Num of Node :" + str(len(xml.readingSessionList[1].NoduleList)))
    print("Num of Node :" + str(len(xml.readingSessionList[2].NoduleList)))
    print("Num of Node :" + str(len(xml.readingSessionList[3].NoduleList)))
