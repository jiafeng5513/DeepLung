#-*- coding: utf-8 -*-
from numpy import array
from numpy.random import normal
from matplotlib import pyplot
from matplotlib.font_manager import FontProperties

font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
pyplot.figure(figsize=(6,6))
"""
统计结节尺寸
"""
def get_data():
    n=0
    lenths=[]
    readfile=open("spacingCT.txt")
    lines=readfile.readlines()
    for line in lines:
        line=float(line)
        lenths.append(line)
        #
        n+=1
    print(n)
    return array(lenths)

lenths=get_data()

def draw_hist(lenths):
    pyplot.hist(lenths,100)

    pyplot.xlabel(u'像素间隔', fontproperties=font)
    pyplot.xlim(0.4,1)
    pyplot.ylabel(u'数量', fontproperties=font)
    pyplot.title(u'LIDC-IDRI中像素间隔的分布', fontproperties=font)
    pyplot.show()

draw_hist(lenths)
