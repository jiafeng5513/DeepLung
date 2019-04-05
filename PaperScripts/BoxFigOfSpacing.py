"""
绘制结节尺寸的箱体图
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.font_manager import FontProperties

from matplotlib.font_manager import FontProperties

font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
plt.figure(figsize=(6,6))

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
    return np.array(lenths)

# 箱体图
def drawTimeBox(data):
    np.random.seed(100)
    # data = np.random.normal(size=(1000,4) , loc = 0 , scale=1)
    labels = [u'LIDC-IDRI']
    plt.boxplot(data, labels = labels)
    #plt.boxplot(data, labels = labels,)
    plt.ylabel(u"毫米每像素",fontproperties=font)
    plt.show()

if __name__ == '__main__':
    drawTimeBox(get_data())