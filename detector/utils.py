import sys
import os
import numpy as np
import torch
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt 

def getFreeId():
    import pynvml 

    pynvml.nvmlInit()
    def getFreeRatio(id):
        handle = pynvml.nvmlDeviceGetHandleByIndex(id)
        use = pynvml.nvmlDeviceGetUtilizationRates(handle)
        ratio = 0.5*(float(use.gpu+float(use.memory)))
        return ratio

    deviceCount = pynvml.nvmlDeviceGetCount()
    available = []
    for i in range(deviceCount):
        if getFreeRatio(i)<70:
            available.append(i)
    gpus = ''
    for g in available:
        gpus = gpus+str(g)+','
    gpus = gpus[:-1]
    return gpus

def setgpu(gpuinput):
    freeids = getFreeId()
    if gpuinput=='all':
        gpus = freeids
    else:
        gpus = gpuinput
        if any([g not in freeids for g in gpus.split(',')]):
            raise ValueError('gpu'+g+'is being used')
    print('using gpu '+gpus)
    os.environ['CUDA_VISIBLE_DEVICES']=gpus
    return len(gpus.split(','))

class Logger(object):
    def __init__(self,logfile):
        self.terminal = sys.stdout
        self.log = open(logfile, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass    





def plotlog(logfile, savepath):
    traintpr = []
    traintnr = []
    trainloss = []
    trainclassifyloss = []
    trainregresslossx = []
    trainregresslossy = []
    trainregresslossz = []
    trainregresslossd = []

    valtpr = []
    valtnr = []
    valloss = []
    valclassifyloss = []
    valregresslossx = []
    valregresslossy = []
    valregresslossz = []
    valregresslossd = []

    eps = 1
    f = open(logfile, 'r')
    line = f.readline()
    while line:
        if line.startswith('Epoch '+'{:03d}'.format(eps)+' (lr '):
            trainline1 = f.readline()
            strlist = trainline1.split('Train:      tpr ')
            # print strlist
            strlist1 = strlist[1].split(',')
            # print strlist1, float(strlist1[0])
            traintpr.append(float(strlist1[0]))
            strlist2 = strlist1[1].split('tnr ')
            # print strlist2, float(strlist2[1])
            traintnr.append(float(strlist2[1])) 

            trainline2 = f.readline()[5:]
            strlist = trainline2.split(', classify loss ')
            # print strlist, float(strlist[0])
            trainloss.append(float(strlist[0]))
            strlist1 = strlist[1].split(', regress loss ')
            # print strlist1, float(strlist1[0])
            trainclassifyloss.append(float(strlist1[0]))
            strlist2 = strlist1[1].split(', ')
            # print strlist2, float(strlist2[0]), float(strlist2[1]), float(strlist2[2]), float(strlist2[3])
            trainregresslossx.append(float(strlist2[0]))
            trainregresslossy.append(float(strlist2[1]))
            trainregresslossz.append(float(strlist2[2]))
            trainregresslossd.append(float(strlist2[3]))

            f.readline()

            valline1 = f.readline()
            strlist = valline1.split('Validation: tpr ')
            # print strlist
            strlist1 = strlist[1].split(',')
            # print strlist1, float(strlist1[0])
            valtpr.append(float(strlist1[0]))
            strlist2 = strlist1[1].split('tnr ')
            # print strlist2, float(strlist2[1])
            valtnr.append(float(strlist2[1])) 

            valline2 = f.readline()[5:]
            strlist = valline2.split(', classify loss ')
            # print strlist, float(strlist[0])
            valloss.append(float(strlist[0]))
            strlist1 = strlist[1].split(', regress loss ')
            # print strlist1, float(strlist1[0])
            valclassifyloss.append(float(strlist1[0]))
            strlist2 = strlist1[1].split(', ')
            # print strlist2, float(strlist2[0]), float(strlist2[1]), float(strlist2[2]), float(strlist2[3])
            valregresslossx.append(float(strlist2[0]))
            valregresslossy.append(float(strlist2[1]))
            valregresslossz.append(float(strlist2[2]))
            valregresslossd.append(float(strlist2[3]))

            eps += 1
        line = f.readline()
    f.close()

    fig = plt.figure()
    plt.plot(range(1, len(traintpr)+1, 1), traintpr, label='train')
    plt.plot(range(1, len(traintpr)+1, 1), valtpr, label='val')
    plt.legend()
    plt.title('True Positive Rate')
    plt.savefig(savepath+'tpr.png')

    fig = plt.figure()
    plt.plot(range(1, len(traintnr)+1, 1), traintnr, label='train')
    plt.plot(range(1, len(traintpr)+1, 1), valtnr, label='val')
    plt.legend()
    plt.title('True Negative Rate')
    plt.savefig(savepath+'tnr.png')

    fig = plt.figure()
    plt.plot(range(1, len(traintpr)+1, 1), trainloss, label='train')
    plt.plot(range(1, len(traintpr)+1, 1), valloss, label='val')
    plt.legend()
    plt.title('Loss')
    plt.savefig(savepath+'loss.png')

    fig = plt.figure()
    plt.plot(range(1, len(traintpr)+1, 1), trainclassifyloss, label='train')
    plt.plot(range(1, len(traintpr)+1, 1), valclassifyloss, label='val')
    plt.legend()
    plt.title('Classification Loss')
    plt.savefig(savepath+'classificationloss.png')

    fig = plt.figure()
    plt.plot(range(1, len(traintpr)+1, 1), trainregresslossx, label='train')
    plt.plot(range(1, len(traintpr)+1, 1), valregresslossx, label='val')
    plt.legend()
    plt.title('Regresion X Loss')
    plt.savefig(savepath+'regressionxloss.png')

    fig = plt.figure()
    plt.plot(range(1, len(traintpr)+1, 1), trainregresslossy, label='train')
    plt.plot(range(1, len(traintpr)+1, 1), valregresslossy, label='val')
    plt.legend()
    plt.title('Regresion Y Loss')
    plt.savefig(savepath+'regressionyloss.png')

    fig = plt.figure()
    plt.plot(range(1, len(traintpr)+1, 1), trainregresslossz, label='train')
    plt.plot(range(1, len(traintpr)+1, 1), valregresslossz, label='val')
    plt.legend()
    plt.title('Regresion Z Loss')
    plt.savefig(savepath+'regressionzloss.png')

    fig = plt.figure()
    plt.plot(range(1, len(traintpr)+1, 1), trainregresslossd, label='train')
    plt.plot(range(1, len(traintpr)+1, 1), valregresslossd, label='val')
    plt.legend()
    plt.title('Regresion D Loss')
    plt.savefig(savepath+'regressiondloss.png')


if __name__ == '__main__':
    plotlog('log', savepath='./')
