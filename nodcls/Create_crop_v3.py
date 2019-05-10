preprocesspath = '/home/RAID1/DataSet/LUNA16/preprocess/all'
savepath = '/home/RAID1/DataSet/LUNA16/crop_v3/'

import os
import os.path
import numpy as np
import pandas as pd
newlst = []
CROPSIZE = 32#24#30#36
print CROPSIZE

newlst = []
import csv

pdframe = pd.read_csv('./data/annotationdetclsconvfnl_v3.csv', names=['seriesuid', 'coordX', 'coordY', 'coordZ', 'diameter_mm', 'malignant'])
srslst = pdframe['seriesuid'].tolist()[1:]
crdxlst = pdframe['coordX'].tolist()[1:]
crdylst = pdframe['coordY'].tolist()[1:]
crdzlst = pdframe['coordZ'].tolist()[1:]
dimlst = pdframe['diameter_mm'].tolist()[1:]
mlglst = pdframe['malignant'].tolist()[1:]


fid = open('./data/annotationdetclsconvfnl_v3.csv', 'w')
writer = csv.writer(fid)
writer.writerow(['seriesuid', 'coordX', 'coordY', 'coordZ', 'diameter_mm', 'malignant'])
for i in xrange(len(srslst)):
    writer.writerow([srslst[i]+'-'+str(i), crdxlst[i], crdylst[i], crdzlst[i], dimlst[i], mlglst[i]])
    newlst.append([srslst[i]+'-'+str(i), crdxlst[i], crdylst[i], crdzlst[i], dimlst[i], mlglst[i]])
fid.close()

if not os.path.exists(savepath): os.mkdir(savepath)
for idx in xrange(len(newlst)):
    fname = newlst[idx][0]
    # if fname != '1.3.6.1.4.1.14519.5.2.1.6279.6001.119209873306155771318545953948-581': continue
    pid = fname.split('-')[0]
    crdx = int(float(newlst[idx][1]))
    crdy = int(float(newlst[idx][2]))
    crdz = int(float(newlst[idx][3]))
    dim = int(float(newlst[idx][4]))
    if os.path.exists(os.path.join(preprocesspath, pid+'_clean.npy')):
        data = np.load(os.path.join(preprocesspath, pid+'_clean.npy'))
        bgx = max(0, crdx-CROPSIZE/2)
        bgy = max(0, crdy-CROPSIZE/2)
        bgz = max(0, crdz-CROPSIZE/2)
        cropdata = np.ones((CROPSIZE, CROPSIZE, CROPSIZE))*170
        cropdatatmp = np.array(data[0, bgx:bgx+CROPSIZE, bgy:bgy+CROPSIZE, bgz:bgz+CROPSIZE])
        cropdata[CROPSIZE/2-cropdatatmp.shape[0]/2:CROPSIZE/2-cropdatatmp.shape[0]/2+cropdatatmp.shape[0], \
            CROPSIZE/2-cropdatatmp.shape[1]/2:CROPSIZE/2-cropdatatmp.shape[1]/2+cropdatatmp.shape[1], \
            CROPSIZE/2-cropdatatmp.shape[2]/2:CROPSIZE/2-cropdatatmp.shape[2]/2+cropdatatmp.shape[2]] = np.array(2-cropdatatmp)
        assert cropdata.shape[0] == CROPSIZE and cropdata.shape[1] == CROPSIZE and cropdata.shape[2] == CROPSIZE
        np.save(os.path.join(savepath, fname+'.npy'), cropdata)