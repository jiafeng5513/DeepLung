from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import transforms as transforms
import os
import argparse
import pickle
import logging
import pandas as pd
import numpy as np
from models import *
from torch.autograd import Variable
from tqdm import tqdm
from sklearn.ensemble import GradientBoostingClassifier as gbt
from dataloader import lunanod


CROPSIZE = 32
gbtdepth = 1
fold = 9    #subset id
blklst = []#['1.3.6.1.4.1.14519.5.2.1.6279.6001.121993590721161347818774929286-388', \
           # '1.3.6.1.4.1.14519.5.2.1.6279.6001.121993590721161347818774929286-389', \
           # '1.3.6.1.4.1.14519.5.2.1.6279.6001.132817748896065918417924920957-660']

preprocesspath = '/home/RAID1/DataSet/LUNA16/crop_v3/'
csvfilepath = './data/annotationdetclsconvfnl_v3.csv'
luna16path = '/home/RAID1/DataSet/LUNA16/'
savemodelpath = './results/checkpoint-'+str(fold)+'/'

logging.basicConfig(filename='./results/log-'+str(fold), level=logging.INFO)
parser = argparse.ArgumentParser(description='nodcls')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()
# Cal mean std

class nodcls():
    def __init__(self):
        self.best_acc = 0
        self.best_acc_gbt = 0
        self.use_cuda = torch.cuda.is_available()
        pixvlu, npix = 0, 0
        for fname in os.listdir(preprocesspath):
            if fname.endswith('.npy'):
                if fname[:-4] in blklst: continue
                data = np.load(os.path.join(preprocesspath, fname))
                pixvlu += np.sum(data)
                npix += np.prod(data.shape)
        pixmean = pixvlu / float(npix)
        pixvlu = 0
        for fname in os.listdir(preprocesspath):
            if fname.endswith('.npy'):
                if fname[:-4] in blklst: continue
                data = np.load(os.path.join(preprocesspath, fname)) - pixmean
                pixvlu += np.sum(data * data)
        pixstd = np.sqrt(pixvlu / float(npix))
        print('pixmean:%.3f, pixstd:%.3f' % (pixmean, pixstd))
        logging.info('mean ' + str(pixmean) + ' std ' + str(pixstd))
        # Datatransforms
        logging.info('==> Preparing data..')  # Random Crop, Zero out, x z flip, scale,
        transform_train = transforms.Compose([
            # transforms.RandomScale(range(28, 38)),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomYFlip(),
            transforms.RandomZFlip(),
            transforms.ZeroOut(4),
            transforms.ToTensor(),
            transforms.Normalize((pixmean), (pixstd)),  # need to cal mean and std, revise norm func
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((pixmean), (pixstd)),
        ])
        # load data list
        self.trfnamelst = []
        trlabellst = []
        trfeatlst = []
        self.tefnamelst = []
        telabellst = []
        tefeatlst = []
        dataframe = pd.read_csv(csvfilepath,
                                names=['seriesuid', 'coordX', 'coordY', 'coordZ', 'diameter_mm', 'malignant'])
        alllst = dataframe['seriesuid'].tolist()[1:]
        labellst = dataframe['malignant'].tolist()[1:]
        crdxlst = dataframe['coordX'].tolist()[1:]
        crdylst = dataframe['coordY'].tolist()[1:]
        crdzlst = dataframe['coordZ'].tolist()[1:]
        dimlst = dataframe['diameter_mm'].tolist()[1:]
        # test id
        teidlst = []
        for fname in os.listdir(luna16path + '/subset' + str(fold) + '/'):
            if fname.endswith('.mhd'):
                teidlst.append(fname[:-4])
        mxx = mxy = mxz = mxd = 0
        for srsid, label, x, y, z, d in zip(alllst, labellst, crdxlst, crdylst, crdzlst, dimlst):
            mxx = max(abs(float(x)), mxx)
            mxy = max(abs(float(y)), mxy)
            mxz = max(abs(float(z)), mxz)
            mxd = max(abs(float(d)), mxd)
            if srsid in blklst: continue
            # crop raw pixel as feature
            if os.path.exists(os.path.join(preprocesspath, srsid + '.npy')):
                data = np.load(os.path.join(preprocesspath, srsid + '.npy'))
            bgx = data.shape[0] / 2 - CROPSIZE / 2
            bgy = data.shape[1] / 2 - CROPSIZE / 2
            bgz = data.shape[2] / 2 - CROPSIZE / 2
            data = np.array(data[bgx:bgx + CROPSIZE, bgy:bgy + CROPSIZE, bgz:bgz + CROPSIZE])
            feat = np.hstack((np.reshape(data, (-1,)) / 255, float(d)))
            if srsid.split('-')[0] in teidlst:
                self.tefnamelst.append(srsid + '.npy')
                telabellst.append(int(label))
                tefeatlst.append(feat)
            else:
                self.trfnamelst.append(srsid + '.npy')
                trlabellst.append(int(label))
                trfeatlst.append(feat)
        for idx in xrange(len(trfeatlst)):
            trfeatlst[idx][-1] /= mxd

        for idx in xrange(len(tefeatlst)):
            tefeatlst[idx][-1] /= mxd
        trainset = lunanod(preprocesspath, self.trfnamelst, trlabellst, trfeatlst, train=True, download=True,
                           transform=transform_train)
        self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=True, num_workers=30)

        testset = lunanod(preprocesspath, self.tefnamelst, telabellst, tefeatlst, train=False, download=True,
                          transform=transform_test)
        self.testloader = torch.utils.data.DataLoader(testset, batch_size=16, shuffle=False, num_workers=30)

        # Model
        if args.resume:
            # Load checkpoint.
            logging.info('==> Resuming from checkpoint..')
            # assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
            checkpoint = torch.load(savemodelpath + 'ckpt.t7')
            self.net = checkpoint['net']
            best_acc = checkpoint['acc']
            start_epoch = checkpoint['epoch']
        else:
            logging.info('==> Building model..')
            self.net = dpn3d.DPN92_3D()


        if self.use_cuda:
            self.net.cuda()
            self.net = torch.nn.DataParallel(self.net, device_ids=range(torch.cuda.device_count()))
            cudnn.benchmark = False  # True

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        pass

    # learning rate
    def get_lr(self, epoch, neptime = 2):
        if epoch < 150 * neptime:
            lr = 0.1  # args.lr
        elif epoch < 250 * neptime:
            lr = 0.01
        else:
            lr = 0.001
        return lr

    # Training a epoch
    def train(self, epoch):
        logging.info('\nEpoch: '+str(epoch))
        self.net.train()
        lr = self.get_lr(epoch)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        train_loss = 0
        correct = 0
        total = 0
        trainfeat = np.zeros((len(self.trfnamelst), 2560+CROPSIZE*CROPSIZE*CROPSIZE+1))
        trainlabel = np.zeros((len(self.trfnamelst),))
        idx = 0
        pbar = tqdm(total=len(self.trainloader), unit="batchs")
        for batch_idx, (inputs, targets, feat) in enumerate(self.trainloader):
            if self.use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            self.optimizer.zero_grad()
            inputs, targets = Variable(inputs), Variable(targets)
            outputs, dfeat = self.net(inputs)
            # add feature into the array
            trainfeat[idx:idx+len(targets), :2560] = np.array((dfeat.data).cpu().numpy())
            for i in xrange(len(targets)):
                trainfeat[idx+i, 2560:] = np.array((Variable(feat[i]).data).cpu().numpy())
                trainlabel[idx+i] = np.array((targets[i].data).cpu().numpy())
            idx += len(targets)

            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.data.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()
            pbar.update(1)
            pbar.set_description('Training: epoch: %d | Loss: %.3f | Acc: %.3f%% (%d/%d)'%
                                 (epoch,train_loss/(batch_idx+1), 100.*correct/total, correct, total))

        pbar.close()
        m = gbt(max_depth=gbtdepth, random_state=0)
        m.fit(trainfeat, trainlabel)
        gbttracc = np.mean(m.predict(trainfeat) == trainlabel)
        #print('ep '+str(epoch)+' tracc '+str(correct/float(total))+' lr '+str(lr)+' gbtacc '+str(gbttracc))
        logging.info('ep '+str(epoch)+' tracc '+str(correct/float(total))+' lr '+str(lr)+' gbtacc '+str(gbttracc))
        return m

    # Test a epoch
    def test(self, epoch, m):
        self.net.eval()
        test_loss = 0
        correct = 0
        total = 0
        testfeat = np.zeros((len(self.tefnamelst), 2560 + CROPSIZE * CROPSIZE * CROPSIZE + 1))
        testlabel = np.zeros((len(self.tefnamelst),))
        idx = 0
        pbar = tqdm(total=len(self.testloader), unit="batchs")
        for batch_idx, (inputs, targets, feat) in enumerate(self.testloader):
            if self.use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            with torch.no_grad():
                inputs, targets = Variable(inputs), Variable(targets)
            outputs, dfeat = self.net(inputs)
            # add feature into the array
            testfeat[idx:idx + len(targets), :2560] = np.array((dfeat.data).cpu().numpy())
            for i in xrange(len(targets)):
                testfeat[idx + i, 2560:] = np.array((Variable(feat[i]).data).cpu().numpy())
                testlabel[idx + i] = np.array((targets[i].data).cpu().numpy())
            idx += len(targets)

            loss = self.criterion(outputs, targets)
            test_loss += loss.data.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()
            pbar.update(1)
            pbar.set_description('Test: epoch: %d | Loss: %.3f | Acc: %.3f%% (%d/%d)' %
                                 (epoch, test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

        gbtteacc = np.mean(m.predict(testfeat) == testlabel)
        if gbtteacc > self.best_acc_gbt:
            pickle.dump(m, open('./results/gbtmodel-' + str(fold) + '.sav', 'wb'))
            logging.info('Saving gbt ..')
            state = {
                'net': self.net.module if self.use_cuda else self.net,
                'epoch': epoch,
            }
            if not os.path.isdir(savemodelpath):
                os.mkdir(savemodelpath)
            torch.save(state, savemodelpath + 'ckptgbt.t7')
            best_acc_gbt = gbtteacc
        # Save checkpoint.
        acc = 100. * correct / total
        if acc > self.best_acc:
            logging.info('Saving..')
            state = {
                'net': self.net.module if self.use_cuda else self.net,
                'acc': acc,
                'epoch': epoch,
            }
            if not os.path.isdir(savemodelpath):
                os.mkdir(savemodelpath)
            torch.save(state, savemodelpath + 'ckpt.t7')
            best_acc = acc
        logging.info('Saving..')
        state = {
            'net': self.net.module if self.use_cuda else self.net,
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir(savemodelpath):
            os.mkdir(savemodelpath)
        if epoch % 50 == 0:
            torch.save(state, savemodelpath + 'ckpt' + str(epoch) + '.t7')

        # print(
        #     '\nteacc ' + str(acc) + ' bestacc ' + str(best_acc) + ' gbttestaccgbt ' + str(gbtteacc) + ' bestgbt ' + str(
        #         best_acc_gbt))
        logging.info('teacc ' + str(acc) + ' bestacc ' + str(best_acc) + ' ccgbt ' + str(gbtteacc) + ' bestgbt ' + str(
            best_acc_gbt))

    # Train and test all nodcls
    def TrainAndTest(self,start_epoch, max_epoch):
        for epoch in range(start_epoch, max_epoch):  # 200):
            m = self.train(epoch)
            self.test(epoch, m)



if __name__ == '__main__':
    classifier = nodcls()
    classifier.TrainAndTest(start_epoch=0,max_epoch=350 * 2)
