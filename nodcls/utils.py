import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')
def plot_log(logname='/media/data1/wentao/CTnoddetector/training/nodcls/log-1'):
    fid = open(logname, 'r')
    flines = fid.readlines()
    tracclst = []
    teacclst = []
    ep = 0
    for line in flines:
        if line.startswith('INFO:root:ep '+str(ep)+' tracc '):
            acc = line.split('INFO:root:ep '+str(ep)+' tracc ')[1]
            # print acc
            acc = acc.split(' gbtacc ')[1]
            # acc = acc.split(' lr ')[0]
            # print acc
            tracclst.append(float(acc))
            ep += 1
        if line.startswith('INFO:root:teacc '):
            acc = line.split('INFO:root:teacc ')[1]
            acc = acc.split(' bestacc ')[0]
            # acc = acc.split(' ccgbt ')[1]
            # acc = acc.split(' bestgbt ')[0]
            # print acc
            teacclst.append(float(acc))#/100)
    fid.close()
    print(max(teacclst))
    plt.plot(range(len(tracclst)), tracclst, label='train accuracy')
    plt.plot(range(len(tracclst)), teacclst, label='test accuracy')
    plt.legend()
    plt.savefig('log-1plt.png')
    # print(max(teacclst))

# plot_log()