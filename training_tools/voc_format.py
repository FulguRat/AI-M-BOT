import os
import random


xmlfilepath = './Annotations/'
imagefilepath = './JPEGImages/'
txtsavepath = './ImageSets/Main/'


if not os.path.exists(txtsavepath):
    os.makedirs(txtsavepath)

trainval_percent = 0.95
train_percent = 0.9
total_xml = os.listdir(xmlfilepath)
num = len(total_xml)
print("Total xml: ", num)
list = range(num)
tv = int(num * trainval_percent)
tr = int(tv * train_percent)
trainval = random.sample(list, tv)
train = random.sample(trainval, tr)

print("train and val size:", tv)
print("train size:", tr)

ftrainval = open(txtsavepath + '/trainval.txt', 'w')
ftest = open(txtsavepath + '/test.txt', 'w')
ftrain = open(txtsavepath + '/train.txt', 'w')
fval = open(txtsavepath + '/val.txt', 'w')

for i in list:
    name = total_xml[i][:-4] + '\n'
    if i in trainval:
        ftrainval.write(name)
        if i in train:
            ftrain.write(name)
        else:
            fval.write(name)
    else:
        ftest.write(name)

ftrainval.close()
ftrain.close()
fval.close()
ftest.close()
