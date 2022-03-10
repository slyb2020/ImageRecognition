import numpy as np
import pandas as pd
import torch
import torchvision
from torch import nn
from torchvision import transforms
from torchvision.datasets import CIFAR10
from vgg import VGG
import time
from CIFAR10_Dataset import *
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# leNet5 = LeNet5()
model = VGG('VGG16')
# model = torch.load('model/LeeNetMNISTDropout.pth')
model.to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
Loss = torch.nn.CrossEntropyLoss()
Loss.to(device)
maxEpoch = 500
maxAccuracy = 0
accuracyTrainList = []
accuracyTestList = []
epochList = []
lossEpochList = []
for epoch in range(maxEpoch):
    model.train()
    lossEpoch = 0
    startTime = time.time()
    for imgs, labels in trainLoader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        predict = model(imgs)
        loss = Loss(predict, labels)
        loss.backward()
        optimizer.step()
        lossEpoch += loss
    endTime = time.time()
    model.eval()
    with torch.no_grad():
        accuracyTotalTrain = 0
        for imgs, labels in trainLoader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            predict = model(imgs)
            accuracy = (torch.argmax(predict, 1) == labels).sum()
            accuracyTotalTrain += accuracy
        accuracyTotalTest = 0
        for imgs, labels in testLoader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            predict = model(imgs)
            accuracy = (torch.argmax(predict, 1) == labels).sum()
            accuracyTotalTest += accuracy
    testTime = time.time()
    print("已训练{}/{}代， 损失：{}，训练集准确率：{}， 测试集准确率{}, 训练用时{}，测试用时{}".format(epoch + 1, maxEpoch, lossEpoch,
                                                                       100 * accuracyTotalTrain / trainSize,
                                                                       100 * accuracyTotalTest / testSize,
                                                                       endTime - startTime, testTime - endTime))
    accuracyTrainList.append(100 * (accuracyTotalTrain.item()) / trainSize)
    accuracyTestList.append(100 * (accuracyTotalTest.item()) / testSize)
    epochList.append(epoch + 1)
    lossEpochList.append(lossEpoch.item())
    if maxAccuracy < (100 * accuracyTotalTest / testSize):
        maxAccuracy = 100 * accuracyTotalTest / testSize
        torch.save(model, "model/VGG/VGG16%s.pth" % (int(maxAccuracy * 100)))

# torch.save(model, "model/LeeNet/LeeNetCIFAR10_%s.pth"%(int(maxAccuracy*100)))
epochList = np.array(epochList).reshape(-1, 1)
lossEpochList = np.array(lossEpochList).reshape(-1, 1)
accuracyTrainList = np.array(accuracyTrainList).reshape(-1, 1)
accuracyTestList = np.array(accuracyTestList).reshape(-1, 1)
dataArray = np.hstack((epochList, lossEpochList, accuracyTrainList, accuracyTestList))
dataFrame = pd.DataFrame(dataArray, columns=['epoch', 'loss', 'accuracyTrain', 'accuracyTest'])
dataFrame.to_csv("./log/VGG/VGG16/VGG16CIFAR10.csv", mode='w', index=None, encoding='utf_8_sig')
