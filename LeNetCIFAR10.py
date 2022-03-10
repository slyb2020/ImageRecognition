import numpy as np
import pandas as pd
import torch
import torchvision
from torch import nn
from torchvision import transforms
from torchvision.datasets import CIFAR10
from LeNet import LeNet5
import time
from CIFAR10_Dataset import *
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

leNet5 = LeNet5()
# leNet5 = torch.load("model/LeNet/LeNetCIFAR10_6337.pth")
leNet5.to(device)
optimizer = torch.optim.SGD(leNet5.parameters(), lr=1e-3, momentum=0.9)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
Loss = torch.nn.CrossEntropyLoss()
Loss.to(device)
maxEpoch = 500
maxAccuracy = 63.37
accuracyTrainList = []
accuracyTestList = []
epochList = []
lossEpochList = []
for epoch in range(maxEpoch):
    leNet5.train()
    lossEpoch = 0
    startTime = time.time()
    for imgs, labels in trainLoader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        predict = leNet5(imgs)
        loss = Loss(predict, labels)
        loss.backward()
        optimizer.step()
        lossEpoch += loss
    endTime = time.time()
    leNet5.eval()
    with torch.no_grad():
        accuracyTotal = 0
        for imgs, labels in testLoader:
            imgs = imgs.to(device)
            labels= labels.to(device)
            predict = leNet5(imgs)
            accuracy = (torch.argmax(predict, 1)==labels).sum()
            accuracyTotal+=accuracy
        accuracyTrain = 0
        for imgs, labels in trainLoader:
            imgs = imgs.to(device)
            labels= labels.to(device)
            predict = leNet5(imgs)
            accuracy = (torch.argmax(predict, 1)==labels).sum()
            accuracyTrain+=accuracy
    testTime = time.time()
    print("已训练{}/{}代， 损失：{}，训练准确率：{}， 测试准确率{}, 训练用时{}，测试用时{}".format(epoch+1, maxEpoch, lossEpoch,
                    100*accuracyTrain/trainSize, 100*accuracyTotal/testSize, endTime-startTime, testTime - endTime))
    accuracyTrainList.append(100*(accuracyTrain.item())/trainSize)
    accuracyTestList.append(100*(accuracyTotal.item())/testSize)
    epochList.append(epoch+1)
    lossEpochList.append(lossEpoch.item())
    if maxAccuracy<(100*accuracyTotal/testSize):
        maxAccuracy=100*accuracyTotal/testSize
        torch.save(leNet5, "model/LeNet/LeNetCIFAR10_%s.pth"%(int(maxAccuracy*100)))
    scheduler.step()


torch.save(leNet5, "model/LeNet/LeNetCIFAR10_%s.pth"%(int(100*accuracyTotal/testSize)))
epochList = np.array(epochList).reshape(-1,1)
lossEpochList = np.array(lossEpochList).reshape(-1,1)
accuracyTrainList = np.array(accuracyTrainList).reshape(-1,1)
accuracyTestList = np.array(accuracyTestList).reshape(-1,1)
dataArray = np.hstack((epochList,lossEpochList,accuracyTrainList,accuracyTestList))
dataFrame = pd.DataFrame(dataArray, columns=['epoch','loss','accuracyTrain','accuracyTest'])
dataFrame.to_csv("./log/LeNet/LeNetCIFAR10/LeNetCIFAR10P.csv",mode='w',index=None,encoding='utf_8_sig')
