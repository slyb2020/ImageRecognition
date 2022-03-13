import os

import pandas as pd
import wx

import images
from DatasetLabelProcess import *
# from YOLOv1Algorithm import *
import wx.lib.scrolledpanel as scrolled
# from MNIST_Dataset import testDataset
from torchvision.datasets import MNIST
from errorDataFinder import ErrorTest
from ID_DEFINE import *
import numpy as np
from wx.lib import plot as wxplot
from LeNetPanel import *

class VGGNetMNISTPanel(LeNetMNISTPanel):
    def __init__(self, parent, log):
        super(VGGNetMNISTPanel, self).__init__(parent,log)

class VGGNetCIFAR10Panel(LeNetMNISTPanel):
    def __init__(self, parent, log):
        super(VGGNetCIFAR10Panel, self).__init__(parent,log)

class VGGNetPanel(wx.Panel):
    def __init__(self, parent, log):
        wx.Panel.__init__(self, parent)
        self.log = log
        self.notebook = wx.Notebook(self, -1, size=(21, 21), style=
        # wx.BK_DEFAULT
        # wx.BK_TOP
        wx.BK_BOTTOM
                                    # wx.BK_LEFT
                                    # wx.BK_RIGHT
                                    # | wx.NB_MULTILINE
                                    )
        il = wx.ImageList(16, 16)
        il.Add(images._rt_smiley.GetBitmap())
        self.total_page_num = 0
        self.notebook.AssignImageList(il)
        idx2 = il.Add(images.GridBG.GetBitmap())
        idx3 = il.Add(images.Smiles.GetBitmap())
        idx4 = il.Add(images._rt_undo.GetBitmap())
        idx5 = il.Add(images._rt_save.GetBitmap())
        idx6 = il.Add(images._rt_redo.GetBitmap())
        hbox = wx.BoxSizer()
        self.vggNetIntroductionPanel = wx.Panel(self.notebook, style=wx.BORDER_THEME)
        self.notebook.AddPage(self.vggNetIntroductionPanel, "VGGNet神经网络模型介绍")
        self.vggNetNMISTlPanel = VGGNetMNISTPanel(self.notebook, self.log)
        self.notebook.AddPage(self.vggNetNMISTlPanel, "VGGNet在MNIST上的应用")
        self.vggNetCIFAR10Panel = VGGNetCIFAR10Panel(self.notebook, self.log)
        self.notebook.AddPage(self.vggNetCIFAR10Panel, "VGGNet在CIFAR10上的应用")
        hbox.Add(self.notebook, 1, wx.EXPAND)
        self.SetSizer(hbox)

        self.vggNetNMISTlPanel.ShowModelStructure("./model/vggNet.mdl")
        self.vggNetCIFAR10Panel.ShowModelStructure("./model/vggNet.mdl")

        self.vggNetCIFAR10Panel.DrawEpochAccuracyLossCurve("log/VGG/VGG16/VGG16CIFAR10.csv","log/VGG/VGG16/VGG16CIFAR10P.csv")
        # self.leeNetCIFAR10Panel.DrawEpochAccuracyLossCurve("log/LeNet/LeeNetCIFAR10/LeeNetCIFAR10.csv")
    #     self.Bind(wx.EVT_BUTTON, self.OnPictureButton)
    #
    # def OnPictureButton(self, event):
    #     id = event.GetId()
    #     # if id in self.trainDatasetPanel.buttonIdList:
    #     #     self.index = self.trainDatasetPanel.buttonIdList.index(id)
    #     #     self.leftPanel.Refresh()
