import torchvision
from torchvision.datasets import MNIST,CIFAR10
from torch.utils.data import DataLoader


trainDataset = CIFAR10(root="D:\\WorkSpace\\DataSet\\CIFAR10", train=True, transform=torchvision.transforms.ToTensor())
testDataset = CIFAR10(root="D:\\WorkSpace\\DataSet\\CIFAR10", train=False, transform=torchvision.transforms.ToTensor())
batchSize = 100
trainSize = trainDataset.__len__()
testSize = testDataset.__len__()
trainLoader = DataLoader(trainDataset, batch_size = batchSize, shuffle=True)
testLoader = DataLoader(testDataset, batch_size = batchSize, shuffle=False)

if __name__ == "__main__":
    img, label = trainDataset[0]
    print(img.size())