import torchvision
from torchvision.datasets import MNIST,CIFAR10
from torch.utils.data import DataLoader
from torchvision import transforms


transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
# trainDataset = CIFAR10(root="D:\\WorkSpace\\DataSet\\CIFAR10", train=True, transform=torchvision.transforms.ToTensor())
# testDataset = CIFAR10(root="D:\\WorkSpace\\DataSet\\CIFAR10", train=False, transform=torchvision.transforms.ToTensor())
# batchSize = 100
trainDataset = CIFAR10(root="D:\\WorkSpace\\DataSet\\CIFAR10", train=True, transform=transform_train)
testDataset = CIFAR10(root="D:\\WorkSpace\\DataSet\\CIFAR10", train=False, transform=transform_test)
batchSize = 128
trainSize = trainDataset.__len__()
testSize = testDataset.__len__()
trainLoader = DataLoader(trainDataset, batch_size = batchSize, shuffle=True)
testLoader = DataLoader(testDataset, batch_size = batchSize, shuffle=False)

if __name__ == "__main__":
    img, label = trainDataset[0]
    print(img.size())