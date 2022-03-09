from torchvision.datasets import ImageNet
import torch


imagenet_data = ImageNet('D:\\WorkSpace\\DataSet\\ImageNet')
print(imagenet_data[0])
data_loader = torch.utils.data.DataLoader(imagenet_data,
                                          batch_size=4,
                                          shuffle=True,
                                          num_workers=args.nThreads)