import torch.nn as nn
import torch
import numpy as np
import random
##################
## ARO2021 LABS ##
##################

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.relu = nn.LeakyReLU(0.1, inplace=True)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.pool2 = nn.MaxPool2d(3, 1, 1)
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(256)
        self.bn6 = nn.BatchNorm2d(512)
        self.bn7 = nn.BatchNorm2d(1024)
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1, bias=False)
        self.conv2 = nn.Conv2d(16, 32, 3, 1, 1, bias=False)    
        self.conv3 = nn.Conv2d(32, 64, 3, 1, 1, bias=False)       
        self.conv4 = nn.Conv2d(64, 128, 3, 1, 1, bias=False)       
        self.conv5 = nn.Conv2d(128, 256, 3, 1, 1, bias=False)      
        self.conv6 = nn.Conv2d(256, 512, 3, 1, 1, bias=False)     
        self.conv7 = nn.Conv2d(512, 1024, 3, 1, 1, bias=False)      
        self.conv8 = nn.Conv2d(1024, 1, 1, 1)

    def forward(self, input):
        x = self.pool1(self.relu(self.bn1(self.conv1(input))))
        x = self.pool1(self.relu(self.bn2(self.conv2(x))))
        x = self.pool1(self.relu(self.bn3(self.conv3(x))))
        x = self.pool2(self.relu(self.bn4(self.conv4(x))))
        x = self.pool2(self.relu(self.bn5(self.conv5(x))))
        x = self.pool2(self.relu(self.bn6(self.conv6(x))))
        x = self.relu(self.bn7(self.conv7(x)))
        x = (self.conv8(x))
        return x

    def train_only_last(self, bool):
        if bool == True:
            i = 0
            for p in self.parameters():
                i += 1
                if i > 21:
                    break
                else:
                    p.requires_grad = False
        else:
            i = 0
            for p in self.parameters():
                i += 1
                if i > 21:
                    break
                else:
                    p.requires_grad = True


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self, flip=False):
        self.flip = flip

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        mean = [118, 117, 117]
        std = [57, 58, 60]
        if self.flip & (random.random() > 0.5):
            image = np.flip(image, 1)
            label = np.flip(label, 1)

        image = image.astype(np.float32)
        image = (image - mean) / std
        image = image.transpose((2, 0, 1))
        image = torch.from_numpy(image.astype(np.float32))
        label = torch.from_numpy(label.astype(np.float32))
        return {'image': image,
                'label': label}
        
