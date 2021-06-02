import torch
import imageio
import torch.nn as nn
import glob
import time
import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
import math
import random
import torchvision.models as models

# input 224 x 224 x 3
class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        # block 1
        block1 = []
        block1.append(nn.Conv2d(in_channels=3, out_channels=64, padding=1, kernel_size=3, stride=1))
        block1.append(nn.ReLU())
        block1.append(nn.Conv2d(in_channels=64, out_channels=64, padding=1, kernel_size=3, stride=1))
        block1.append(nn.ReLU())
        block1.append(nn.MaxPool2d(kernel_size=2, stride=2)) # pool 1
        self.block1 = nn.Sequential(*block1)

        # block 2
        block2 = []
        block2.append(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1))
        block2.append(nn.ReLU())
        block2.append(nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1))
        block2.append(nn.ReLU())
        block2.append(nn.MaxPool2d(kernel_size=2, stride=2)) # pool 2
        self.block2 = nn.Sequential(*block2)
        # block 3
        block3 = []
        block3.append(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=1))
        block3.append(nn.ReLU())
        block3.append(nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1))
        block3.append(nn.ReLU())
        block3.append(nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1))
        block3.append(nn.ReLU())
        block3.append(nn.MaxPool2d(kernel_size=2, stride=2)) # pool 3
        self.block3 = nn.Sequential(*block3)
        # block 4
        block4 = []
        block4.append(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1, stride=1))
        block4.append(nn.ReLU())
        block4.append(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1))
        block4.append(nn.ReLU())
        block4.append(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1))
        block4.append(nn.ReLU())
        block4.append(nn.MaxPool2d(kernel_size=2, stride=2)) # pool 4
        self.block4 = nn.Sequential(*block4)
        # block 5
        block5 = []
        block5.append(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1))
        block5.append(nn.ReLU())
        block5.append(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1))
        block5.append(nn.ReLU())
        block5.append(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1))
        block5.append(nn.ReLU())
        block5.append(nn.MaxPool2d(kernel_size=2, stride=2)) # pool 5
        self.block5 = nn.Sequential(*block5)

    def forward(self, x):
        self.p1 = self.block1(x)
        self.p2 = self.block2(self.p1)
        self.p3 = self.block3(self.p2)
        self.p4 = self.block4(self.p3)
        self.p5 = self.block5(self.p4)
        return self.p5

class Fusion(nn.Module):
    def __init__(self, in_channels, out_channels, pooling_factor, kernel_size=5, stride=1):
        super(Fusion, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding= 1, stride=stride)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=pooling_factor)
    def forward(self, x):
        # return self.upsample(self.conv(x))
        return self.upsample(self.conv(x))

class Refinement(nn.Module):
    def __init__(self):
        super(Refinement, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=9, padding=1, stride=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=256, kernel_size=5, padding=0, stride=1) # 52 x 52 107
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=5, padding=1, stride=1) # 50 x 50 105
        self.conv4 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=5, padding=1, stride=1) # 48 x 48 103
        self.conv5 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=5, padding=1, stride=1) # 46 x 46 101
        self.conv6 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=5, padding=1, stride=1) # 44 x 44 99
        self.conv7 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=5, padding=1, stride=1) # 42 x 42 97
        self.conv8 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=5, padding=1, stride=1) # 40 x 40 95
        self.conv9 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=5, padding=1, stride=1) # 38 x 38 93
        self.conv10 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=5, padding=1, stride=1) # 512 x 36 x 36 91

        self.fusion1 = Fusion(in_channels=256, out_channels=256, pooling_factor=2)
        self.fusion2 = Fusion(in_channels=512, out_channels=512, pooling_factor=4)
        self.fusion3 = Fusion(in_channels=3, out_channels=512, pooling_factor=4)

    def forward(self, x, p3, p4, r):

        p3 = self.fusion1(p3)
        print('p3.shape ', p3.shape)
        x = self.conv2(self.pool(self.conv1(x)))
        print('x shape ', x.shape)
        exit()
        x = torch.cat((self.conv2(x), p3), dim=3)
        # x = self.conv2(x) + p3  # 1x64x52x52
        x = self.conv3(x)
        print('x shape ', x.shape)
        exit()

        p4 = self.fusion2(p4)
        r = self.fusion3(r)


        x = self.conv4(self.conv3(x)) + p4

        x = self.conv6(self.conv5(x)) + r

        x = self.conv10(self.conv9(self.conv8(self.conv7(x))))

        exit()
        return x


# two scale
class MaskEstimation(nn.Module):
    def __init__(self, batch_size=3):
        self.bs = batch_size
        super(MaskEstimation, self).__init__()
        self.vgg = VGG()
        fcn = []
        fcn.append(nn.Linear(in_features=512 * 7 * 7 , out_features= 10240))
        fcn.append(nn.Linear(in_features=10240, out_features=3 * 56 * 56))
        self.fcn = nn.Sequential(*fcn)
        self.refinement = Refinement()

    def forward(self, x):
        p1, p2, p3, p4, p5 = self.vgg(x)
        #p3 256x28x28 ,p4 512x14x14 512 x 7 x 7
        print('current p3 shape ',p3.shape,  p4.shape)
        # p5 = p5.view(-1, 512 * 7 * 7)
        # reshape = self.fcn(p5)
        # reshape = reshape.view(-1,3, 56, 56)
        x = self.refinement(x, p3, p4, p5)
        return x

if __name__ == '__main__':
    randomImg = torch.rand(1, 3, 427, 561)
    # print(randomImg.shape)
    me = MaskEstimation()
    depth = me(randomImg)
    # print(depth.shape)
