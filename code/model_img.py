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
import torch.nn.functional as F
from mask_estimation import VGG

class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)

        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss

        return Dice_BCE

def weights_init(m):
    if type(m) in [nn.Linear]:
        # print("setting custom wts")
        # m.weight.models.register_hook(lambda grad: print(grad))
        m.weight.data = torch.randn(m.weight.data.shape).float() * math.sqrt(2 / m.weight.data.shape[1])
        m.bias.data = torch.randn(m.bias.data.shape).float() * math.sqrt(2 / m.weight.data.shape[1])
        # print(m.weight.models, m.bias.models)
    elif type(m) in [nn.Conv2d]:
        # print("setting custom wts")
        # m.weight.models.register_hook(lambda grad: print(grad))
        m.weight.data = torch.randn(m.weight.data.shape).float() * math.sqrt(
            2 / (m.weight.data.shape[1] * m.weight.data.shape[2] * m.weight.data.shape[3]))
        m.bias.data = torch.randn(m.bias.data.shape).float() * math.sqrt(
            2 / (m.weight.data.shape[1] * m.weight.data.shape[2] * m.weight.data.shape[3]))
        # print(m.weight.models, m.bias.models)


class upmodel(nn.Module):
    def __init__(self, inp_dim):
        super(upmodel, self).__init__()
        self.inp_dim = inp_dim
        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(in_channels=inp_dim[0], out_channels=int(inp_dim[0] / 2), kernel_size=5, stride=1,
                               padding=2)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=int(inp_dim[0] / 2), out_channels=int(inp_dim[0] / 4), kernel_size=3,
                               stride=1, padding=1)
        self.conv_side = nn.Conv2d(in_channels=int(inp_dim[0]), out_channels=int(inp_dim[0] / 4), kernel_size=5,
                                   stride=1, padding=2)

    def forward(self, inp):
        x = torch.zeros(inp.shape[0], inp.shape[1], 2 * inp.shape[2], 2 * inp.shape[3]).float().cuda()
        x[:, :, ::2, ::2] = inp
        # print(x)
        # Without using maxUnpool3d
        x1 = self.conv1(x)
        x1 = self.relu(x1)
        x1 = self.conv2(x1)
        x2 = self.conv_side(x)
        return torch.cat((x1, x2), dim=1)


class fullmodel(nn.Module):
    def __init__(self):
        super(fullmodel, self).__init__()

        self.res = models.resnet50(pretrained=True)
        self.res = nn.Sequential(*(list(self.res.children())[:-2]))
        for param in self.res.parameters():
            param.requires_grad = False
        self.first_conv = nn.Conv2d(in_channels=2048, out_channels=1024, kernel_size=1, stride=1, padding=0)

        self.batchNorm = nn.BatchNorm2d(1024)
        # self.upsample1 = upmodel([1024,7,9])
        self.upsample1 = upmodel([1024, 7, 7])
        # self.upsample2 = upmodel([512,14,18])
        self.upsample2 = upmodel([512, 14, 14])
        # self.upsample3 = upmodel([256,28,36])
        self.upsample3 = upmodel([256, 28, 28])

        self.upsample4 = upmodel([128, 56, 56])
        self.upsample5 = upmodel([64, 112, 112])
        # self.convlayer = nn.Conv2d(in_channels=128,out_channels=1,kernel_size=3,stride=1,padding=0)
        self.convlayer = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1, stride=1, padding=0)

        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.upsample1.apply(weights_init)
        self.upsample2.apply(weights_init)
        self.upsample3.apply(weights_init)
        self.first_conv.apply(weights_init)
        self.convlayer.apply(weights_init)
        print("init custom wts ok!")

    def forward(self, inp):
        x = self.res(inp)
        # print("# res shape " , x.shape )
        x = self.first_conv(x)

        x = self.batchNorm(x)
        # print("# first_conv shape " , x.shape )
        x = self.upsample1(x)
        # print("# upsample1 shape " , x.shape )
        x = self.upsample2(x)
        # print("# upsample2 shape " , x.shape )
        x = self.upsample3(x)
        # print("# upsample3 shape " , x.shape )
        x = self.upsample4(x)
        # print("# upsample4 shape " , x.shape )
        x = self.upsample5(x)
        # print("# upsample5 shape " , x.shape )
        x = self.convlayer(x)
        # print("# convlayer shape " , x.shape )

        x = self.relu(x)

        # x = x[:,:,:-1,:]
        # print('# x shape ', x.shape )
        x1 = torch.zeros(x.shape[0], x.shape[2], x.shape[3]).cuda()
        for i in range(x.shape[0]):
            x1[i] = x[i, 0, :, :]
        # print('# x1 shape ', x1.shape )
        return x1


class finemodel(nn.Module):
    def __init__(self, fmpath):
        super(finemodel, self).__init__()
        self.coarse_model = fullmodel()
        self.coarse_model = torch.load(fmpath)
        self.coarse_model.float().cuda()
        # self.coarse_model.load_state_dict(torch.load("../bins/residual/BestResidualmodel2.pt"))
        for param in self.coarse_model.parameters():
            param.requires_grad = False
        self.conv_start = nn.Conv2d(in_channels=3, out_channels=255, kernel_size=1)
        self.relu_start = nn.ReLU()
        self.batch_start = nn.BatchNorm2d(255)
        self.pool = nn.MaxPool2d(kernel_size=2)

        self.depth_conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=9, stride=2, padding=4)
        self.depth_relu = nn.ReLU()
        self.depth_batch = nn.BatchNorm2d(1)
        self.depth_pool = nn.MaxPool2d(kernel_size=2)

        self.conv1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.relu1 = nn.ReLU()
        self.batch1 = nn.BatchNorm2d(256)

        self.conv2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.relu2 = nn.ReLU()
        self.batch2 = nn.BatchNorm2d(256)

        self.conv3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        self.relu3 = nn.ReLU()
        self.batch3 = nn.BatchNorm2d(256)

        self.conv4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2)
        self.relu4 = nn.ReLU()
        self.batch4 = nn.BatchNorm2d(256)

        self.conv5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1)
        self.pool5 = nn.MaxPool2d(kernel_size=2)
        self.relu5 = nn.ReLU()
        self.batch5 = nn.BatchNorm2d(256)

        self.conv6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1)
        self.pool6 = nn.MaxPool2d(kernel_size=2)
        self.relu6 = nn.ReLU()
        self.batch6 = nn.BatchNorm2d(256)

        self.linear = nn.Linear(2304, 256)
        self.init_weights()

    def init_weights(self):
        self.conv_start.apply(weights_init)
        self.conv1.apply(weights_init)
        self.conv2.apply(weights_init)

    def forward(self, inp):
        # inp : B, 3, 224, 224
        # depth: B,1, 224, 224
        depth = self.coarse_model(inp)
        x1 = torch.zeros(depth.shape[0], 1, depth.shape[1], depth.shape[2]).cuda()
        for i in range(depth.shape[0]):
            x1[i, 0, :, :] = depth[i, :, :]
        x = self.conv_start(inp)

        x = self.relu_start(x)
        x = self.batch_start(x)

        x = torch.cat((x, x1), dim=1)

        x = self.conv1(x)
        x = self.pool1(x)
        x = self.relu1(x)
        x = self.batch1(x)

        x = self.conv2(x)
        x = self.pool2(x)
        x = self.relu2(x)
        x = self.batch2(x)

        x = self.conv3(x)
        x = self.pool3(x)
        x = self.relu3(x)
        x = self.batch3(x)

        x = self.conv4(x)
        x = self.pool4(x)
        x = self.relu4(x)
        x = self.batch4(x)

        x = self.conv5(x)
        x = self.pool5(x)
        x = self.relu5(x)
        x = self.batch5(x)

        x = self.conv6(x)
        x = self.pool6(x)
        x = self.relu6(x)
        x = self.batch6(x)

        # x1 = torch.zeros(x.shape[0],x.shape[2],x.shape[3]).cuda()
        # for i in range(x.shape[0]):
        #   x1[i] = x[i,0,:,:]

        x1 = x.flatten(start_dim=1)
        x1 = self.linear(x1)

        return x1




class CNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CNN, self).__init__()
        self.hidden = 512
        conv = []
        conv.append(nn.Conv2d(in_channels=in_channels, out_channels=64,kernel_size=3, stride=1, padding=1))
        conv.append(nn.ReLU())
        conv.append(nn.MaxPool2d(kernel_size=2, stride=2))
        conv.append(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1))
        conv.append(nn.ReLU())
        conv.append(nn.MaxPool2d(kernel_size=2, stride=2))
        conv.append(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1))
        conv.append(nn.ReLU())
        conv.append(nn.MaxPool2d(kernel_size=2, stride=2))
        conv.append(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1))
        conv.append(nn.ReLU())
        conv.append(nn.MaxPool2d(kernel_size=2, stride=2))
        conv.append(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1))
        conv.append(nn.ReLU())
        conv.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.conv = nn.Sequential(*conv)

    def forward(self ,x):
        return self.conv(x)

class ImageParse(nn.Module):
    def __init__(self):
        super(ImageParse, self).__init__()
        self.res = VGG()
    def forward(self, x):
        x = self.res(x)
        return x

class ImageEncoder(nn.Module):
    def __init__(self, mask_estimator=None, code_length=50):
        super(ImageEncoder, self).__init__()
        # input image is 224 x 224

        # image parse

        if mask_estimator is not None:
            self.image_parse = ImageParse()
            self.mask_estimator = mask_estimator
            self.cnn = CNN(in_channels=1, out_channels=7)
        else:
            self.encoder = ImageParse()
        # fcn
        fcn = []
        fcn.append(nn.Linear(512 * 7 * 7 * 2, 4096))
        fcn.append(nn.Linear(4096, code_length))
        self.fcn = nn.Sequential(*fcn)

    def forward(self, x):
        parsing = self.image_parse(x)
        # print(parsing.shape)
        mask_result = self.mask_estimator(x)
        mask_result = torch.unsqueeze(mask_result, dim=1)
        mask_result = self.cnn(mask_result)
        # print(mask_result.shape)
        encoder = torch.cat((parsing, mask_result), dim=1)
        encoder = encoder.view(encoder.shape[0],-1)
        # print(encoder.shape)
        output = self.fcn(encoder)
        # print(output.shape)
        return output
