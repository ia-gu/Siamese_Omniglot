import torch
import torch.nn as nn
from torchvision import models

# SiameseNet作成
# EncoderはSiamese Neural Networks for one-shot Image Recognitionを参考に作成
class Siamese(nn.Module):

    def __init__(self):
        super(Siamese, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, 10),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 7),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 128, 4),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 4),
            nn.ReLU(),
        )
        self.liner = nn.Sequential(nn.Linear(256*6*6, 4096), nn.Sigmoid())
        self.out = nn.Linear(4096, 1)

    def forward_one(self, x):
        x = self.conv(x)
        # GPAの代わり
        x = x.view(x.size()[0], -1)
        x = self.liner(x)
        return x

    def forward(self, x1, x2):
        out1 = self.forward_one(x1)
        out2 = self.forward_one(x2)
        dis = torch.abs(out1 - out2)
        out = self.out(dis)
        return out

# EncoderをVGG16に変更したver
class VGG(nn.Module):
    def __init__(self, enc_dim):
        self.enc_dim = enc_dim
        super(VGG, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, self.enc_dim, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.enc_dim, self.enc_dim, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(self.enc_dim, self.enc_dim*2, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.enc_dim*2, self.enc_dim*2, 3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(self.enc_dim*2, self.enc_dim*4, 3, 1, 1),
            nn.ReLU(inplace=True), 
            nn.Conv2d(self.enc_dim*4, self.enc_dim*4, 3, 1, 1),
            nn.ReLU(inplace=True), 
            nn.Conv2d(self.enc_dim*4, self.enc_dim*4, 3,1,1),
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(2),

            nn.Conv2d(self.enc_dim*4, self.enc_dim*8, 3, 1, 1),
            nn.ReLU(inplace=True), 
            nn.Conv2d(self.enc_dim*8, self.enc_dim*8, 3, 1, 1),
            nn.ReLU(inplace=True), 
            nn.Conv2d(self.enc_dim*8, self.enc_dim*8, 3, 1, 1),
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(2),

            nn.Conv2d(self.enc_dim*8, self.enc_dim*8, 3, 1, 1),
            nn.ReLU(inplace=True), 
            nn.Conv2d(self.enc_dim*8, self.enc_dim*8, 3, 1, 1),
            nn.ReLU(inplace=True), 
            nn.Conv2d(self.enc_dim*8, self.enc_dim*8, 3, 1, 1),
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(2),
        )
        self.liner = nn.Sequential(nn.Linear(self.enc_dim*8*3*3, 4096), nn.Sigmoid())

        self.out = nn.Linear(4096, 1)

    def forward_one(self, x):
        x = self.conv(x)
        x = x.view(x.size()[0], -1)
        x = self.liner(x)
        return x

    def forward(self, x1, x2):
        out1 = self.forward_one(x1)
        out2 = self.forward_one(x2)
        dis = torch.abs(out1 - out2)
        out = self.out(dis)
        return out


def conv3x3(in_channels, out_channels, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                    padding=dilation, groups=groups, bias=True,
                    dilation=dilation)

def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=True)


class BasicBlock(nn.Module):
    #  Implementation of Basic Building Block

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()

        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity_x = x          # hold input for shortcut connection

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity_x = self.downsample(x)

        out += identity_x       # shortcut connection
        return self.relu(out)
        
class ResidualLayer(nn.Module):

     def __init__(self, num_blocks, in_channels, out_channels, block=BasicBlock):
         super(ResidualLayer, self).__init__()
         downsample = None
         if in_channels != out_channels:
             downsample = nn.Sequential(
                 conv1x1(in_channels, out_channels),
                 nn.BatchNorm2d(out_channels)
         )
         self.first_block = block(in_channels, out_channels, downsample=downsample)
         self.blocks = nn.ModuleList(block(out_channels, out_channels) for _ in range(num_blocks))

     def forward(self, x):
         out = self.first_block(x)
         for block in self.blocks:
             out = block(out)
         return out

class ResNet18(nn.Module):

    def __init__(self):
        super(ResNet18, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = ResidualLayer(2, in_channels=64, out_channels=64)
        self.layer2 = ResidualLayer(2, in_channels=64, out_channels=128)
        self.layer3 = ResidualLayer(
            2, in_channels=128, out_channels=256)
        self.layer4 = ResidualLayer(
            2, in_channels=256, out_channels=512)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.out = nn.Linear(512, 1)

    def forward_one(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avg_pool(out)
        out = out.view(out.size()[0], -1)

        return out

    def forward(self, x1, x2):
        out1 = self.forward_one(x1)
        out2 = self.forward_one(x2)
        dis = torch.abs(out1 - out2)
        out = self.out(dis)
        return out

# ResNet from PyTorch
# Much faster than the hand-made ResNet
class PretrainedResNet(nn.Module):
    def __init__(self, pretrain):
        super().__init__()
        self.pretrained_resnet = models.resnet18(pretrained=pretrain)
        self.pretrained_resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.out = nn.Linear(512, 1)

    def forward_one(self, x):
        x = self.pretrained_resnet.conv1(x)
        x = self.pretrained_resnet.bn1(x)
        x = self.pretrained_resnet.relu(x)
        x = self.pretrained_resnet.maxpool(x)
        x = self.pretrained_resnet.layer1(x)
        x = self.pretrained_resnet.layer2(x)
        x = self.pretrained_resnet.layer3(x)
        x = self.pretrained_resnet.layer4(x)
        x = self.pretrained_resnet.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, x1, x2):
        x1 = self.forward_one(x1)
        x2 = self.forward_one(x2)
        dis = torch.abs(x1-x2)
        out = self.out(dis)
        return out