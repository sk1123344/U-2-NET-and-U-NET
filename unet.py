import torch.nn as nn
import torch

class Conv2d(nn.Module):
    def __init__(self, in_ch, out_ch, ksize, stride=1, padding=0, groups=1, dilation=1, bias=False):
        '''
        Implementation of the conv with batch normalization and relu
        :param in_ch:int, the number of channels for input
        :param out_ch:int, the number of channels for output
        :param ksize:int or tuple, the size of the conv kernel
        :param stride: int or tuple, the stride for the conv
        :param padding:int, the size of padding
        :param groups:int, the number of groups for conv
        :param dilation:int, the dilation rate for conv
        :param bias:boolean, whether use bias
        '''
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=ksize, stride=stride, padding=padding, groups=groups, dilation=dilation, bias=bias)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x_ = self.conv(x)
        x_ = self.bn(x_)
        x_ = self.relu(x_)
        return x_


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        '''

        :param in_ch:
        :param out_ch:
        '''
        super(DoubleConv, self).__init__()
        # self.conv1 = nn.Sequential(
        #     nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(out_ch),
        #     nn.ReLU(inplace=True)
        # )
        self.conv1 = Conv2d(in_ch, out_ch, 3, stride=1, padding=1)
        self.conv2 = Conv2d(out_ch, out_ch, 3, stride=1, padding=1)

    def forward(self, x):
        x_ = self.conv1(x)
        x_ = self.conv2(x_)
        return x_


class Downsample(nn.Module):
    def __init__(self, in_ch, out_ch):
        '''

        :param in_ch:
        :param out_ch:
        '''
        super(Downsample, self).__init__()
        self.conv = DoubleConv(in_ch, out_ch)
        #self.down = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.down = nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=(3, 3), padding=2, stride=1, dilation=2)
        self.down = nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=(2, 2), padding=0, stride=2)

    def forward(self, x):
        x_ = self.conv(x)
        before_down = x_
        x_ = self.down(x_)
        return x_, before_down


class Upsample(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=False):
        '''

        :param in_ch:
        :param out_ch:
        '''
        super(Upsample, self).__init__()
        if bilinear:
            self.up = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor=2.0),
                Conv2d(in_ch, out_ch, 1, stride=1)
            )
        else:
            self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2, padding=0)

        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1_ = self.up(x1)
        x1_ = torch.cat([x2, x1_], dim=1)
        x1_ = self.conv(x1_)
        return x1_


class UNET(nn.Module):
    def __init__(self, in_ch=3, num_classes=1):
        '''

        :param in_ch:
        :param num_classes:
        '''
        super(UNET, self).__init__()
        self.down1 = Downsample(in_ch, 64)
        self.down2 = Downsample(64, 128)
        self.down3 = Downsample(128, 256)
        self.down4 = Downsample(256, 512)
        self.down5 = DoubleConv(512, 1024)

        self.up6 = Upsample(1024, 512)
        self.up7 = Upsample(512, 256)
        self.up8 = Upsample(256, 128)
        self.up9 = Upsample(128, 64)
        self.out = nn.Conv2d(64, num_classes, kernel_size=1, stride=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        net, down1_0 = self.down1(x)
        net, down2_0 = self.down2(net)
        net, down3_0 = self.down3(net)
        net, down4_0 = self.down4(net)
        net = self.down5(net)
        net = self.up6(net, down4_0)
        net = self.up7(net, down3_0)
        net = self.up8(net, down2_0)
        net = self.up9(net, down1_0)
        net = self.out(net)
        return self.sigmoid(net)
