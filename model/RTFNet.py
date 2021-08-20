# coding:utf-8
# By Yuxiang Sun, Aug. 2, 2019
# Email: sun.yuxiang@outlook.com

import torch
import torch.nn as nn
import torchvision.models as models

import fusion_strategy


class RTFNet(nn.Module):

    def __init__(self, n_class):
        super(RTFNet, self).__init__()
        resnet_raw_model1 = models.resnet50(pretrained=True)
        resnet_raw_model2 = models.resnet50(pretrained=True)
        resnet_raw_model3 = models.resnet50(pretrained=True)
        self.inplanes = 2048

        ########  fusion Module  ########
        self.inc_1 = inconv(3, 64)  # 假设输入通道数n_channels为3，输出通道数为64
        self.inc_2 = inconv(1, 64)  # 假设输入通道数n_channels为3，输出通道数为64
        self.down1_1 = down(64, 128)
        self.down1_2 = down(64, 128)
        self.down2_1 = down(128, 256)
        self.down2_2 = down(128, 256)
        self.down3_1 = down(256, 512)
        self.down3_2 = down(256, 512)
        self.up1 = up(768, 128)
        self.up2 = up(256, 64)
        self.up3 = up(128, 64)
        self.outc = outconv(64, 3)

        ########  Thermal ENCODER  ########

        self.encoder_thermal_conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.encoder_thermal_conv1.weight.data = torch.unsqueeze(torch.mean(resnet_raw_model1.conv1.weight.data, dim=1),
                                                                 dim=1)
        self.encoder_thermal_bn1 = resnet_raw_model1.bn1
        self.encoder_thermal_relu = resnet_raw_model1.relu
        self.encoder_thermal_maxpool = resnet_raw_model1.maxpool
        self.encoder_thermal_layer1 = resnet_raw_model1.layer1
        self.encoder_thermal_layer2 = resnet_raw_model1.layer2
        self.encoder_thermal_layer3 = resnet_raw_model1.layer3
        self.encoder_thermal_layer4 = resnet_raw_model1.layer4

        ########  RGB ENCODER  ########

        self.encoder_rgb_conv1 = resnet_raw_model2.conv1
        self.encoder_rgb_bn1 = resnet_raw_model2.bn1
        self.encoder_rgb_relu = resnet_raw_model2.relu
        self.encoder_rgb_maxpool = resnet_raw_model2.maxpool
        self.encoder_rgb_layer1 = resnet_raw_model2.layer1
        self.encoder_rgb_layer2 = resnet_raw_model2.layer2
        self.encoder_rgb_layer3 = resnet_raw_model2.layer3
        self.encoder_rgb_layer4 = resnet_raw_model2.layer4

        ########  Fuse ENCODER  ########

        self.encoder_fuse_conv1 = resnet_raw_model3.conv1
        self.encoder_fuse_bn1 = resnet_raw_model3.bn1
        self.encoder_fuse_relu = resnet_raw_model3.relu
        self.encoder_fuse_maxpool = resnet_raw_model3.maxpool
        self.encoder_fuse_layer1 = resnet_raw_model3.layer1
        self.encoder_fuse_layer2 = resnet_raw_model3.layer2
        self.encoder_fuse_layer3 = resnet_raw_model3.layer3
        self.encoder_fuse_layer4 = resnet_raw_model3.layer4


        #######  1 x 1 bottleneck layer #########
        self.layers_1 = nn.Sequential(
            nn.Conv2d(64 * 3, 64, kernel_size=1, stride=1), nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.layers_2 = nn.Sequential(
            nn.Conv2d(256 * 3, 256, kernel_size=1, stride=1), nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.layers_3 = nn.Sequential(
            nn.Conv2d(512 * 3, 512, kernel_size=1, stride=1), nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.layers_4 = nn.Sequential(
            nn.Conv2d(1024 * 3, 1024, kernel_size=1, stride=1), nn.BatchNorm2d(1024),
            nn.ReLU()
        )
        self.layers_5 = nn.Sequential(
            nn.Conv2d(2048 * 3, 2048, kernel_size=1, stride=1), nn.BatchNorm2d(2048),
            nn.ReLU()
        )

        ########  DECODER  ########

        self.deconv1 = self._make_transpose_layer(TransBottleneck, self.inplanes // 4, 2,
                                                  stride=2)  # using // for python 3.6
        self.deconv2 = self._make_transpose_layer(TransBottleneck, self.inplanes // 2, 2,
                                                  stride=2)  # using // for python 3.6
        self.deconv3 = self._make_transpose_layer(TransBottleneck, self.inplanes // 2, 2,
                                                  stride=2)  # using // for python 3.6

        self.deconv4 = self._make_transpose_layer(TransBottleneck, self.inplanes // 2, 2,
                                                  stride=2)  # using // for python 3.6
        self.deconv5 = self._make_transpose_layer(TransBottleneck, n_class, 2,
                                                  stride=2)  # using // for python 3.6
        # self.deconv5 = self._make_transpose_layer(TransBottleneck, n_class, 2, stride=1)

    def _make_transpose_layer(self, block, planes, blocks, stride=1):

        upsample = None
        if stride != 1:
            upsample = nn.Sequential(
                nn.ConvTranspose2d(self.inplanes, planes, kernel_size=2, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(planes),
            )
        elif self.inplanes != planes:
            upsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(planes),
            )

        for m in upsample.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        layers = []

        for i in range(1, blocks):
            layers.append(block(self.inplanes, self.inplanes))

        layers.append(block(self.inplanes, planes, stride, upsample))
        self.inplanes = planes

        return nn.Sequential(*layers)

    def forward(self, input):
        rgb = input[:, :3]
        thermal = input[:, 3:4]

        ###########     fusion Module     ##########

        I1 = rgb
        I2 = thermal

        I1_1 = self.inc_1(I1)
        I2_1 = self.inc_2(I2)

        I1_2 = self.down1_1(I1_1)
        I2_2 = self.down1_2(I2_1)

        I1_3 = self.down2_1(I1_2)
        I2_3 = self.down2_2(I2_2)

        I1_4 = self.down3_1(I1_3)
        I2_4 = self.down3_2(I2_3)

        f1 = fusion_strategy.attention_fusion_weight(I1_1, I2_1, 'avg')
        f2 = fusion_strategy.attention_fusion_weight(I1_2, I2_2, 'avg')
        f3 = fusion_strategy.attention_fusion_weight(I1_3, I2_3, 'avg')
        f4 = fusion_strategy.attention_fusion_weight(I1_4, I2_4, 'avg')

        I = self.up1(f4, f3)

        I = self.up2(I, f2)

        fuse = I

        I = self.up3(I, f1)

        I = self.outc(I)



        ##############    Three Encoder Stream    ###############

        rgb = self.encoder_rgb_conv1(rgb)
        rgb = self.encoder_rgb_bn1(rgb)
        rgb = self.encoder_rgb_relu(rgb)

        thermal = self.encoder_thermal_conv1(thermal)
        thermal = self.encoder_thermal_bn1(thermal)
        thermal = self.encoder_thermal_relu(thermal)

        fuse = self.encoder_fuse_bn1(fuse)  #
        fuse = self.encoder_fuse_relu(fuse)  #

        fuse = torch.cat((rgb, thermal, fuse), dim=1)
        fuse = self.layers_1(fuse)

        rgb = self.encoder_rgb_maxpool(rgb)
        thermal = self.encoder_thermal_maxpool(thermal)
        fuse = self.encoder_fuse_maxpool(fuse)  #
        ######################################################################

        rgb = self.encoder_rgb_layer1(rgb)
        thermal = self.encoder_thermal_layer1(thermal)
        fuse = self.encoder_fuse_layer1(fuse)  #
        # print("f1.size() : ", fuse.size())
        fuse = torch.cat((rgb, thermal, fuse), dim=1)
        fuse = self.layers_2(fuse)
        ######################################################################

        rgb = self.encoder_rgb_layer2(rgb)
        thermal = self.encoder_thermal_layer2(thermal)
        fuse = self.encoder_fuse_layer2(fuse)  #

        fuse = torch.cat((rgb, thermal, fuse), dim=1)
        fuse = self.layers_3(fuse)
        ######################################################################

        rgb = self.encoder_rgb_layer3(rgb)
        thermal = self.encoder_thermal_layer3(thermal)
        fuse = self.encoder_fuse_layer3(fuse)  #

        fuse = torch.cat((rgb, thermal, fuse), dim=1)

        fuse = self.layers_4(fuse)
        # ######################################################################

        rgb = self.encoder_rgb_layer4(rgb)
        thermal = self.encoder_thermal_layer4(thermal)
        fuse = self.encoder_fuse_layer4(fuse)  #

        fuse = torch.cat((rgb, thermal, fuse), dim=1)
        fuse = self.layers_5(fuse)
        # ######################################################################

        #############     Dencoder Stream    ###############

        fuse = self.deconv1(fuse)

        fuse = self.deconv2(fuse)

        fuse = self.deconv3(fuse)

        fuse = self.deconv4(fuse)

        fuse = self.deconv5(fuse)

        return fuse

class TransBottleneck(nn.Module):

    def __init__(self, inplanes, planes, stride=1, upsample=None):
        super(TransBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        if upsample is not None and stride != 1:
            self.conv3 = nn.ConvTranspose2d(planes, planes, kernel_size=2, stride=stride, padding=0, bias=False)
        else:
            self.conv3 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)

        self.bn3 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.upsample = upsample
        self.stride = stride

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.upsample is not None:
            residual = self.upsample(x)

        out += residual
        out = self.relu(out)

        return out



class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            # 以第一层为例进行讲解
            # 输入通道数in_ch，输出通道数out_ch，卷积核设为kernal_size 3*3，padding为1，stride为1，dilation=1
            # 所以图中H*W能从572*572 变为 570*570,计算为570 = ((572 + 2*padding - dilation*(kernal_size-1) -1) / stride ) +1
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),  # 进行批标准化，在训练时，该层计算每次输入的均值与方差，并进行移动平均
            nn.ReLU(inplace=True),  # 激活函数
            nn.Conv2d(out_ch, out_ch, 3, padding=1),  # 再进行一次卷积，从570*570变为 568*568
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


# 实现左边第一行的卷积
class inconv(nn.Module):  #
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)  # 输入通道数in_ch为3， 输出通道数out_ch为64

    def forward(self, x):
        x = self.conv(x)
        return x


# 实现左边的向下池化操作，并完成另一层的卷积
class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


import torch.nn.functional as F
# 实现右边的向上的采样操作，并完成该层相应的卷积操作
class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:  # 声明使用的上采样方法为bilinear——双线性插值，默认使用这个值，计算方法为 floor(H*scale_factor)，所以由28*28变为56*56
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:  # 否则就使用转置卷积来实现上采样，计算式子为 （Height-1）*stride - 2*padding -kernal_size +output_padding
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):  # x2是左边特征提取传来的值
        # 第一次上采样返回56*56，但是还没结束
        x1 = self.up(x1)
        # print("x1: ", x1.size())
        # print("x2: ", x2.size())

        # input is CHW, [0]是batch_size, [1]是通道数，更改了下，与源码不同
        diffY = x1.size()[2] - x2.size()[2]  # 得到图像x2与x1的H的差值，56-64=-8
        diffX = x1.size()[3] - x2.size()[3]  # 得到图像x2与x1的W差值，56-64=-8
        # 用第一次上采样为例,即当上采样后的结果大小与右边的特征的结果大小不同时，通过填充来使x2的大小与x1相同
        # 对图像进行填充(-4,-4,-4,-4),左右上下都缩小4，所以最后使得64*64变为56*56
        x2 = F.pad(x2, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))

        # for padding issues, see

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


# 实现右边的最高层的最右边的卷积
class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x
