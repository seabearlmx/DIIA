"""
# Code Adapted from:
# https://github.com/sthalles/deeplab_v3
#
# MIT License
#
# Copyright (c) 2018 Thalles Santos Silva
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
"""
import logging
import torch
from torch import nn
import math
import numpy as np
import torch.nn.functional as F
from network import Resnet
from network import Mobilenet
from network import Shufflenet
from network.cov_settings import CovMatrix_ISW, CovMatrix_IRW
from network.instance_whitening import instance_whitening_loss, get_covariance_matrix
from network.mynn import initialize_weights, Norm2d, Upsample, freeze_weights, unfreeze_weights
from network.instance_whitening import InstanceWhitening

import torchvision.models as models
from network.layer import MultiSpectralAttentionLayer
affine_par = True


class _AtrousSpatialPyramidPoolingModule(nn.Module):
    """
    operations performed:
      1x1 x depth
      3x3 x depth dilation 6
      3x3 x depth dilation 12
      3x3 x depth dilation 18
      image pooling
      concatenate all together
      Final 1x1 conv
    """

    def __init__(self, in_dim, reduction_dim=256, output_stride=16, rates=(6, 12, 18)):
        super(_AtrousSpatialPyramidPoolingModule, self).__init__()

        # Check if we are using distributed BN and use the nn from encoding.nn
        # library rather than using standard pytorch.nn
        print("output_stride = ", output_stride)
        if output_stride == 8:
            rates = [2 * r for r in rates]
        elif output_stride == 4:
            rates = [4 * r for r in rates]
        elif output_stride == 16:
            pass
        elif output_stride == 32:
            rates = [r // 2 for r in rates]
        else:
            raise 'output stride of {} not supported'.format(output_stride)

        self.features = []
        # 1x1
        self.features.append(
            nn.Sequential(nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                          Norm2d(reduction_dim), nn.ReLU(inplace=True)))
        # other rates
        for r in rates:
            self.features.append(nn.Sequential(
                nn.Conv2d(in_dim, reduction_dim, kernel_size=3,
                          dilation=r, padding=r, bias=False),
                Norm2d(reduction_dim),
                nn.ReLU(inplace=True)
            ))
        self.features = torch.nn.ModuleList(self.features)

        # img level features
        self.img_pooling = nn.AdaptiveAvgPool2d(1)
        self.img_conv = nn.Sequential(
            nn.Conv2d(in_dim, 256, kernel_size=1, bias=False),
            Norm2d(256), nn.ReLU(inplace=True))

    def forward(self, x):
        x_size = x.size()

        img_features = self.img_pooling(x)
        img_features = self.img_conv(img_features)
        img_features = Upsample(img_features, x_size[2:])
        out = img_features

        for f in self.features:
            y = f(x)
            out = torch.cat((out, y), 1)
        return out


class DeepV3Plus(nn.Module):
    """
    Implement DeepLab-V3 model
    A: stride8
    B: stride16
    with skip connections
    """

    def __init__(self, num_classes, trunk='resnet-101', criterion=None, criterion_aux=None, L1_loss=None,
                variant='D', skip='m1', skip_num=48, args=None):
        super(DeepV3Plus, self).__init__()
        self.criterion = criterion
        self.criterion_aux = criterion_aux
        self.l1_loss = L1_loss
        self.variant = variant
        self.args = args
        self.trunk = trunk

        if trunk == 'shufflenetv2':
            channel_1st = 3
            channel_2nd = 24
            channel_3rd = 116
            channel_4th = 232
            prev_final_channel = 464
            final_channel = 1024
            resnet = Shufflenet.shufflenet_v2_x1_0(pretrained=True, iw=self.args.wt_layer)

            class Layer0(nn.Module):
                def __init__(self, iw):
                    super(Layer0, self).__init__()
                    self.layer = nn.Sequential(resnet.conv1, resnet.maxpool)
                    self.instance_norm_layer = resnet.instance_norm_layer1
                    self.iw = iw

                def forward(self, x_tuple):
                    if len(x_tuple) == 2:
                        w_arr = x_tuple[1]
                        x = x_tuple[0]
                    else:
                        print("error in shufflnet layer 0 forward path")
                        return

                    x = self.layer[0][0](x)
                    if self.iw >= 1:
                        if self.iw == 1 or self.iw == 2:
                            x, w = self.instance_norm_layer(x)
                            w_arr.append(w)
                        else:
                            x = self.instance_norm_layer(x)
                    else:
                        x = self.layer[0][1](x)

                    x = self.layer[0][2](x)
                    x = self.layer[1](x)

                    return [x, w_arr]

            class Layer4(nn.Module):
                def __init__(self, iw):
                    super(Layer4, self).__init__()
                    self.layer = resnet.conv5
                    self.instance_norm_layer = resnet.instance_norm_layer2
                    self.iw = iw

                def forward(self, x_tuple):
                    if len(x_tuple) == 2:
                        w_arr = x_tuple[1]
                        x = x_tuple[0]
                    else:
                        print("error in shufflnet layer 4 forward path")
                        return

                    x = self.layer[0](x)
                    if self.iw >= 1:
                        if self.iw == 1 or self.iw == 2:
                            x, w = self.instance_norm_layer(x)
                            w_arr.append(w)
                        else:
                            x = self.instance_norm_layer(x)
                    else:
                        x = self.layer[1](x)
                    x = self.layer[2](x)

                    return [x, w_arr]


            self.layer0 = Layer0(iw=self.args.wt_layer[2])
            self.layer1 = resnet.stage2
            self.layer2 = resnet.stage3
            self.layer3 = resnet.stage4
            self.layer4 = Layer4(iw=self.args.wt_layer[6])

            if self.variant == 'D':
                for n, m in self.layer2.named_modules():
                    if isinstance(m, nn.Conv2d) and m.stride==(2,2):
                        m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
                for n, m in self.layer3.named_modules():
                    if isinstance(m, nn.Conv2d) and m.stride==(2,2):
                        m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
            elif self.variant == 'D16':
                for n, m in self.layer3.named_modules():
                    if isinstance(m, nn.Conv2d) and m.stride==(2,2):
                        m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
            else:
                # raise 'unknown deepv3 variant: {}'.format(self.variant)
                print("Not using Dilation ")

        elif trunk == 'mnasnet_05' or trunk == 'mnasnet_10':

            if trunk == 'mnasnet_05':
                resnet = models.mnasnet0_5(pretrained=True)
                channel_1st = 3
                channel_2nd = 16
                channel_3rd = 24
                channel_4th = 48
                prev_final_channel = 160
                final_channel = 1280

                print("# of layers", len(resnet.layers))
                self.layer0 = nn.Sequential(resnet.layers[0],resnet.layers[1],resnet.layers[2],
                                            resnet.layers[3],resnet.layers[4],resnet.layers[5],resnet.layers[6],resnet.layers[7])   # 16
                self.layer1 = nn.Sequential(resnet.layers[8], resnet.layers[9]) # 24, 40
                self.layer2 = nn.Sequential(resnet.layers[10], resnet.layers[11])   # 48, 96
                self.layer3 = nn.Sequential(resnet.layers[12], resnet.layers[13]) # 160, 320
                self.layer4 = nn.Sequential(resnet.layers[14], resnet.layers[15], resnet.layers[16])  # 1280
            else:
                resnet = models.mnasnet1_0(pretrained=True)
                channel_1st = 3
                channel_2nd = 16
                channel_3rd = 40
                channel_4th = 96
                prev_final_channel = 320
                final_channel = 1280

                print("# of layers", len(resnet.layers))
                self.layer0 = nn.Sequential(resnet.layers[0],resnet.layers[1],resnet.layers[2],
                                            resnet.layers[3],resnet.layers[4],resnet.layers[5],resnet.layers[6],resnet.layers[7])   # 16
                self.layer1 = nn.Sequential(resnet.layers[8], resnet.layers[9]) # 24, 40
                self.layer2 = nn.Sequential(resnet.layers[10], resnet.layers[11])   # 48, 96
                self.layer3 = nn.Sequential(resnet.layers[12], resnet.layers[13]) # 160, 320
                self.layer4 = nn.Sequential(resnet.layers[14], resnet.layers[15], resnet.layers[16])  # 1280

            if self.variant == 'D':
                for n, m in self.layer2.named_modules():
                    if isinstance(m, nn.Conv2d) and m.stride==(2,2):
                        m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
                for n, m in self.layer3.named_modules():
                    if isinstance(m, nn.Conv2d) and m.stride==(2,2):
                        m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
            elif self.variant == 'D16':
                for n, m in self.layer3.named_modules():
                    if isinstance(m, nn.Conv2d) and m.stride==(2,2):
                        m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
            else:
                # raise 'unknown deepv3 variant: {}'.format(self.variant)
                print("Not using Dilation ")
        elif trunk == 'mobilenetv2':
            channel_1st = 3
            channel_2nd = 16
            channel_3rd = 32
            channel_4th = 64

            # prev_final_channel = 160
            prev_final_channel = 320

            final_channel = 1280
            resnet = Mobilenet.mobilenet_v2(pretrained=True,
                    iw=self.args.wt_layer)
            self.layer0 = nn.Sequential(resnet.features[0],
                                        resnet.features[1])
            self.layer1 = nn.Sequential(resnet.features[2], resnet.features[3],
                                        resnet.features[4], resnet.features[5], resnet.features[6])
            self.layer2 = nn.Sequential(resnet.features[7], resnet.features[8], resnet.features[9], resnet.features[10])

            # self.layer3 = nn.Sequential(resnet.features[11], resnet.features[12], resnet.features[13], resnet.features[14], resnet.features[15], resnet.features[16])
            # self.layer4 = nn.Sequential(resnet.features[17], resnet.features[18])

            self.layer3 = nn.Sequential(resnet.features[11], resnet.features[12], resnet.features[13],
                                        resnet.features[14], resnet.features[15], resnet.features[16],
                                        resnet.features[17])
            self.layer4 = nn.Sequential(resnet.features[18])

            if self.variant == 'D':
                for n, m in self.layer2.named_modules():
                    if isinstance(m, nn.Conv2d) and m.stride==(2,2):
                        m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
                for n, m in self.layer3.named_modules():
                    if isinstance(m, nn.Conv2d) and m.stride==(2,2):
                        m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
            elif self.variant == 'D16':
                for n, m in self.layer3.named_modules():
                    if isinstance(m, nn.Conv2d) and m.stride==(2,2):
                        m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
            else:
                # raise 'unknown deepv3 variant: {}'.format(self.variant)
                print("Not using Dilation ")
        else:
            channel_1st = 3
            channel_2nd = 64
            channel_3rd = 256
            channel_4th = 512
            prev_final_channel = 1024
            final_channel = 2048

            if trunk == 'resnet-18':
                channel_1st = 3
                channel_2nd = 64
                channel_3rd = 64
                channel_4th = 128
                prev_final_channel = 256
                final_channel = 512
                resnet = Resnet.resnet18(wt_layer=self.args.wt_layer)
                resnet.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
            elif trunk == 'resnet-50':
                resnet = Resnet.resnet50(wt_layer=self.args.wt_layer)
                resnet.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
            elif trunk == 'resnet-101': # three 3 X 3
                resnet = Resnet.resnet101(pretrained=True, wt_layer=self.args.wt_layer)
                resnet.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu1,
                                            resnet.conv2, resnet.bn2, resnet.relu2,
                                            resnet.conv3, resnet.bn3, resnet.relu3, resnet.maxpool)
            elif trunk == 'resnet-152':
                resnet = Resnet.resnet152()
                resnet.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
            elif trunk == 'resnext-50':
                resnet = models.resnext50_32x4d(pretrained=True)
                resnet.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
            elif trunk == 'resnext-101':
                resnet = models.resnext101_32x8d(pretrained=True)
                resnet.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
            elif trunk == 'wide_resnet-50':
                resnet = models.wide_resnet50_2(pretrained=True)
                resnet.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
            elif trunk == 'wide_resnet-101':
                resnet = models.wide_resnet101_2(pretrained=True)
                resnet.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
            else:
                raise ValueError("Not a valid network arch")

            self.layer0 = resnet.layer0
            self.layer1, self.layer2, self.layer3, self.layer4 = \
                resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

            if self.variant == 'D':
                for n, m in self.layer3.named_modules():
                    if 'conv2' in n:
                        m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
                    elif 'downsample.0' in n:
                        m.stride = (1, 1)
                for n, m in self.layer4.named_modules():
                    if 'conv2' in n:
                        m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
                    elif 'downsample.0' in n:
                        m.stride = (1, 1)
            elif self.variant == 'D4':
                for n, m in self.layer2.named_modules():
                    if 'conv2' in n:
                        m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
                    elif 'downsample.0' in n:
                        m.stride = (1, 1)
                for n, m in self.layer3.named_modules():
                    if 'conv2' in n:
                        m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
                    elif 'downsample.0' in n:
                        m.stride = (1, 1)
                for n, m in self.layer4.named_modules():
                    if 'conv2' in n:
                        m.dilation, m.padding, m.stride = (8, 8), (8, 8), (1, 1)
                    elif 'downsample.0' in n:
                        m.stride = (1, 1)
            elif self.variant == 'D16':
                for n, m in self.layer4.named_modules():
                    if 'conv2' in n:
                        m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
                    elif 'downsample.0' in n:
                        m.stride = (1, 1)
            else:
                # raise 'unknown deepv3 variant: {}'.format(self.variant)
                print("Not using Dilation ")

        if self.variant == 'D':
            os = 8
        elif self.variant == 'D4':
            os = 4
        elif self.variant == 'D16':
            os = 16
        else:
            os = 32

        self.output_stride = os
        self.aspp = _AtrousSpatialPyramidPoolingModule(final_channel, 256,
                                                       output_stride=os)

        # self.bot_fine = nn.Sequential(
        #     nn.Conv2d(channel_3rd, 48, kernel_size=1, bias=False),
        #     Norm2d(48),
        #     nn.ReLU(inplace=True))

        self.bot_aspp = nn.Sequential(
            nn.Conv2d(1280, 256, kernel_size=1, bias=False),
            Norm2d(256),
            nn.ReLU(inplace=True))

        self.final1 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            Norm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            Norm2d(256),
            nn.ReLU(inplace=True))

        self.final2 = nn.Sequential(
            nn.Conv2d(256, num_classes, kernel_size=1, bias=True))

        self.dsn = nn.Sequential(
            nn.Conv2d(prev_final_channel, 512, kernel_size=3, stride=1, padding=1),
            Norm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        )

        self.encoder_struct = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1, bias=False),
            Norm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=True),
            Norm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=True),
        )

        # self.aspp_struct = _AtrousSpatialPyramidPoolingModule(256, 256,
        #                                                       output_stride=os)
        #
        # self.bot_aspp_struct = nn.Sequential(
        #     nn.Conv2d(1280, 256, kernel_size=1, bias=False),
        #     Norm2d(256),
        #     nn.ReLU(inplace=True))

        self.final1_struct = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            Norm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            Norm2d(256),
            nn.ReLU(inplace=True))

        self.final2_struct = nn.Sequential(
            nn.Conv2d(256, num_classes, kernel_size=1, bias=True))

        self.encoder_boundary = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0, bias=True),
            Norm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=True)
        )

        self.decoder_boundary = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=3, stride=2, padding=1, bias=True),
            Norm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=True)
        )

        self.final1_boundary = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            Norm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            Norm2d(256),
            nn.ReLU(inplace=True))

        self.final2_boundary = nn.Sequential(
            nn.Conv2d(256, num_classes, kernel_size=1, bias=True))

        self.aspp_sg = _AtrousSpatialPyramidPoolingModule(128, 256,
                                                          output_stride=os)

        self.bot_fine_sg = nn.Sequential(
            nn.Conv2d(channel_3rd, 48, kernel_size=1, bias=False),
            Norm2d(48),
            nn.ReLU(inplace=True))

        self.bot_aspp_sg = nn.Sequential(
            nn.Conv2d(1280, 256, kernel_size=1, bias=False),
            Norm2d(256),
            nn.ReLU(inplace=True))

        self.final1_sg = nn.Sequential(
            nn.Conv2d(304, 256, kernel_size=3, padding=1, bias=False),
            Norm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            Norm2d(256),
            nn.ReLU(inplace=True))

        self.final2_sg = nn.Sequential(
            nn.Conv2d(256, num_classes, kernel_size=1, bias=True))

        self.query_conv = nn.Conv2d(256, 128, kernel_size=1)
        self.key_conv = nn.Conv2d(256, 128, kernel_size=1)
        self.value_conv = nn.Conv2d(2048, 128, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

        self.up_conv = nn.Conv2d(2048, 128, kernel_size=1)

        self.query_conv.weight.data.normal_(0, 0.01)
        self.key_conv.weight.data.normal_(0, 0.01)
        self.value_conv.weight.data.normal_(0, 0.01)
        self.up_conv.weight.data.normal_(0, 0.01)

        initialize_weights(self.encoder_struct)

        initialize_weights(self.encoder_boundary)
        initialize_weights(self.decoder_boundary)

        initialize_weights(self.dsn)

        initialize_weights(self.aspp)
        initialize_weights(self.bot_aspp)
        initialize_weights(self.final1)
        initialize_weights(self.final2)

        initialize_weights(self.final1_struct)
        initialize_weights(self.final2_struct)

        initialize_weights(self.final1_boundary)
        initialize_weights(self.final2_boundary)

        initialize_weights(self.aspp_sg)
        initialize_weights(self.bot_aspp_sg)
        initialize_weights(self.bot_fine_sg)
        initialize_weights(self.final1_sg)
        initialize_weights(self.final2_sg)

        # Setting the flags
        self.eps = 1e-5
        self.whitening = False

        if trunk == 'resnet-101':
            self.three_input_layer = True
            in_channel_list = [64, 64, 128, 256, 512, 1024, 2048]   # 8128, 32640, 130816
            out_channel_list = [32, 32, 64, 128, 256,  512, 1024]
        elif trunk == 'resnet-18':
            self.three_input_layer = False
            in_channel_list = [0, 0, 64, 64, 128, 256, 512]   # 8128, 32640, 130816
            out_channel_list = [0, 0, 32, 32, 64,  128, 256]
        elif trunk == 'shufflenetv2':
            self.three_input_layer = False
            in_channel_list = [0, 0, 24, 116, 232, 464, 1024]
        elif trunk == 'mobilenetv2':
            self.three_input_layer = False
            in_channel_list = [0, 0, 16, 32, 64, 320, 1280]
        else: # ResNet-50
            self.three_input_layer = False
            in_channel_list = [0, 0, 64, 256, 512, 1024, 2048]   # 8128, 32640, 130816
            out_channel_list = [0, 0, 32, 128, 256,  512, 1024]

        self.cov_matrix_layer = []
        self.cov_type = []
        for i in range(len(self.args.wt_layer)):
            if self.args.wt_layer[i] > 0:
                self.whitening = True
                if self.args.wt_layer[i] == 1:
                    self.cov_matrix_layer.append(CovMatrix_IRW(dim=in_channel_list[i], relax_denom=self.args.relax_denom))
                    self.cov_type.append(self.args.wt_layer[i])
                elif self.args.wt_layer[i] == 2:
                    self.cov_matrix_layer.append(CovMatrix_ISW(dim=in_channel_list[i], relax_denom=self.args.relax_denom, clusters=self.args.clusters))
                    self.cov_type.append(self.args.wt_layer[i])

    def set_mask_matrix(self):
        for index in range(len(self.cov_matrix_layer)):
            self.cov_matrix_layer[index].set_mask_matrix()

    def reset_mask_matrix(self):
        for index in range(len(self.cov_matrix_layer)):
            self.cov_matrix_layer[index].reset_mask_matrix()

    def generated_coarse_layout(self, seg):
        pred_seg = seg.cpu().detach().numpy()
        L, C, W, H = pred_seg.shape
        predicted_label = np.zeros((L, W, H))
        predicted_prob = np.zeros((L, W, H))
        layout = np.zeros((L, W, H))
        for index in range(L):
            output = pred_seg[index].transpose(1, 2, 0)
            label, prob = np.argmax(output, axis=2), np.max(output, axis=2)
            predicted_label[index] = label.copy()
            predicted_prob[index] = prob.copy()
        thres = []
        for i in range(19):
            x = predicted_prob[predicted_label == i]
            if len(x) == 0:
                thres.append(0)
                continue
            x = np.sort(x)
            try:
                thres.append(x[np.int(np.round(len(x) * 0.5))])
            except Exception as e:
                print(len(x))
                print(e)

        thres = np.array(thres)
        thres[thres > 0.99] = 0.99
        for index in range(L):
            label = predicted_label[index]
            prob = predicted_prob[index]
            for i in range(19):
                label[(prob < thres[i]) * (label == i)] = 255
            output = np.asarray(label, dtype=np.uint8)
            layout[index] = output
        layout = torch.from_numpy(layout).unsqueeze(1).cuda().float()
        return layout

    def pam(self, x, struct, depth):
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(struct)
        proj_query = proj_query.view(m_batchsize, -1, width * height).permute(0, 2, 1)

        proj_key = self.key_conv(depth)
        proj_key = proj_key.view(m_batchsize, -1, width * height)

        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)

        proj_value = self.value_conv(x)
        proj_value = proj_value.view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, -1, height, width)
        x = self.up_conv(x)

        out = self.gamma * out + x
        return out

    def forward(self, x, gts=None, aux_gts=None, img_gt=None, boundary_gt=None, visualize=False, cal_covstat=False, apply_wtloss=True):
        w_arr = []

        if cal_covstat:
            x = torch.cat(x, dim=0)

        x_size = x.size()  # 800

        if self.trunk == 'mobilenetv2' or self.trunk == 'shufflenetv2':
            x_tuple = self.layer0([x, w_arr])
            x = x_tuple[0]
            w_arr = x_tuple[1]
        else:   # ResNet
            if self.three_input_layer:
                x = self.layer0[0](x)
                if self.args.wt_layer[0] == 1 or self.args.wt_layer[0] == 2:
                    x, w = self.layer0[1](x)
                    w_arr.append(w)
                else:
                    x = self.layer0[1](x)
                x = self.layer0[2](x)
                x = self.layer0[3](x)
                if self.args.wt_layer[1] == 1 or self.args.wt_layer[1] == 2:
                    x, w = self.layer0[4](x)
                    w_arr.append(w)
                else:
                    x = self.layer0[4](x)
                x = self.layer0[5](x)
                x = self.layer0[6](x)
                if self.args.wt_layer[2] == 1 or self.args.wt_layer[2] == 2:
                    x, w = self.layer0[7](x)
                    w_arr.append(w)
                else:
                    x = self.layer0[7](x)
                x = self.layer0[8](x)
                x = self.layer0[9](x)
            else:   # Single Input Layer
                x = self.layer0[0](x)
                if self.args.wt_layer[2] == 1 or self.args.wt_layer[2] == 2:
                    x, w = self.layer0[1](x)
                    w_arr.append(w)
                else:
                    x = self.layer0[1](x)
                x = self.layer0[2](x)
                x = self.layer0[3](x)

        x_tuple = self.layer1([x, w_arr])
        low_level = x_tuple[0]

        x_tuple = self.layer2(x_tuple)
        x_tuple = self.layer3(x_tuple)
        aux_out = x_tuple[0]
        x_tuple = self.layer4(x_tuple)
        x = x_tuple[0]
        w_arr = x_tuple[1]

        if cal_covstat:
            for index, f_map in enumerate(w_arr):
                # Instance Whitening
                B, C, H, W = f_map.shape  # i-th feature size (B X C X H X W)
                HW = H * W
                f_map = f_map.contiguous().view(B, C, -1)  # B X C X H X W > B X C X (H X W)
                eye, reverse_eye = self.cov_matrix_layer[index].get_eye_matrix()
                f_cor = torch.bmm(f_map, f_map.transpose(1, 2)).div(HW - 1) + (self.eps * eye)  # B X C X C / HW
                off_diag_elements = f_cor * reverse_eye
                self.cov_matrix_layer[index].set_variance_of_covariance(torch.var(off_diag_elements, dim=0))
            return 0

        boundary_input = low_level
        temp_x = x
        x = self.aspp(x)
        dec0_up = self.bot_aspp(x)
        dec0 = dec0_up
        dec1 = self.final1(dec0)
        dec2 = self.final2(dec1)
        main_out = Upsample(dec2, x_size[2:])

        coarse_layout = self.generated_coarse_layout(main_out)
        struct_feats = self.encoder_struct(coarse_layout)
        dec0_struct = struct_feats
        dec1_struct = self.final1_struct(dec0_struct)
        dec2_struct = self.final2_struct(dec1_struct)
        struct_seg = Upsample(dec2_struct, x_size[2:])

        boundary_enc_feats = self.encoder_boundary(boundary_input)
        boundary = torch.mean(boundary_enc_feats, dim=1, keepdim=True)
        boundary = Upsample(boundary, x_size[2:])
        boundary_dec_feats = self.decoder_boundary(boundary_enc_feats)
        dec0_boundary = boundary_dec_feats
        dec1_boundary = self.final1_boundary(dec0_boundary)
        dec2_boundary = self.final2_boundary(dec1_boundary)
        boundary_seg = Upsample(dec2_boundary, x_size[2:])

        final_feats = self.pam(temp_x, struct_feats, boundary_dec_feats)

        final_feats = self.aspp_sg(final_feats)
        dec0_up_sg = self.bot_aspp_sg(final_feats)
        dec0_fine_sg = self.bot_fine_sg(low_level)
        dec0_up_sg = Upsample(dec0_up_sg, low_level.size()[2:])
        dec0_sg = [dec0_fine_sg, dec0_up_sg]
        dec0_sg = torch.cat(dec0_sg, 1)
        dec1_sg = self.final1_sg(dec0_sg)
        dec2_sg = self.final2_sg(dec1_sg)
        main_out_sg = Upsample(dec2_sg, x_size[2:])

        if self.training:
            loss1 = self.criterion(main_out, gts)
            loss_final = self.criterion(main_out_sg, gts)
            loss_struct = self.criterion(struct_seg, coarse_layout.squeeze().long())
            loss_boundary = self.criterion(boundary_seg, gts)
            loss_l1_boundary = self.l1_loss(boundary, boundary_gt)

            if self.args.use_wtloss:
                wt_loss = torch.FloatTensor([0]).cuda()
                if apply_wtloss:
                    for index, f_map in enumerate(w_arr):
                        eye, mask_matrix, margin, num_remove_cov = self.cov_matrix_layer[index].get_mask_matrix()
                        loss = instance_whitening_loss(f_map, eye, mask_matrix, margin, num_remove_cov)
                        wt_loss = wt_loss + loss
                wt_loss = wt_loss / len(w_arr)

            aux_out = self.dsn(aux_out)
            if aux_gts.dim() == 1:
                aux_gts = gts
            aux_gts = aux_gts.unsqueeze(1).float()
            aux_gts = nn.functional.interpolate(aux_gts, size=aux_out.shape[2:], mode='nearest')
            aux_gts = aux_gts.squeeze(1).long()
            loss2 = self.criterion_aux(aux_out, aux_gts)

            return_loss = [loss1, loss2, loss_final, loss_struct, loss_boundary, loss_l1_boundary]
            if self.args.use_wtloss:
                return_loss.append(wt_loss)

            if self.args.use_wtloss and visualize:
                f_cor_arr = []
                for f_map in w_arr:
                    f_cor, _ = get_covariance_matrix(f_map)
                    f_cor_arr.append(f_cor)
                return_loss.append(f_cor_arr)
            return return_loss
        else:
            if visualize:
                f_cor_arr = []
                for f_map in w_arr:
                    f_cor, _ = get_covariance_matrix(f_map)
                    f_cor_arr.append(f_cor)
                return main_out, main_out_sg, f_cor_arr
            else:
                return main_out, main_out_sg


def get_final_layer(model):
    unfreeze_weights(model.final)
    return model.final


def DeepR18V3PlusD(args, num_classes, criterion, criterion_aux):
    """
    Resnet 18 Based Network
    """
    print("Model : DeepLabv3+, Backbone : ResNet-18")
    return DeepV3Plus(num_classes, trunk='resnet-18', criterion=criterion, criterion_aux=criterion_aux,
                    variant='D32', skip='m1', args=args)


def DeepR50V3PlusD_OS8(args, num_classes, criterion, criterion_aux):
    """
    Resnet 50 Based Network
    """
    print("Model : DeepLabv3+, Backbone : ResNet-50")
    return DeepV3Plus(num_classes, trunk='resnet-50', criterion=criterion, criterion_aux=criterion_aux,
                    variant='D', skip='m1', args=args)

def DeepR50V3PlusD(args, num_classes, criterion, criterion_aux, L1_loss):
    """
    Resnet 50 Based Network
    """
    print("Model : DeepLabv3+, Backbone : ResNet-50")
    return DeepV3Plus(num_classes, trunk='resnet-50', criterion=criterion, criterion_aux=criterion_aux, L1_loss=L1_loss,
                    variant='D16', skip='m1', args=args)

def DeepR101V3PlusD(args, num_classes, criterion, criterion_aux, L1_loss):
    """
    Resnet 101 Based Network
    """
    print("Model : DeepLabv3+, Backbone : ResNet-101")
    return DeepV3Plus(num_classes, trunk='resnet-101', criterion=criterion, criterion_aux=criterion_aux, L1_loss=L1_loss,
                    variant='D16', skip='m1', args=args)

def DeepR101V3PlusD_OS8(args, num_classes, criterion, criterion_aux, L1_loss):
    """
    Resnet 101 Based Network
    """
    print("Model : DeepLabv3+, Backbone : ResNet-101")
    return DeepV3Plus(num_classes, trunk='resnet-101', criterion=criterion, criterion_aux=criterion_aux, L1_loss=L1_loss,
                    variant='D', skip='m1', args=args)


def DeepR152V3PlusD_OS8(args, num_classes, criterion, criterion_aux):
    """
    Resnet 152 Based Network
    """
    print("Model : DeepLabv3+, Backbone : ResNet-152")
    return DeepV3Plus(num_classes, trunk='resnet-152', criterion=criterion, criterion_aux=criterion_aux,
                    variant='D', skip='m1', args=args)



def DeepResNext50V3PlusD(args, num_classes, criterion, criterion_aux):
    """
    Resnext 50 Based Network
    """
    print("Model : DeepLabv3+, Backbone : ResNext-50 32x4d")
    return DeepV3Plus(num_classes, trunk='resnext-50', criterion=criterion, criterion_aux=criterion_aux,
                    variant='D16', skip='m1', args=args)

def DeepResNext101V3PlusD(args, num_classes, criterion, criterion_aux):
    """
    Resnext 101 Based Network
    """
    print("Model : DeepLabv3+, Backbone : ResNext-101 32x8d")
    return DeepV3Plus(num_classes, trunk='resnext-101', criterion=criterion, criterion_aux=criterion_aux,
                    variant='D16', skip='m1', args=args)

def DeepWideResNet50V3PlusD(args, num_classes, criterion, criterion_aux):
    """
    Wide ResNet 50 Based Network
    """
    print("Model : DeepLabv3+, Backbone : wide_resnet-50")
    return DeepV3Plus(num_classes, trunk='wide_resnet-50', criterion=criterion, criterion_aux=criterion_aux,
                    variant='D16', skip='m1', args=args)

def DeepWideResNet50V3PlusD_OS8(args, num_classes, criterion, criterion_aux):
    """
    Wide ResNet 50 Based Network
    """
    print("Model : DeepLabv3+, Backbone : wide_resnet-50")
    return DeepV3Plus(num_classes, trunk='wide_resnet-50', criterion=criterion, criterion_aux=criterion_aux,
                    variant='D', skip='m1', args=args)

def DeepWideResNet101V3PlusD(args, num_classes, criterion, criterion_aux):
    """
    Wide ResNet 101 Based Network
    """
    print("Model : DeepLabv3+, Backbone : wide_resnet-101")
    return DeepV3Plus(num_classes, trunk='wide_resnet-101', criterion=criterion, criterion_aux=criterion_aux,
                    variant='D16', skip='m1', args=args)

def DeepWideResNet101V3PlusD_OS8(args, num_classes, criterion, criterion_aux):
    """
    Wide ResNet 101 Based Network
    """
    print("Model : DeepLabv3+, Backbone : wide_resnet-101")
    return DeepV3Plus(num_classes, trunk='wide_resnet-101', criterion=criterion, criterion_aux=criterion_aux,
                    variant='D', skip='m1', args=args)


def DeepResNext101V3PlusD_OS8(args, num_classes, criterion, criterion_aux):
    """
    ResNext 101 Based Network
    """
    print("Model : DeepLabv3+, Backbone : resnext-101")
    return DeepV3Plus(num_classes, trunk='resnext-101', criterion=criterion, criterion_aux=criterion_aux,
                    variant='D', skip='m1', args=args)

def DeepResNext101V3PlusD_OS4(args, num_classes, criterion, criterion_aux):
    """
    ResNext 101 Based Network
    """
    print("Model : DeepLabv3+, Backbone : resnext-101")
    return DeepV3Plus(num_classes, trunk='resnext-101', criterion=criterion, criterion_aux=criterion_aux,
                    variant='D4', skip='m1', args=args)

def DeepShuffleNetV3PlusD_OS32(args, num_classes, criterion, criterion_aux):
    """
    ShuffleNet Based Network
    """
    print("Model : DeepLabv3+, Backbone : shufflenetv2")
    return DeepV3Plus(num_classes, trunk='shufflenetv2', criterion=criterion, criterion_aux=criterion_aux,
                    variant='D32', skip='m1', args=args)


def DeepMNASNet05V3PlusD(args, num_classes, criterion, criterion_aux):
    """
    MNASNET Based Network
    """
    print("Model : DeepLabv3+, Backbone : mnas_0_5")
    return DeepV3Plus(num_classes, trunk='mnasnet_05', criterion=criterion, criterion_aux=criterion_aux,
                    variant='D16', skip='m1', args=args)

def DeepMNASNet10V3PlusD(args, num_classes, criterion, criterion_aux):
    """
    MNASNET Based Network
    """
    print("Model : DeepLabv3+, Backbone : mnas_1_0")
    return DeepV3Plus(num_classes, trunk='mnasnet_10', criterion=criterion, criterion_aux=criterion_aux,
                    variant='D16', skip='m1', args=args)


def DeepShuffleNetV3PlusD(args, num_classes, criterion, criterion_aux):
    """
    ShuffleNet Based Network
    """
    print("Model : DeepLabv3+, Backbone : shufflenetv2")
    return DeepV3Plus(num_classes, trunk='shufflenetv2', criterion=criterion, criterion_aux=criterion_aux,
                    variant='D16', skip='m1', args=args)

def DeepMobileNetV3PlusD(args, num_classes, criterion, criterion_aux):
    """
    ShuffleNet Based Network
    """
    print("Model : DeepLabv3+, Backbone : mobilenetv2")
    return DeepV3Plus(num_classes, trunk='mobilenetv2', criterion=criterion, criterion_aux=criterion_aux,
                    variant='D16', skip='m1', args=args)

def DeepMobileNetV3PlusD_OS8(args, num_classes, criterion, criterion_aux):
    """
    ShuffleNet Based Network
    """
    print("Model : DeepLabv3+, Backbone : mobilenetv2")
    return DeepV3Plus(num_classes, trunk='mobilenetv2', criterion=criterion, criterion_aux=criterion_aux,
                    variant='D', skip='m1', args=args)

def DeepShuffleNetV3PlusD_OS8(args, num_classes, criterion, criterion_aux):
    """
    ShuffleNet Based Network
    """
    print("Model : DeepLabv3+, Backbone : shufflenetv2")
    return DeepV3Plus(num_classes, trunk='shufflenetv2', criterion=criterion, criterion_aux=criterion_aux,
                    variant='D', skip='m1', args=args)
