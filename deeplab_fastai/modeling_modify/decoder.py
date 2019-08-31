import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from fastai.vision import *

class ConvBn2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dilation=1, stride=1, groups=1, is_bn=True):
        super(ConvBn2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups, bias=False)
        #self.bn = SynchronizedBatchNorm2d(out_channels)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, z):
        x = self.conv(z)
        x = self.bn(x)
        return x

class sSE(nn.Module):
    def __init__(self, out_channels):
        super(sSE, self).__init__()
        self.conv = ConvBn2d(in_channels=out_channels,out_channels=1,kernel_size=1,padding=0)
    def forward(self,x):
        x=self.conv(x)
        #print('spatial',x.size())
        x=torch.sigmoid(x)
        return x

class cSE(nn.Module):
    def __init__(self, out_channels):
        super(cSE, self).__init__()
        self.conv1 = ConvBn2d(in_channels=out_channels,out_channels=int(out_channels/2),kernel_size=1,padding=0)
        self.conv2 = ConvBn2d(in_channels=int(out_channels/2),out_channels=out_channels,kernel_size=1,padding=0)
    def forward(self,x):
        x=nn.AvgPool2d(x.size()[2:])(x)
        #print('channel',x.size())
        x=self.conv1(x)
        x=F.relu(x)
        x=self.conv2(x)
        x=torch.sigmoid(x)
        return x



class Se_Decoder(nn.Module):
    def __init__(self, in_channels, out_channels,kernel_size=3,padding=1):#channels,
        super(Se_Decoder, self).__init__()
        self.conv1 = ConvBn2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        #self.conv2 = ConvBn2d(channels, out_channels, kernel_size=3, padding=1)
        self.spatial_gate = sSE(out_channels)
        self.channel_gate = cSE(out_channels)
       #kernel_size=3, stride=1, padding=1
    def forward(self, x, e=None):
        #print('inp',x.size())
        if x.size()[3]!=128:
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True) #线性插值 64 -- 128 #nn.functional.interpolate
        #print('x_inter',x.size())
        #if e is not None: print('e',e.size())
        if e is not None:
            x = torch.cat([x,e],1)

        x = F.relu(self.conv1(x),inplace=True)
        #x = F.relu(self.conv2(x),inplace=True)
        #print('x_new',x.size())
        g1 = self.spatial_gate(x)
        #print('g1',g1.size())
        g2 = self.channel_gate(x)
        #print('g2',g2.size())
        x = g1*x + g2*x

        return x
    
class Decoder(nn.Module):
    def __init__(self, num_classes, backbone,BatchNorm):
        super(Decoder,self).__init__()
        if backbone == 'resnet' or backbone == 'drn':
            low_level_inplanes = 256
            
        else:
            raise NotImplementedError
        BatchNorm = nn.BatchNorm2d
        self.conv1 = Se_Decoder(low_level_inplanes,64,kernel_size=1,padding=0)
        #self.bn1 = BatchNorm(64)
        #self.relu = nn.ReLU()
        self.last_conv = nn.Sequential(Se_Decoder(320,256),
                                        #BatchNorm(256),
                                        nn.Dropout(0.5),
                                        Se_Decoder(256,256),
                                        #BatchNorm(256),
                                        nn.Dropout(0.3),)
                                        #nn.Conv2d(256, num_classes, kernel_size=1, stride=1))
        self.ps = PixelShuffle_ICNR(256)
        self.layers=[]
        self.layers.append(MergeLayer(dense=False))
        self.layers.append(res_block(320,))#ks=(3, 3), stride=1, padding=1, bias=False))
        self.layers += [conv_layer(320, num_classes, ks=1, use_activ=False)]
        self.last_cross = SequentialEx(*self.layers)
        self._init_weight()
        
        self.ps2 = PixelShuffle_ICNR(32)

    def forward(self, x, low_level_feat):
        #print("low_level_feat",low_level_feat.size())
        low_level_feat = self.conv1(low_level_feat)

        x = self.ps(x)# ([12, 256, 32, 32])
        #print("x_ps",x.size())#([12, 256, 64, 64])
        x = F.interpolate(x, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)
        #print("x_interpolate",x.size())#([12, 256, 128, 128])
        x = torch.cat((x, low_level_feat), dim=1) #([12, 320, 128, 128])
        #print("x_cat",x.size())
        
        x = self.last_cross(x)  #last_cross torch.Size([8, 32, 128, 128])
        #print("last_cross",x.size())
        x = self.ps2(x)
        #print("ps2",x.size())
        return x
    
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            #elif isinstance(m, SynchronizedBatchNorm2d):
            #    m.weight.data.fill_(1)
             #   m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

def build_decoder(num_classes, backbone, BatchNorm):
    return Decoder(num_classes, backbone, BatchNorm)