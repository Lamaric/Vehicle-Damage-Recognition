from .xresnet import *
from torchvision.models import ResNet,resnet18,resnet34,resnet50,resnet101,resnet152
from torchvision.models import SqueezeNet,squeezenet1_0,squeezenet1_1
from torchvision.models import densenet121,densenet169,densenet201,densenet161
from torchvision.models import vgg16_bn,vgg19_bn,alexnet
from .darknet import *
from .unet import *
from .wrn import *
from .xception import *

from .resnext_50_32x4d import resnext_50_32x4d
from .senet import se_resnet50,se_resnext50_32x4d,se_resnet101,se_resnext101_32x4d
import os
import torch

def children(m): return m if isinstance(m, (list, tuple)) else list(m.children())
def save_model(m, p): torch.save(m.state_dict(), p)
def load_model(m, p):
    sd = torch.load(p, map_location=lambda storage, loc: storage)
    names = set(m.state_dict().keys())
    for n in list(sd.keys()): # list "detatches" the iterator
        if n not in names and n+'_raw' in names:
            if n+'_raw' not in sd: sd[n+'_raw'] = sd[n]
            del sd[n]
    m.load_state_dict(sd)

def load_pre(pre, f, fn):
    m = f()
    path = os.path.dirname(__file__)
    if pre: load_model(m, f'{path}/weights/{fn}.pth')
    return m

def resnext50(pre):     return load_pre(pre, resnext_50_32x4d, 'resnext_50_32x4d')
def Se_resnet50(pre):   return load_pre(pre, se_resnet50, 'se_resnet50')