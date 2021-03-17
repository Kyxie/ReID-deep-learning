# This code is for metric learning, ResNet
# Engineer: Kunyang Xie
# Last Update: 17/3/2021

import torchvision
from torch import nn
from torch.nn import functional as F

class ResNet50(nn.Module):
    def __init__(self, num_classes, loss={'softmax, metric'}, **kwargs):
        super(ResNet50, self).__init__()
        resnet50 = torchvision.models.resnet50(pretrained=True)
        self.loss = loss
        self.base = nn.Sequential(*list(resnet50.children())[:-2])
        if not self.loss == {'metric'}:
            self.classifier = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.base(x)
        x = F.avg_pool2d(x, x.size()[2:])
        f = x.view(x.size(0), -1)   # f = feature
        y = self.classifier(f)

        # # Normalization
        # x = 1. * x / (torch.norm(x, 2, dim=-1, keepdim=True).expand_as(x) + 1e-12)

        if not self.training:
            return f

        if self.loss == {'metric'}:
            return f

        elif self.loss == {'softmax'}:
            return y

        elif self.loss == {'softmax', 'metric'}:
            return y, f

        else:
            print('No such loss')