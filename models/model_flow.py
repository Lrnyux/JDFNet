import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import math
from basic_blocks import *
import time
import numpy as np

class hourglass(nn.Module):
    def __init__(self, inplanes):
        super(hourglass, self).__init__()

        self.conv1 = nn.Sequential(convbn_3d(inplanes, inplanes * 2, kernel_size=3, stride=2, pad=1),
                                   nn.ReLU(inplace=True))

        self.conv2 = convbn_3d(inplanes * 2, inplanes * 2, kernel_size=3, stride=1, pad=1)

        self.conv3 = nn.Sequential(convbn_3d(inplanes * 2, inplanes * 2, kernel_size=3, stride=2, pad=1),
                                   nn.ReLU(inplace=True))

        self.conv4 = nn.Sequential(convbn_3d(inplanes * 2, inplanes * 2, kernel_size=3, stride=1, pad=1),
                                   nn.ReLU(inplace=True))

        self.conv5 = nn.Sequential(
            nn.ConvTranspose3d(inplanes * 2, inplanes * 2, kernel_size=3, padding=1, output_padding=1, stride=2,
                               bias=False),
            nn.BatchNorm3d(inplanes * 2))

        self.conv6 = nn.Sequential(
            nn.ConvTranspose3d(inplanes * 2, inplanes, kernel_size=3, padding=1, output_padding=1, stride=2,
                               bias=False),
            nn.BatchNorm3d(inplanes))

    def forward(self, x, presqu, postsqu):

        out = self.conv1(x)
        pre = self.conv2(out)
        if postsqu is not None:
            pre = F.relu(pre + postsqu, inplace=True)
        else:
            pre = F.relu(pre, inplace=True)

        out = self.conv3(pre)
        out = self.conv4(out)

        if presqu is not None:
            post = F.relu(self.conv5(out) + presqu, inplace=True)
        else:
            post = F.relu(self.conv5(out) + pre, inplace=True)

        out = self.conv6(post)

        return out, pre, post


class Flowregression(nn.Module):
    def __init__(self, maxrange):
        super(Flowregression, self).__init__()
        self.flow_u = torch.Tensor(np.reshape(np.array(range(-maxrange//2, maxrange//2)), [1, 1, maxrange, 1, 1])) / (maxrange//2)
        self.flow_v = torch.Tensor(np.reshape(np.array(range(-maxrange//2, maxrange//2)), [1, 1, maxrange, 1, 1])) / (maxrange//2)
        self.flow = torch.concat((self.flow_u,self.flow_v),dim=1)

    def forward(self, x):
        out = torch.sum(x * self.flow.data, 2, keepdim=False)
        return out




class FlowNet(nn.Module):

    def __init__(self, device, maxrange=64):
        super(FlowNet, self).__init__()
        self.maxrange = maxrange
        self.device = device
        self.feature_extraction = feature_extraction()

        self.dres0 = nn.Sequential(convbn_3d(64, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True))

        self.dres1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1))

        self.dres2 = hourglass(32)

        self.dres3 = hourglass(32)

        self.dres4 = hourglass(32)

        self.classif1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 2, kernel_size=3, padding=1, stride=1, bias=False))

        self.classif2 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 2, kernel_size=3, padding=1, stride=1, bias=False))

        self.classif3 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 2, kernel_size=3, padding=1, stride=1, bias=False))
        self.flowregression = Flowregression(self.maxrange)
        self.flowregression.flow = self.flowregression.flow.to(self.device)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def _forward(self, left, right ):

        left_feats = self.feature_extraction(left)
        right_feats = self.feature_extraction(right)

        cost = Variable(
            torch.FloatTensor(left_feats.size()[0], left_feats.size()[1] * 2, self.maxrange // 4, left_feats.size()[2],
                              left_feats.size()[3]).zero_()).to(left.device)

        for i in range(self.maxrange // 4):
            if i > 0:
                cost[:, :left_feats.size()[1], i, :, i:] = left_feats[:, :, :, i:]
                cost[:, left_feats.size()[1]:, i, :, i:] = right_feats[:, :, :, :-i]
            else:
                cost[:, :left_feats.size()[1], i, :, :] = left_feats
                cost[:, left_feats.size()[1]:, i, :, :] = right_feats

        cost = cost.contiguous()
        cost0 = self.dres0(cost)
        cost0 = self.dres1(cost0) + cost0


        out1, pre1, post1 = self.dres2(cost0, None, None)
        out1 = out1 + cost0

        out2, pre2, post2 = self.dres3(out1, pre1, post1)
        out2 = out2 + cost0

        out3, pre3, post3 = self.dres4(out2, pre1, post2)
        out3 = out3 + cost0

        cost1 = self.classif1(out1)
        cost2 = self.classif2(out2) + cost1
        cost3 = self.classif3(out3) + cost2

        cost1 = F.interpolate(cost1, [self.maxrange, left.size()[2], left.size()[3]], mode='trilinear')
        cost2 = F.interpolate(cost2, [self.maxrange, left.size()[2], left.size()[3]], mode='trilinear')

        pred1 = F.softmax(cost1, dim=2)
        pred1 = self.flowregression(pred1)



        pred2 = F.softmax(cost2, dim=2)
        pred2 = self.flowregression(pred2)

        cost3 = F.interpolate(cost3, [self.maxrange, left.size()[2], left.size()[3]], mode='trilinear')
        pred3 = F.softmax(cost3, dim=2)
        pred3 = self.flowregression(pred3)


        if self.training:
            return pred1, pred2, pred3
        else:
            return pred3

    def forward(self, left, right):
        if self.training:
            pred_l1, pred_l2, pred_l3 = self._forward(left, right)
            flows = [pred_l1, pred_l2, pred_l3]
            return flows
        else:
            pred_l_r = self._forward(left, right )
            flows = pred_l_r
            return flows


if __name__ == '__main__':
    device = torch.device('cuda:0')
    model = FlowNet(device=device, maxrange=64).to(device)

    model.eval()

    input_L = torch.ones((1, 3, 256, 320)).to(device)
    input_R = torch.ones((1, 3, 256, 320)).to(device)

    output = model(input_L, input_R)



