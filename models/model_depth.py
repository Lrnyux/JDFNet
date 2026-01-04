import torch.utils.data
from basic_blocks import *
import numpy as np
from torch.autograd import Variable

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



class disparityregression(nn.Module):
    def __init__(self, maxdisp):
        super(disparityregression, self).__init__()
        self.maxdisp = maxdisp
        self.disp = torch.Tensor(np.reshape(np.array(range(maxdisp)),[1, maxdisp,1,1]))/ self.maxdisp

    def forward(self, x):
        out = torch.sum(x*self.disp.data,1, keepdim=True)
        return out

class DepthNet(nn.Module):

    def __init__(self, device, maxdisp=64):
        super(DepthNet, self).__init__()
        self.maxdisp = maxdisp
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
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1,bias=False))

        self.classif2 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1,bias=False))

        self.classif3 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1,bias=False))
        self.disparityregression = disparityregression(self.maxdisp)
        self.disparityregression.disp = self.disparityregression.disp.to(self.device)




    def _forward(self, left, right, neg=False):


        left_feats     = self.feature_extraction(left)
        right_feats  = self.feature_extraction(right)

        cost = Variable(torch.FloatTensor(left_feats.size()[0], left_feats.size()[1]*2, self.maxdisp//4,  right_feats.size()[2],  right_feats.size()[3]).zero_()).to(left.device)

        for i in range(self.maxdisp//4):
            if i > 0 :
             cost[:, :left_feats.size()[1], i, :,i:]   = left_feats[:,:,:,i:]
             cost[:, left_feats.size()[1]:, i, :,i:] = right_feats[:,:,:,:-i]
            else:
             cost[:, :left_feats.size()[1], i, :,:]   = left_feats
             cost[:, left_feats.size()[1]:, i, :,:]   = right_feats

        cost = cost.contiguous()
        cost0 = self.dres0(cost)
        cost0 = self.dres1(cost0) + cost0

        out1, pre1, post1 = self.dres2(cost0, None, None)
        out1 = out1+cost0

        out2, pre2, post2 = self.dres3(out1, pre1, post1)
        out2 = out2+cost0

        out3, pre3, post3 = self.dres4(out2, pre1, post2)
        out3 = out3+cost0


        cost1 = self.classif1(out1)
        cost2 = self.classif2(out2) + cost1
        cost3 = self.classif3(out3) + cost2

        cost1 = F.interpolate(cost1, [self.maxdisp,left.size()[2],left.size()[3]], mode='trilinear')
        cost2 = F.interpolate(cost2, [self.maxdisp,left.size()[2],left.size()[3]], mode='trilinear')

        cost1 = torch.squeeze(cost1,1)
        pred1 = F.softmax(cost1,dim=1)
        pred1 = self.disparityregression(pred1)

        cost2 = torch.squeeze(cost2,1)
        pred2 = F.softmax(cost2,dim=1)
        pred2 = self.disparityregression(pred2)

        if neg:
            pred1 = -1 *  pred1
            pred2 = -1 *  pred2

        cost3 = F.interpolate(cost3, [self.maxdisp,left.size()[2],left.size()[3]], mode='trilinear')
        cost3 = torch.squeeze(cost3,1)
        pred3 = F.softmax(cost3,dim=1)
        pred3 = self.disparityregression(pred3)

        if neg:
            pred3 = -1 * pred3

        if self.training:
            return pred1, pred2, pred3
        else:
            return pred1


    def forward(self,left,right):
        if self.training:
            pred_l1,pred_l2,pred_l3=self._forward(left,right,neg=False)
            dips = [pred_l1, pred_l2, pred_l3]
            return dips
        else:
            pred_l_r = self._forward(left, right, neg=False)
            dips = pred_l_r
            return dips





if __name__ == '__main__':
    device = torch.device('cuda:0')
    model = DepthNet(device=device,maxdisp=64).to(device)
    model.eval()
    input_L = torch.ones((1, 3, 320, 256)).to(device)
    input_R = torch.ones((1, 3, 320, 256)).to(device)
    output = model(input_L, input_R)
