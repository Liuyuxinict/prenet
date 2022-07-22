import torch.nn as nn
import torch
import torch.nn.functional as F
from self_attention import self_attention
from layer_self_attention import layer_self_attention
from dropblock import DropBlock2D
import numpy as np

class PMG(nn.Module):
    def __init__(self, model, feature_size, classes_num):
        super(PMG, self).__init__()

        self.features = model

        self.num_ftrs = 2048 * 1 * 1
        self.elu = nn.ELU(inplace=True)

        self.dk = 0.5
        self.dq = 0.5
        self.dv = 0.5
        self.Nh = 8


        self.classifier_concat = nn.Sequential(
            nn.BatchNorm1d(1024 * 5),
            nn.Linear(1024 * 5, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Linear(feature_size, classes_num),
        )

        self.conv_block0 = nn.Sequential(
            BasicConv(self.num_ftrs // 8, feature_size, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(feature_size, self.num_ftrs // 2, kernel_size=3, stride=1, padding=1, relu=True)
        )
        self.classifier0 = nn.Sequential(
            nn.BatchNorm1d(self.num_ftrs // 2),
            nn.Linear(self.num_ftrs // 2, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Linear(feature_size, classes_num),
        )

        self.conv_block1 = nn.Sequential(
            BasicConv(self.num_ftrs//4, feature_size, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(feature_size, self.num_ftrs//2, kernel_size=3, stride=1, padding=1, relu=True)
        )
        self.classifier1 = nn.Sequential(
            nn.BatchNorm1d(self.num_ftrs//2),
            nn.Linear(self.num_ftrs//2, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Linear(feature_size, classes_num),
        )

        self.conv_block2 = nn.Sequential(
            BasicConv(self.num_ftrs//2, feature_size, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(feature_size, self.num_ftrs//2, kernel_size=3, stride=1, padding=1, relu=True)
        )
        self.classifier2 = nn.Sequential(
            nn.BatchNorm1d(self.num_ftrs//2),
            nn.Linear(self.num_ftrs//2, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Linear(feature_size, classes_num),
        )

        self.conv_block3 = nn.Sequential(
            BasicConv(self.num_ftrs, feature_size, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(feature_size, self.num_ftrs//2, kernel_size=3, stride=1, padding=1, relu=True)
        )
        self.classifier3 = nn.Sequential(
            nn.BatchNorm1d(self.num_ftrs//2),
            nn.Linear(self.num_ftrs//2, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Linear(feature_size, classes_num),
        )

        self.Avgmax = nn.AdaptiveMaxPool2d(output_size=(1,1))

        self.attn1_1 = self_attention(self.num_ftrs // 2,self.num_ftrs // 2, self.dk, self.dq, self.dv, self.Nh)
        self.attn2_2 = self_attention(self.num_ftrs // 2,self.num_ftrs // 2, self.dk, self.dq, self.dv, self.Nh)
        self.attn3_3 = self_attention(self.num_ftrs // 2,self.num_ftrs // 2, self.dk, self.dq, self.dv, self.Nh)

        '''
        self.attn1_2 = layer_self_attention(self.num_ftrs // 2,self.num_ftrs // 2, self.dk, self.dq, self.dv, self.Nh)
        self.attn1_3 = layer_self_attention(self.num_ftrs // 2,self.num_ftrs // 2, self.dk, self.dq, self.dv, self.Nh)
        self.attn2_3 = layer_self_attention(self.num_ftrs // 2,self.num_ftrs // 2, self.dk, self.dq, self.dv, self.Nh)

        self.attn2_1 = layer_self_attention(self.num_ftrs // 2, self.num_ftrs // 2, self.dk, self.dq, self.dv, self.Nh)
        self.attn3_1 = layer_self_attention(self.num_ftrs // 2, self.num_ftrs // 2, self.dk, self.dq, self.dv, self.Nh)
        self.attn3_2 = layer_self_attention(self.num_ftrs // 2, self.num_ftrs // 2, self.dk, self.dq, self.dv, self.Nh)
        '''

        self.sconv1 = nn.Conv2d((self.num_ftrs // 2), self.num_ftrs // 2, kernel_size= 3, padding= 1)
        self.sconv2 = nn.Conv2d((self.num_ftrs // 2), self.num_ftrs // 2, kernel_size= 3, padding= 1)
        self.sconv3 = nn.Conv2d((self.num_ftrs // 2), self.num_ftrs // 2, kernel_size= 3, padding= 1)
        self.drop_block = DropBlock2D(block_size=3, drop_prob=0.5)

    def forward(self, x, label):
        xf1, xf2, xf3, xf4, xf5, xn = self.features(x)
        batch_size, _, _, _ = x.shape

        #get feature pyramid
        xl1 = self.conv_block1(xf3)
        xl2 = self.conv_block2(xf4)
        xl3 = self.conv_block3(xf5)

        xk1 = self.Avgmax(xl1)
        xk1 = xk1.view(xk1.size(0), -1)
        xc1 = self.classifier1(xk1)

        xk2 = self.Avgmax(xl2)
        xk2 = xk2.view(xk2.size(0), -1)
        xc2 = self.classifier2(xk2)

        xk3 = self.Avgmax(xl3)
        xk3 = xk3.view(xk3.size(0), -1)
        xc3 = self.classifier3(xk3)


        if label:
            # xs1_2 means that using x2 to strength x1
            #(batch, 1024, 56, 56)
            xs1 = self.attn1_1(xl1)
            #xs1_2 = self.attn1_2(xl1, xl2)
            #xs1_3 = self.attn1_3(xl1, xl3)
            # (batch, 1024, 28, 28)
            xs2 = self.attn1_1(xl2)
            #xs2_3 = self.attn2_3(xl2, xl3)
            #xs2_1 = self.attn2_1(xl2, xl1)
            # (batch, 1024, 14, 14)
            xs3 = self.attn1_1(xl3)
            #xs3_1 = self.attn2_1(xl3, xl1)
            #xs3_2 = self.attn2_1(xl3, xl2)

            #xr1 = self.drop_block(self.sconv1(torch.cat([xs1,xs1_2,xs1_3], dim=1)))
            #xr2 = self.drop_block(self.sconv2(torch.cat([xs2,xs2_3,xs2_1], dim=1)))
            #xr3 = self.drop_block(self.sconv3(torch.cat([xs3,xs3_1,xs3_2], dim=1)))
            xr1 = self.drop_block(self.sconv1(xs1))
            xr2 = self.drop_block(self.sconv2(xs2))
            xr3 = self.drop_block(self.sconv3(xs3))

            xm1 = self.Avgmax(xr1)
            xm1 = xm1.view(xm1.size(0), -1)
            #print(np.argmax(F.softmax(xm1, dim=1).cpu().detach().numpy(),axis=1))
            #input()

            xm2 = self.Avgmax(xr2)
            xm2 = xm2.view(xm2.size(0), -1)
            #print(np.argmax(F.softmax(xm2, dim=1).cpu().detach().numpy(),axis=1))
            #input()

            xm3 = self.Avgmax(xr3)
            xm3 = xm3.view(xm3.size(0), -1)
            #print(np.argmax(F.softmax(xm3, dim=1).cpu().detach().numpy(),axis=1))
            #input()

            x_concat = torch.cat((xm1, xm2, xm3, xn), dim=1)
            x_concat = self.classifier_concat(x_concat)
        else:
            x_concat = torch.cat((xk1, xk2, xk3, xn), dim=1)
            x_concat = self.classifier_concat(x_concat)

        #get origal feature vector


        return xk1, xk2, xk3, x_concat, xc1, xc2, xc3
    
    
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5,
                                 momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
