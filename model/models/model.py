import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange, repeat
from model.models.fc4.ModelFC4 import ModelFC4


device = 'cuda' if torch.cuda.is_available() else 'cpu'


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU, drop=0.3):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=True)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=True)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class ConvMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU, drop=0.3):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, 1, 0, bias=True)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, 1, 0, bias=True)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class DepthWiseConv(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(DepthWiseConv, self).__init__()
        self.depth_conv = nn.Conv2d(in_channels=in_channel,
                                    out_channels=in_channel,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    groups=in_channel)
        self.point_conv = nn.Conv2d(in_channels=in_channel,
                                    out_channels=out_channel,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1)
    def forward(self,input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out



class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels),
            nn.ReLU6()
        )


class ConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False):
        super(ConvBN, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels)
        )


class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, bias=False):
        super(Conv, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2)
        )


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)


class CALayer(nn.Module):
    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return x * y



class HM(nn.Module):
    def __init__(self, channel):
        super(HM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ka = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )
        self.td = nn.Sequential(
            nn.Conv2d(channel, channel, 3, padding=1, bias=True),
            nn.Conv2d(channel, channel // 8, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, channel, 3, padding=1, bias=True),
            nn.Sigmoid()
        )

        self.mix = DynamicMixBlock(m=2, merge=True)

    def forward(self, x, bias):
        a = self.avg_pool(x)
        a = self.ka(a)
        a = self.mix(a, bias)
        t = self.td(x)
        j = torch.mul((1 - t), a) + torch.mul(t, x)
        return j
    


class Block(nn.Module):
    def __init__(self, conv, dim, kernel_size):
        super(Block, self).__init__()
        self.conv = conv(dim, dim, kernel_size, bias=True)
        self.calayer = CALayer(dim)
        self.pdu = HM(dim)

    def forward(self, x):
        x, bias = x
        res = self.conv(x)
        res = self.calayer(res)
        res = self.pdu(res, bias)
        res += x
        return (res, bias)
    

class HMGroup(nn.Module):
    def __init__(self, conv=default_conv, dim=64, kernel_size=3, blocks=19):
        super(HMGroup, self).__init__()
        modules = [Block(conv, dim, kernel_size) for _ in range(blocks)]
        self.gp = nn.Sequential(*modules)
        self.conv = conv(dim, dim, kernel_size)

    def forward(self, x, bias):
        res = self.gp((x, bias))
        res, _ = res
        res = self.conv(res)
        res += x
        return res
    



class DynamicLearnableTensor(nn.Module):
    def __init__(self, channels):
        super(DynamicLearnableTensor, self).__init__()
        self.initial_param = nn.Parameter(torch.randn(1, channels, 1, 1))

    def forward(self, x):
        batch_size, _, height, width = x.shape
        learnable_tensor = F.interpolate(self.initial_param, size=(height, width), mode='nearest')
        learnable_tensor = learnable_tensor.repeat(batch_size, 1, 1, 1)
        return learnable_tensor


class DynamicMixBlock(nn.Module):
    def __init__(self, m=-0.80, merge=False):
        super(DynamicMixBlock, self).__init__()
        self.merge = merge

        w = torch.nn.Parameter(torch.FloatTensor([m]), requires_grad=True)
        w = torch.nn.Parameter(w, requires_grad=True)
        self.w = w
        self.mix_block = nn.Sigmoid()

    def forward(self, x, y):
        mix_factor = self.mix_block(self.w)
        x = x * mix_factor.expand_as(x) 
        y = y * (1 - mix_factor.expand_as(y))
        if self.merge:
            return x + y
        return x, y
    
class ChannelAttentionModul(nn.Module):  
    def __init__(self, in_channel, r=0.5):  
        super(ChannelAttentionModul, self).__init__()
        self.MaxPool = nn.AdaptiveMaxPool2d(1)

        self.fc_MaxPool = nn.Sequential(
            nn.Linear(in_channel, int(in_channel * r)),  
            nn.ReLU(),
            nn.Linear(int(in_channel * r), in_channel),
            nn.Sigmoid(),
        )

        self.AvgPool = nn.AdaptiveAvgPool2d(1)

        self.fc_AvgPool = nn.Sequential(
            nn.Linear(in_channel, int(in_channel * r)), 
            nn.ReLU(),
            nn.Linear(int(in_channel * r), in_channel),
            nn.Sigmoid(),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_branch = self.MaxPool(x)
        max_in = max_branch.view(max_branch.size(0), -1)
        max_weight = self.fc_MaxPool(max_in)

        avg_branch = self.AvgPool(x)
        avg_in = avg_branch.view(avg_branch.size(0), -1)
        avg_weight = self.fc_AvgPool(avg_in)

        weight = max_weight + avg_weight
        weight = self.sigmoid(weight)

        h, w = weight.shape
        Mc = torch.reshape(weight, (h, w, 1, 1))

        x = Mc * x

        return x

class SpatialAttentionModul(nn.Module): 
    def __init__(self, in_channel):
        super(SpatialAttentionModul, self).__init__()
        self.conv = nn.Conv2d(2, 1, 7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        MaxPool = torch.max(x, dim=1).values 
        AvgPool = torch.mean(x, dim=1)

        MaxPool = torch.unsqueeze(MaxPool, dim=1)
        AvgPool = torch.unsqueeze(AvgPool, dim=1)

        x_cat = torch.cat((MaxPool, AvgPool), dim=1) 

        x_out = self.conv(x_cat)
        Ms = self.sigmoid(x_out)

        x = Ms * x

        return x

class CBAM(nn.Module):
    def __init__(self, in_channel):
        super(CBAM, self).__init__()
        self.Cam = ChannelAttentionModul(in_channel=in_channel)  
        self.Sam = SpatialAttentionModul(in_channel=in_channel) 

    def forward(self, x):
        x = self.Cam(x)
        x = self.Sam(x)
        return x


    
    

class Network(nn.Module):
    def __init__(self,
                 dim=24
                 ):
        super().__init__()

        self.dim = dim

        self.fc4 = ModelFC4()
        self.path_to_pretrained = 'model/models/fc4/weights'
        self.fc4.load(self.path_to_pretrained)
        self.fc4.evaluation_mode()

        
        self.pre_process1 = default_conv(3, self.dim, kernel_size=3)
        self.conv1 = nn.Sequential(nn.Conv2d(self.dim, self.dim*2, kernel_size=3, stride=2, padding=1),
                                   nn.ReLU(True))
        self.conv2 = nn.Sequential(nn.Conv2d(self.dim*2, self.dim*4, kernel_size=3, stride=2, padding=1),
                                   nn.ReLU(True))

        self.blp_conv1 = default_conv(3, self.dim, kernel_size=3)
        self.blp_conv2 = nn.Sequential(nn.Conv2d(self.dim, self.dim*2, kernel_size=3, stride=2, padding=1),
                                   nn.ReLU(True))
        self.blp_conv3 = nn.Sequential(nn.Conv2d(self.dim*2, self.dim*4, kernel_size=3, stride=2, padding=1),
                                   nn.ReLU(True))
        

        self.att1 = CBAM(self.dim)
        self.att2 = CBAM(self.dim*2)
        self.att3 = CBAM(self.dim*4)

        self.hm1 = HMGroup(dim=self.dim, blocks=20)
        self.hm2 = HMGroup(dim=self.dim*2, blocks=20)
        self.hm3 = HMGroup(dim=self.dim*4, blocks=20)

        self.learnable_color = nn.Parameter(torch.zeros(1, 3, 1, 1), requires_grad=True)
        
        self.mix1 = DynamicMixBlock(m=0.6, merge=True)
        self.mix2 = DynamicMixBlock(m=0.4, merge=True)

        self.up1 = nn.Sequential(nn.ConvTranspose2d(self.dim*4, self.dim*2, kernel_size=3, stride=2, padding=1, output_padding=1),
                                 nn.ReLU(True))
        self.up2 = nn.Sequential(nn.ConvTranspose2d(self.dim*2, self.dim, kernel_size=3, stride=2, padding=1, output_padding=1),
                                 nn.ReLU(True))
        
        self.head = Conv(self.dim, 3, kernel_size=3)
        


    def forward(self, x):      

        color_bias = self.fc4.predict(x)
        
        color_bias = color_bias.unsqueeze(2).unsqueeze(3) + self.learnable_color
        color_bias = color_bias.expand_as(x)

        feat1 = self.pre_process1(x)
        feat1 = self.att1(feat1)
        color_feat = self.blp_conv1(color_bias)

        feat2 = self.conv1(feat1)
        feat2 = self.att2(feat2)
        color_down1 = self.blp_conv2(color_feat)

        feat3 = self.conv2(feat2)
        feat3 = self.att3(feat3)
        color_down2 = self.blp_conv3(color_down1)

        j1 = self.hm1(feat1, color_feat)
        j2 = self.hm2(feat2, color_down1)
        j3 = self.hm3(feat3, color_down2)

        j3 = self.up1(j3)
        j2 = self.mix1(j3, j2)
        j2 = self.up2(j2)
        j1 = self.mix2(j2, j1)


        out = self.head(j1)


        return out
    