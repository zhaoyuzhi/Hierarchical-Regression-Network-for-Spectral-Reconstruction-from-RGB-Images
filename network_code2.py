import torch
import torch.nn as nn
import torch.nn.functional as F

from network_module import *
import PixelUnShuffle

# ----------------------------------------
#         Initialize the networks
# ----------------------------------------
def weights_init(net, init_type = 'normal', init_gain = 0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal
    In our paper, we choose the default setting: zero mean Gaussian distribution with a standard deviation of 0.02
    """
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain = init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a = 0, mode = 'fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain = init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)

    # apply the initialization function <init_func>
    print('initialize network with %s type' % init_type)
    net.apply(init_func)

# ----------------------------------------
#                 Generator
# ----------------------------------------
class SGN(nn.Module):
    def __init__(self, opt):
        super(SGN, self).__init__()
        # Top subnetwork, K = 3
        self.top1 = Conv2dLayer(opt.in_channels * (4 ** 3), opt.start_channels * (2 ** 3), 3, 1, 1, pad_type = opt.pad, activation = opt.activ, norm = opt.norm)
        self.top21 = GlobalBlock(opt.start_channels * (2 ** 3), 3, 1, 1, pad_type = opt.pad, activation = opt.activ, norm = opt.norm, sn = False, reduction = 4)
        self.top22 = ResidualDenseBlock_5C(opt.start_channels * (2 ** 3), opt.start_channels * (2 ** 2), 3, 1, 1, pad_type = opt.pad, activation = opt.activ, norm = opt.norm)
        self.top3 = Conv2dLayer(opt.start_channels * (2 ** 3), opt.start_channels * (2 ** 3), 1, 1, 0, pad_type = opt.pad, activation = opt.activ, norm = opt.norm)
        # Middle subnetwork, K = 2
        self.mid1 = Conv2dLayer(opt.in_channels * (4 ** 2), opt.start_channels * (2 ** 2), 3, 1, 1, pad_type = opt.pad, activation = opt.activ, norm = opt.norm)
        self.mid2 = Conv2dLayer(int(opt.start_channels * (2 ** 2 + 2 ** 3 / 4)), opt.start_channels * (2 ** 2), 3, 1, 1, pad_type = opt.pad, activation = opt.activ, norm = opt.norm)
        self.mid31 = GlobalBlock(opt.start_channels * (2 ** 2), 3, 1, 1, pad_type = opt.pad, activation = opt.activ, norm = opt.norm, sn = False, reduction = 4)
        self.mid32 = ResidualDenseBlock_5C(opt.start_channels * (2 ** 2), opt.start_channels * (2 ** 1), 3, 1, 1, pad_type = opt.pad, activation = opt.activ, norm = opt.norm)
        self.mid4 = Conv2dLayer(opt.start_channels * (2 ** 2), opt.start_channels * (2 ** 2), 3, 1, 1, pad_type = opt.pad, activation = opt.activ, norm = opt.norm)
        # Bottom subnetwork, K = 1
        self.bot1 = Conv2dLayer(opt.in_channels * (4 ** 1), opt.start_channels * (2 ** 1), 3, 1, 1, pad_type = opt.pad, activation = opt.activ, norm = opt.norm)
        self.bot2 = Conv2dLayer(int(opt.start_channels * (2 ** 1 + 2 ** 2 / 4)), opt.start_channels * (2 ** 1), 3, 1, 1, pad_type = opt.pad, activation = opt.activ, norm = opt.norm)
        self.bot31 = GlobalBlock(opt.start_channels * (2 ** 1), 3, 1, 1, pad_type = opt.pad, activation = opt.activ, norm = opt.norm, sn = False, reduction = 4)
        self.bot32 = ResidualDenseBlock_5C(opt.start_channels * (2 ** 1), opt.start_channels * (2 ** 0), 3, 1, 1, pad_type = opt.pad, activation = opt.activ, norm = opt.norm)
        self.bot33 = ResidualDenseBlock_5C(opt.start_channels * (2 ** 1), opt.start_channels * (2 ** 0), 3, 1, 1, pad_type = opt.pad, activation = opt.activ, norm = opt.norm)
        self.bot4 = Conv2dLayer(opt.start_channels * (2 ** 1), opt.start_channels * (2 ** 1), 3, 1, 1, pad_type = opt.pad, activation = opt.activ, norm = opt.norm)
        # Mainstream
        self.main1 = Conv2dLayer(opt.in_channels, opt.start_channels, 3, 1, 1, pad_type = opt.pad, activation = opt.activ, norm = opt.norm)
        self.main2 = Conv2dLayer(int(opt.start_channels * (2 ** 0 + 2 ** 1 / 4)), opt.start_channels, 3, 1, 1, pad_type = opt.pad, activation = opt.activ, norm = opt.norm)
        self.main31 = GlobalBlock(opt.start_channels, 3, 1, 1, pad_type = opt.pad, activation = opt.activ, norm = opt.norm, sn = False, reduction = 4)
        self.main32 = ResidualDenseBlock_5C(opt.start_channels, opt.start_channels // 2, 3, 1, 1, pad_type = opt.pad, activation = opt.activ, norm = opt.norm)
        self.main33 = ResidualDenseBlock_5C(opt.start_channels, opt.start_channels // 2, 3, 1, 1, pad_type = opt.pad, activation = opt.activ, norm = opt.norm)
        self.main34 = ResidualDenseBlock_5C(opt.start_channels, opt.start_channels // 2, 3, 1, 1, pad_type = opt.pad, activation = opt.activ, norm = opt.norm)
        self.main35 = ResidualDenseBlock_5C(opt.start_channels, opt.start_channels // 2, 3, 1, 1, pad_type = opt.pad, activation = opt.activ, norm = opt.norm)
        self.main4 = Conv2dLayer(opt.start_channels, opt.out_channels, 3, 1, 1, pad_type = opt.pad, activation = opt.activ, norm = opt.norm)

    def forward(self, x):
        # PixelUnShuffle                                        input: batch * 3 * 256 * 256
        x1 = PixelUnShuffle.pixel_unshuffle(x, 2)               # out: batch * 12 * 128 * 128
        x2 = PixelUnShuffle.pixel_unshuffle(x, 4)               # out: batch * 48 * 64 * 64
        x3 = PixelUnShuffle.pixel_unshuffle(x, 8)               # out: batch * 192 * 32 * 32
        # Top subnetwork                                        suppose the start_channels = 32
        x3 = self.top1(x3)                                      # out: batch * 256 * 32 * 32
        x3 = self.top21(x3)                                     # out: batch * 256 * 32 * 32
        x3 = self.top22(x3)                                     # out: batch * 256 * 32 * 32
        x3 = self.top3(x3)                                      # out: batch * 256 * 32 * 32
        x3 = F.pixel_shuffle(x3, 2)                             # out: batch * 64 * 64 * 64, ready to be concatenated
        # Middle subnetwork
        x2 = self.mid1(x2)                                      # out: batch * 128 * 64 * 64
        x2 = torch.cat((x2, x3), 1)                             # out: batch * (128 + 64) * 64 * 64
        x2 = self.mid2(x2)                                      # out: batch * 128 * 64 * 64
        x2 = self.mid31(x2)                                     # out: batch * 128 * 64 * 64
        x2 = self.mid32(x2)                                     # out: batch * 128 * 64 * 64
        x2 = self.mid4(x2)                                      # out: batch * 128 * 64 * 64
        x2 = F.pixel_shuffle(x2, 2)                             # out: batch * 32 * 128 * 128, ready to be concatenated
        # Bottom subnetwork
        x1 = self.bot1(x1)                                      # out: batch * 64 * 128 * 128
        x1 = torch.cat((x1, x2), 1)                             # out: batch * (64 + 32) * 128 * 128
        x1 = self.bot2(x1)                                      # out: batch * 64 * 128 * 128
        x1 = self.bot31(x1)                                     # out: batch * 64 * 128 * 128
        x1 = self.bot32(x1)                                     # out: batch * 64 * 128 * 128
        x1 = self.bot33(x1)                                     # out: batch * 64 * 128 * 128
        x1 = self.bot4(x1)                                      # out: batch * 64 * 128 * 128
        x1 = F.pixel_shuffle(x1, 2)                             # out: batch * 16 * 256 * 256, ready to be concatenated
        # U-Net generator with skip connections from encoder to decoder
        x = self.main1(x)                                       # out: batch * 32 * 256 * 256
        x = torch.cat((x, x1), 1)                               # out: batch * (32 + 16) * 256 * 256
        x = self.main2(x)                                       # out: batch * 32 * 256 * 256
        x = self.main31(x)                                      # out: batch * 32 * 256 * 256
        x = self.main32(x)                                      # out: batch * 32 * 256 * 256
        x = self.main33(x)                                      # out: batch * 32 * 256 * 256
        x = self.main34(x)                                      # out: batch * 32 * 256 * 256
        x = self.main35(x)                                      # out: batch * 32 * 256 * 256
        x = self.main4(x)                                       # out: batch * 3 * 256 * 256

        return x

'''
class SubNet(nn.Module):
    def __init__(self, opt):
        super(SubNet, self).__init__()
        self.E1 = Conv2dLayer(opt.in_channels, opt.start_channels, 7, 1, 3, pad_type = opt.pad, activation = opt.activ, norm = 'none')
        self.E2 = Conv2dLayer(opt.start_channels, opt.start_channels * 2, 4, 2, 1, pad_type = opt.pad, activation = opt.activ, norm = opt.norm)
        self.E3 = Conv2dLayer(opt.start_channels * 2, opt.start_channels * 4, 4, 2, 1, pad_type = opt.pad, activation = opt.activ, norm = opt.norm)
        self.RDB1 = ResidualDenseBlock_5C(opt.start_channels * 4, opt.start_channels * 2, 3, 1, 1, pad_type = opt.pad, activation = opt.activ, norm = opt.norm)
        self.RDB2 = ResidualDenseBlock_5C(opt.start_channels * 4, opt.start_channels * 2, 3, 1, 1, pad_type = opt.pad, activation = opt.activ, norm = opt.norm)
        self.RDB3 = ResidualDenseBlock_5C(opt.start_channels * 4, opt.start_channels * 2, 3, 1, 1, pad_type = opt.pad, activation = opt.activ, norm = opt.norm)
        self.RDB4 = ResidualDenseBlock_5C(opt.start_channels * 4, opt.start_channels * 2, 3, 1, 1, pad_type = opt.pad, activation = opt.activ, norm = opt.norm)
        self.RDB5 = ResidualDenseBlock_5C(opt.start_channels * 4, opt.start_channels * 2, 3, 1, 1, pad_type = opt.pad, activation = opt.activ, norm = opt.norm)
        self.RDB6 = ResidualDenseBlock_5C(opt.start_channels * 4, opt.start_channels * 2, 3, 1, 1, pad_type = opt.pad, activation = opt.activ, norm = opt.norm)
        self.RDB7 = ResidualDenseBlock_5C(opt.start_channels * 4, opt.start_channels * 2, 3, 1, 1, pad_type = opt.pad, activation = opt.activ, norm = opt.norm)
        self.RDB8 = ResidualDenseBlock_5C(opt.start_channels * 4, opt.start_channels * 2, 3, 1, 1, pad_type = opt.pad, activation = opt.activ, norm = opt.norm)
        self.D1 = TransposeConv2dLayer(opt.start_channels * 4, opt.start_channels * 2, 3, 1, 1, pad_type = opt.pad, activation = opt.activ, norm = opt.norm)
        self.D2 = TransposeConv2dLayer(opt.start_channels * 2, opt.start_channels, 3, 1, 1, pad_type = opt.pad, activation = opt.activ, norm = opt.norm)
        self.TM = Conv2dLayer(opt.start_channels, opt.start_channels, 1, 1, 0, pad_type = opt.pad, activation = opt.activ, norm = opt.norm)
        self.D3 = Conv2dLayer(opt.start_channels, opt.out_channels, 7, 1, 3, pad_type = opt.pad, activation = 'sigmoid', norm = 'none')

    def forward(self, x):
        # Encoder
        x = self.E1(x)                                         # out: batch * 64 * 128 * 128
        x = self.E2(x)                                         # out: batch * 128 * 64 * 64
        x = self.E3(x)                                         # out: batch * 256 * 32 * 32
        # Bottleneck
        x = self.RDB1(x)                                       # out: batch * 256 * 32 * 32
        x = self.RDB2(x)                                       # out: batch * 256 * 32 * 32
        x = self.RDB3(x)                                       # out: batch * 256 * 32 * 32
        x = self.RDB4(x)                                       # out: batch * 256 * 32 * 32
        x = self.RDB5(x)                                       # out: batch * 256 * 32 * 32
        x = self.RDB6(x)                                       # out: batch * 256 * 32 * 32
        x = self.RDB7(x)                                       # out: batch * 256 * 32 * 32
        x = self.RDB8(x)                                       # out: batch * 256 * 32 * 32
        # Decoder
        x = self.D1(x)                                         # out: batch * 128 * 64 * 64
        x = self.D2(x)                                         # out: batch * 64 * 128 * 128
        tm = self.TM(x)                                        # out: batch * 64 * 128 * 128
        x = self.D3(tm)                                        # out: batch * 3 * 128 * 128
        return tm, x
'''
'''
class SubNet(nn.Module):
    def __init__(self, opt):
        super(SubNet, self).__init__()
        self.E1 = Conv2dLayer(opt.in_channels, opt.start_channels, 7, 1, 3, pad_type = opt.pad, activation = opt.activ, norm = 'none')
        self.E2 = Conv2dLayer(opt.start_channels, opt.start_channels * 2, 3, 1, 1, pad_type = opt.pad, activation = opt.activ, norm = opt.norm)
        self.E3 = Conv2dLayer(opt.start_channels * 2, opt.start_channels * 4, 3, 1, 1, pad_type = opt.pad, activation = opt.activ, norm = opt.norm)
        self.RDB1 = ResidualDenseBlock_5C(opt.start_channels * 4, opt.start_channels * 2, 3, 1, 1, pad_type = opt.pad, activation = opt.activ, norm = opt.norm)
        self.RDB2 = ResidualDenseBlock_5C(opt.start_channels * 4, opt.start_channels * 2, 3, 1, 1, pad_type = opt.pad, activation = opt.activ, norm = opt.norm)
        self.RDB3 = ResidualDenseBlock_5C(opt.start_channels * 4, opt.start_channels * 2, 3, 1, 1, pad_type = opt.pad, activation = opt.activ, norm = opt.norm)
        self.RDB4 = ResidualDenseBlock_5C(opt.start_channels * 4, opt.start_channels * 2, 3, 1, 1, pad_type = opt.pad, activation = opt.activ, norm = opt.norm)
        self.RDB5 = ResidualDenseBlock_5C(opt.start_channels * 4, opt.start_channels * 2, 3, 1, 1, pad_type = opt.pad, activation = opt.activ, norm = opt.norm)
        self.RDB6 = ResidualDenseBlock_5C(opt.start_channels * 4, opt.start_channels * 2, 3, 1, 1, pad_type = opt.pad, activation = opt.activ, norm = opt.norm)
        self.RDB7 = ResidualDenseBlock_5C(opt.start_channels * 4, opt.start_channels * 2, 3, 1, 1, pad_type = opt.pad, activation = opt.activ, norm = opt.norm)
        self.RDB8 = ResidualDenseBlock_5C(opt.start_channels * 4, opt.start_channels * 2, 3, 1, 1, pad_type = opt.pad, activation = opt.activ, norm = opt.norm)
        self.D1 = Conv2dLayer(opt.start_channels, opt.start_channels * 2, 3, 1, 1, pad_type = opt.pad, activation = opt.activ, norm = opt.norm)
        self.D2 = Conv2dLayer(opt.start_channels * 2, opt.start_channels, 3, 1, 1, pad_type = opt.pad, activation = opt.activ, norm = opt.norm)
        self.D3 = Conv2dLayer(opt.start_channels, opt.out_channels, 7, 1, 3, pad_type = opt.pad, activation = 'sigmoid', norm = 'none')

    def forward(self, x):
        # Encoder
        x = self.E1(x)                                         # out: batch * 64 * 128 * 128
        x = self.E2(x)                                         # out: batch * 128 * 128 * 128
        x = self.E3(x)                                         # out: batch * 256 * 128 * 128
        # Bottleneck
        x = self.RDB1(x)                                       # out: batch * 256 * 128 * 128
        x = self.RDB2(x)                                       # out: batch * 256 * 128 * 128
        x = self.RDB3(x)                                       # out: batch * 256 * 128 * 128
        x = self.RDB4(x)                                       # out: batch * 256 * 128 * 128
        x = self.RDB5(x)                                       # out: batch * 256 * 128 * 128
        x = self.RDB6(x)                                       # out: batch * 256 * 128 * 128
        x = self.RDB7(x)                                       # out: batch * 256 * 128 * 128
        x = self.RDB8(x)                                       # out: batch * 256 * 128 * 128
        # Decoder
        x = self.D1(x)                                         # out: batch * 128 * 128 * 128
        tm = self.D2(x)                                        # out: batch * 64 * 128 * 128
        x = self.D3(tm)                                        # out: batch * 3 * 128 * 128
        return tm, x
'''
