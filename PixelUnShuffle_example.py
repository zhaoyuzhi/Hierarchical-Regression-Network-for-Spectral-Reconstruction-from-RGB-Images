import PixelUnShuffle
import torch
import torch.nn as nn
import torch.nn.functional as F

x = torch.range(start = 0, end = 31).reshape([1, 8, 2, 2])
print('x:')
print(x.shape)
print(x)

y = F.pixel_shuffle(x, 2)
print('y:')
print(y.shape)
print(y)

x_ = PixelUnShuffle.pixel_unshuffle(y, 2)
print('x_:')
print(x_.shape)
print(x_)

s = torch.randn(1, 3, 256, 256)
ss = PixelUnShuffle.pixel_unshuffle(s, 2)
ssss = PixelUnShuffle.pixel_unshuffle(s, 4)
print(ss.shape, ssss.shape)
