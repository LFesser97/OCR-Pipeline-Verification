import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from utils import get_default_device
import R2
import time

conv1=nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride=(2,1), padding=1)
bn=nn.BatchNorm2d(num_features=4, affine=False)
relu=nn.ReLU()
conv2=nn.Conv2d(in_channels=4, out_channels=4, kernel_size=3, stride=1, padding=1)
maxpool=nn.MaxPool2d(kernel_size=(2,2),stride=(2,2))

h=16
w=16

pre=torch.rand(100,1,h,w)
bn(conv1(pre))

# print(bn.running_mean,bn.running_var)


bn.eval()
input = torch.rand(1,1, h, w)
# print(input)
output = conv1(input)
output=bn(output)
output=relu(output)
output=conv2(output)
output=maxpool(output)
# output=relu(output)
print(output)

# print(bn.running_mean,bn.running_var)
#input flatten from shape (h,w,c)
input=input.squeeze(0)
input=input.permute(1,2,0)
input=input.flatten()
# print(input)

input_dp = R2.DeepPoly.deeppoly_from_perturbation(input, eps=0.001, truncate=(0, 1))

# lin1 = R2.Linear(len(input), len(input))
# lin1.assign(torch.eye(len(input)))
# lin1_out = lin1(input_dp)

R2conv1,h,w = R2.Convolution_Layer.convert(conv1,h,w)
conv1_out = R2conv1(input_dp)
R2bn = R2.BatchNormalization.convert(bn,prev_layer=R2conv1)
bn_out = R2bn(conv1_out)
R2relu = R2.ReLU(prev_layer=R2bn)
nr1_out = R2relu(bn_out)
# print(nr1_out.lb,nr1_out.ub)


R2conv2,h,w = R2.Convolution_Layer.convert(conv2,h,w,prev_layer=R2relu)
conv2_out = R2conv2(nr1_out)
R2maxpool=R2.MaxPool.convert(maxpool,int(conv2_out.dim/(h*w)),h,w,prev_layer=R2conv2)
out=R2maxpool(conv2_out)
# R2relu2 = R2.ReLU(prev_layer=R2maxpool)
# out = R2relu2(maxpool_out)
print(out.lb)
print(out.ub)

# R2select=R2.Selection([1,3],prev_layer=R2maxpool)
# select=R2select(out)
# print(select.lb)
# print(select.ub)

# R2output=R2conv1.weight @ input + R2conv1.bias
# R2output=R2conv2.weight @ R2output + R2conv2.bias
# print(R2output)