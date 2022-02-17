import os
import sys

import torch

sys.path.append(os.path.dirname(os.getcwd()))
from cave import CAVE
from utils.cave_base_functions import Sigmoid


x = torch.rand(500, 200)
a = torch.ones(1)
b = torch.zeros(1)
mean = torch.Tensor([0.2])
var = torch.Tensor([0.09])
lr = torch.rand(1)
dim = None

n_gd = 10
n_nm = 25
lr_gd = 2.0
lr_nm = 1.0

cave = CAVE(func = Sigmoid(),
            n_step_gd = n_gd,
            n_step_nm = n_nm,
            lr_gd = lr_gd,
            lr_nm = lr_nm,
            output_log = 'test1')

print(x.mean(), x.var(unbiased = False))
out = cave(x,
           low = 0.0,
           high = 1.0,
           mean = 0.1,
           var = 0.09)
print(out.mean(), out.var(unbiased = False))