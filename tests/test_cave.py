import os
import sys

import torch

sys.path.append(os.path.dirname(os.getcwd()))
from cave import CAVE
from utils.cave_base_functions import Sigmoid, Softplus, ReLU


if __name__ == '__main__':
	
	# User-defined variables
	x = torch.rand(100, 1)

	mean = 0.2
	var = 0.16
	sparse = True

	n_gd = 10
	n_nm = 15
	lr_gd = 2.0
	lr_nm = 1.0

	output_log = 'test1'

	# Other initializations
	cave = CAVE(func = Sigmoid(),
	            n_step_gd = n_gd,
	            n_step_nm = n_nm,
	            lr_gd = lr_gd,
	            lr_nm = lr_nm,
	            output_log = output_log)

	def loss_fn(x, mean, var):
		return (x.mean() - mean) ** 2 + (x.var(unbiased = False) - var) ** 2

	# Print target stats
	m = '{:.8f}'.format(mean)
	v = '{:.8f}'.format(var)
	l = '{:.8f}'.format(0)
	print(f'Target   Mean / Var / Loss: {m} | {v} | {l}')

	# Print input stats
	m = '{:.8f}'.format(x.mean().item())
	v = '{:.8f}'.format(x.var(unbiased = False).item())
	l = '{:.8f}'.format(loss_fn(x, mean, var).item())
	print(f'Input    Mean / Var / Loss: {m} | {v} | {l}')

	# Run cave and get loss w.r.t. x
	x.requires_grad = True
	out = cave(x,
	           low = 0.0,
	           high = 1.0,
	           mean = mean,
	           var = var,
	           sparse = sparse)
	loss = loss_fn(out, mean, var)
	loss.backward()

	# Print output stats
	m = '{:.8f}'.format(out.mean().item())
	v = '{:.8f}'.format(out.var(unbiased = False).item())
	l = '{:.8f}'.format(loss.item())
	print(f'Output   Mean / Var / Loss: {m} | {v} | {l}')

	# Backward to get closer to target
	grad = x.grad.clone().abs()
	x = x - 0.01 * x.grad
	with torch.no_grad():
		out = cave(x,
		           low = 0.0,
		           high = 1.0,
		           mean = mean,
		           var = var,
		           sparse = sparse)

	# Print backward stats
	m = '{:.8f}'.format(out.mean().item())
	v = '{:.8f}'.format(out.var(unbiased = False).item())
	l = '{:.8f}'.format(loss_fn(out, mean, var).item())
	print(f'Backward Mean / Var / Loss: {m} | {v} | {l}\n')

	m = '{:.4e}'.format(grad.mean().item())
	v = '{:.4e}'.format(grad.std().item())
	print(f'Grad Mag Mean / Std:        {m} | {v}')


###