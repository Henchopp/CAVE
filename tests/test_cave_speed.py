import os
import sys
import time

import torch

sys.path.append(os.path.dirname(os.getcwd()))
from utils.cave_base_functions import Sigmoid as SigmoidGT
from utils.cave_base_functions import Softplus as SoftplusGT
from utils.cave_base_functions2 import Sigmoid as SigmoidTest
from utils.cave_base_functions2 import Softplus as SoftplusTest


if __name__ == '__main__':

	# Initializations
	sig_gt = SigmoidGT()
	sp_gt = SoftplusGT()
	sig_test = SigmoidTest()
	sp_test = SoftplusTest()

	sig_pairs = [[sig_gt.Ga, sig_test.Ga],
	             [sig_gt.Gb, sig_test.Gb],
	             [sig_gt.Gab, sig_test.Gab],
	             [sig_gt.dGa_dx, sig_test.dGa_dx],
	             [sig_gt.dGb_dx, sig_test.dGb_dx],
	             [sig_gt.dGab_dx, sig_test.dGab_dx],
	             [sig_gt.Na, sig_test.Na],
	             [sig_gt.Nb, sig_test.Nb],
	             [sig_gt.Nab, sig_test.Nab],
	             [sig_gt.dNa_dx, sig_test.dNa_dx],
	             [sig_gt.dNb_dx, sig_test.dNb_dx],
	             [sig_gt.dNab_dx, sig_test.dNab_dx]]

	sp_pairs = [[sp_gt.Ga, sp_test.Ga],
	            [sp_gt.Gb, sp_test.Gb],
	            [sp_gt.Gab, sp_test.Gab],
	            [sp_gt.dGa_dx, sp_test.dGa_dx],
	            [sp_gt.dGb_dx, sp_test.dGb_dx],
	            [sp_gt.dGab_dx, sp_test.dGab_dx],
	            [sp_gt.Na, sp_test.Na],
	            [sp_gt.Nb, sp_test.Nb],
	            [sp_gt.Nab, sp_test.Nab],
	            [sp_gt.dNa_dx, sp_test.dNa_dx],
	            [sp_gt.dNb_dx, sp_test.dNb_dx],
	            [sp_gt.dNab_dx, sp_test.dNab_dx]]

	a = torch.ones(1)
	b = torch.zeros(1)
	mean = torch.rand(1)
	var = torch.rand(1) * 0.1
	lr = torch.rand(1)
	dim = {'dim': [0,1], 'keepdim': True}

	# Input data (standard normalized)
	x = torch.rand(1000, 500)
	x = (x - x.mean(**dim)) / (x.std(**dim))

	for f_gt, f_test in sig_pairs:
		one_out = True
		if 'Gab' in f_gt.__name__ or 'Nab' in f_gt.__name__:
			one_out = False

		if 'Gab' in f_gt.__name__:
			t_gt = time.time()
			f_gt1, f_gt2 = f_gt(x, a, b, mean, var, dim, lr)
			t_gt = time.time() - t_gt
			t_test = time.time()
			f_test1, f_test2 = f_test(x, a, b, mean, var, dim, lr)
			t_test = time.time() - t_test

		elif 'Ga' in f_gt.__name__:
			t_gt = time.time()
			f_gt1 = f_gt(x, a, b, var, dim, lr)
			t_gt = time.time() - t_gt
			t_test = time.time()
			f_test1 = f_test(x, a, b, var, dim, lr)
			t_test = time.time() - t_test

		elif 'Gb' in f_gt.__name__:
			t_gt = time.time()
			f_gt1 = f_gt(x, a, b, mean, dim, lr)
			t_gt = time.time() - t_gt
			t_test = time.time()
			f_test1 = f_test(x, a, b, mean, dim, lr)
			t_test = time.time() - t_test

		elif 'Nab' in f_gt.__name__:
			t_gt = time.time()
			f_gt1, f_gt2 = f_gt(x, a, b, mean, var, dim, lr)
			t_gt = time.time() - t_gt
			t_test = time.time()
			f_test1, f_test2 = f_test(x, a, b, mean, var, dim, lr)
			t_test = time.time() - t_test

		elif 'Na' in f_gt.__name__:
			t_gt = time.time()
			f_gt1 = f_gt(x, a, b, var, dim, lr)
			t_gt = time.time() - t_gt
			t_test = time.time()
			f_test1 = f_test(x, a, b, var, dim, lr)
			t_test = time.time() - t_test

		elif 'Nb' in f_gt.__name__:
			t_gt = time.time()
			f_gt1 = f_gt(x, a, b, mean, dim, lr)
			t_gt = time.time() - t_gt
			t_test = time.time()
			f_test1 = f_test(x, a, b, mean, dim, lr)
			t_test = time.time() - t_test

		with torch.no_grad():
			name = f_gt.__name__

			if one_out:
				name = 'Sigmoid.' + name + ':' + ' ' * (9 - len(name))
				pe = ((f_test1 - f_gt1) / f_gt1).abs().mean().item()
				print(f'Mean abs percent error of {name} {pe * 100} | {t_test - t_gt}')

			else:
				name = 'Sigmoid.' + name + ':' + ' ' * (9 - len(name))
				pe = ((f_test1 - f_gt1) / f_gt1).abs().mean().item()
				print(f'Mean abs percent error of {name} {pe * 100} | {t_test - t_gt}')

				pe = ((f_test2 - f_gt2) / f_gt2).abs().mean().item()
				print(f'Mean abs percent error of {name} {pe * 100} | {t_test - t_gt}')


	for f_gt, f_test in sp_pairs:
		one_out = True
		if 'Gab' in f_gt.__name__ or 'Nab' in f_gt.__name__:
			one_out = False

		if 'Gab' in f_gt.__name__:
			t_gt = time.time()
			f_gt1, f_gt2 = f_gt(x, a, b, mean, var, dim, lr)
			t_gt = time.time() - t_gt
			t_test = time.time()
			f_test1, f_test2 = f_test(x, a, b, mean, var, dim, lr)
			t_test = time.time() - t_test

		elif 'Ga' in f_gt.__name__:
			t_gt = time.time()
			f_gt1 = f_gt(x, a, b, var, dim, lr)
			t_gt = time.time() - t_gt
			t_test = time.time()
			f_test1 = f_test(x, a, b, var, dim, lr)
			t_test = time.time() - t_test

		elif 'Gb' in f_gt.__name__:
			t_gt = time.time()
			f_gt1 = f_gt(x, a, b, mean, dim, lr)
			t_gt = time.time() - t_gt
			t_test = time.time()
			f_test1 = f_test(x, a, b, mean, dim, lr)
			t_test = time.time() - t_test

		elif 'Nab' in f_gt.__name__:
			t_gt = time.time()
			f_gt1, f_gt2 = f_gt(x, a, b, mean, var, dim, lr)
			t_gt = time.time() - t_gt
			t_test = time.time()
			f_test1, f_test2 = f_test(x, a, b, mean, var, dim, lr)
			t_test = time.time() - t_test

		elif 'Na' in f_gt.__name__:
			t_gt = time.time()
			f_gt1 = f_gt(x, a, b, var, dim, lr)
			t_gt = time.time() - t_gt
			t_test = time.time()
			f_test1 = f_test(x, a, b, var, dim, lr)
			t_test = time.time() - t_test

		elif 'Nb' in f_gt.__name__:
			t_gt = time.time()
			f_gt1 = f_gt(x, a, b, mean, dim, lr)
			t_gt = time.time() - t_gt
			t_test = time.time()
			f_test1 = f_test(x, a, b, mean, dim, lr)
			t_test = time.time() - t_test

		with torch.no_grad():
			name = f_gt.__name__

			if one_out:
				name = 'Softplus.' + name + ':' + ' ' * (8 - len(name))
				pe = ((f_test1 - f_gt1) / f_gt1).abs().mean().item()
				print(f'Mean abs percent error of {name} {pe * 100} | {t_test - t_gt}')

			else:
				name = 'Softplus.' + name + ':' + ' ' * (8 - len(name))
				pe = ((f_test1 - f_gt1) / f_gt1).abs().mean().item()
				print(f'Mean abs percent error of {name} {pe * 100} | {t_test - t_gt}')

				pe = ((f_test2 - f_gt2) / f_gt2).abs().mean().item()
				print(f'Mean abs percent error of {name} {pe * 100} | {t_test - t_gt}')


###