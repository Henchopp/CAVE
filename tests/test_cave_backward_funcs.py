import os
import sys

import torch

sys.path.append(os.path.dirname(os.getcwd()))
from utils.cave_base_functions import Sigmoid, Softplus


if __name__ == '__main__':

	torch.manual_seed(0)
	torch.autograd.set_detect_anomaly(True)

	# Initializations
	sig = Sigmoid()
	sp = Softplus()

	sig_pairs = [[sig.Ga, sig.dGa_dx],
	             [sig.Gb, sig.dGb_dx],
	             [sig.Gab, sig.dGab_dx],
	             [sig.Na, sig.dNa_dx],
	             [sig.Nb, sig.dNb_dx],
	             [sig.Nab, sig.dNab_dx]]

	sp_pairs = [[sp.Ga, sp.dGa_dx],
	            [sp.Gb, sp.dGb_dx],
	            [sp.Gab, sp.dGab_dx],
	            [sp.Na, sp.dNa_dx],
	            [sp.Nb, sp.dNb_dx],
	            [sp.Nab, sp.dNab_dx]]

	a = torch.ones(1)
	b = torch.zeros(1)
	mean = torch.rand(1)
	var = torch.rand(1) * 0.1
	lr = torch.rand(1)
	dim = {'dim': [0,1], 'keepdim': True}

	# Input data (standard normalized)
	x = torch.rand(5000, 1000)
	x = (x - x.mean(**dim)) / (x.std(**dim))

	for f, df_dx in sig_pairs:
		xg = x.clone()
		xg.requires_grad = True
		one_out = True
		if 'Gab' in f.__name__ or 'Nab' in f.__name__:
			xgg = x.clone()
			xgg.requires_grad = True
			one_out = False

		if 'Em' in f.__name__:
			df_out = df_dx(x, a, b, dim)
			if f == sig.Em:
				f_out = f(xg, a, b, mean, dim)
			else:
				f_out = f(xg, a, b, dim)
			f_out.backward()

		elif 'Ev' in f.__name__:
			df_out = df_dx(x, a, b, dim)
			if f == sig.Ev:
				f_out = f(xg, a, b, var, dim)
			else:
				f_out = f(xg, a, b, dim)
			f_out.backward()

		elif 'Lm' in f.__name__:
			df_out = df_dx(x, a, b, mean, dim)
			f_out = f(xg, a, b, mean, dim)
			f_out.backward()

		elif 'Lv' in f.__name__:
			df_out = df_dx(x, a, b, var, dim)
			f_out = f(xg, a, b, var, dim)
			f_out.backward()

		elif 'Gab' in f.__name__:
			df_out1, df_out2 = df_dx(x, a, b, mean, var, dim, lr)
			f_out1, _ = f(xg, a, b, mean, var, dim, lr)
			_, f_out2 = f(xgg, a, b, mean, var, dim, lr)
			f_out1.backward()
			f_out2.backward()

		elif 'Ga' in f.__name__:
			df_out = df_dx(x, a, b, var, dim, lr)
			f_out = f(xg, a, b, var, dim, lr)
			f_out.backward()

		elif 'Gb' in f.__name__:
			df_out = df_dx(x, a, b, mean, dim, lr)
			f_out = f(xg, a, b, mean, dim, lr)
			f_out.backward()

		elif 'Nab' in f.__name__:
			df_out1, df_out2 = df_dx(x, a, b, mean, var, dim, lr)
			f_out1, _ = f(xg, a, b, mean, var, dim, lr)
			_, f_out2 = f(xgg, a, b, mean, var, dim, lr)
			f_out1.backward()
			f_out2.backward()

		elif 'Na' in f.__name__:
			df_out = df_dx(x, a, b, var, dim, lr)
			f_out = f(xg, a, b, var, dim, lr)
			f_out.backward()

		elif 'Nb' in f.__name__:
			df_out = df_dx(x, a, b, mean, dim, lr)
			f_out = f(xg, a, b, mean, dim, lr)
			f_out.backward()

		with torch.no_grad():
			name = f.__name__

			if one_out:
				name = 'Sigmoid.' + name + ':' + ' ' * (5 - len(name))
				pe = ((df_out - xg.grad) / xg.grad).abs().mean().item()
				pe = '{:e}'.format(pe * 100)
				print(f'Mean abs percent error of {name} {pe} %')

			else:
				name = 'Sigmoid.' + name + ':' + ' ' * (5 - len(name))
				pe = ((df_out1 - xg.grad) / xg.grad).abs().mean().item()
				pe = '{:e}'.format(pe * 100)
				print(f'Mean abs percent error of {name} {pe} %')

				pe = ((df_out2 - xgg.grad) / xgg.grad).abs().mean().item()
				pe = '{:e}'.format(pe * 100)
				print(f'Mean abs percent error of {name} {pe} %')


	for f, df_dx in sp_pairs:
		xg = x.clone()
		xg.requires_grad = True
		one_out = True
		if 'Gab' in f.__name__ or 'Nab' in f.__name__:
			xgg = x.clone()
			xgg.requires_grad = True
			one_out = False

		if 'Em' in f.__name__:
			df_out = df_dx(x, a, b, dim)
			if f == sp.Em:
				f_out = f(xg, a, b, mean, dim)
			else:
				f_out = f(xg, a, b, dim)
			f_out.backward()

		elif 'Ev' in f.__name__:
			df_out = df_dx(x, a, b, dim)
			if f == sp.Ev:
				f_out = f(xg, a, b, var, dim)
			else:
				f_out = f(xg, a, b, dim)
			f_out.backward()

		elif 'Lm' in f.__name__:
			df_out = df_dx(x, a, b, mean, dim)
			f_out = f(xg, a, b, mean, dim)
			f_out.backward()

		elif 'Lv' in f.__name__:
			df_out = df_dx(x, a, b, var, dim)
			f_out = f(xg, a, b, var, dim)
			f_out.backward()

		elif 'Gab' in f.__name__:
			df_out1, df_out2 = df_dx(x, a, b, mean, var, dim, lr)
			f_out1, _ = f(xg, a, b, mean, var, dim, lr)
			_, f_out2 = f(xgg, a, b, mean, var, dim, lr)
			f_out1.backward()
			f_out2.backward()

		elif 'Ga' in f.__name__:
			df_out = df_dx(x, a, b, var, dim, lr)
			f_out = f(xg, a, b, var, dim, lr)
			f_out.backward()

		elif 'Gb' in f.__name__:
			df_out = df_dx(x, a, b, mean, dim, lr)
			f_out = f(xg, a, b, mean, dim, lr)
			f_out.backward()

		elif 'Nab' in f.__name__:
			df_out1, df_out2 = df_dx(x, a, b, mean, var, dim, lr)
			f_out1, _ = f(xg, a, b, mean, var, dim, lr)
			_, f_out2 = f(xgg, a, b, mean, var, dim, lr)
			f_out1.backward()
			f_out2.backward()

		elif 'Na' in f.__name__:
			df_out = df_dx(x, a, b, var, dim, lr)
			f_out = f(xg, a, b, var, dim, lr)
			f_out.backward()

		elif 'Nb' in f.__name__:
			df_out = df_dx(x, a, b, mean, dim, lr)
			f_out = f(xg, a, b, mean, dim, lr)
			f_out.backward()

		with torch.no_grad():
			name = f.__name__

			if one_out:
				name = 'Softplus.' + name + ':' + ' ' * (4 - len(name))
				pe = ((df_out - xg.grad) / xg.grad).abs().mean().item()
				pe = '{:e}'.format(pe * 100)
				print(f'Mean abs percent error of {name} {pe} %')

			else:
				name = 'Softplus.' + name + ':' + ' ' * (4 - len(name))
				pe = ((df_out1 - xg.grad) / xg.grad).abs().mean().item()
				pe = '{:e}'.format(pe * 100)
				print(f'Mean abs percent error of {name} {pe} %')

				pe = ((df_out2 - xgg.grad) / xgg.grad).abs().mean().item()
				pe = '{:e}'.format(pe * 100)
				print(f'Mean abs percent error of {name} {pe} %')





###