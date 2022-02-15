import os
import sys

import torch

sys.path.append(os.path.dirname(os.getcwd()))
from utils.cave_base_functions import Sigmoid, Softplus


if __name__ == '__main__':

	# Initializations
	sig = Sigmoid()
	sp = Softplus()

	sig_pairs = [[sig.Em, sig.dEm_dx],
	             [sig.dEm_da, sig.d2Em_dax],
	             [sig.dEm_db, sig.d2Em_dbx],
	             [sig.d2Em_da2, sig.d3Em_da2x],
	             [sig.d2Em_dab, sig.d3Em_dabx],
	             [sig.d2Em_db2, sig.d3Em_db2x],
	             [sig.Ev, sig.dEv_dx],
	             [sig.dEv_da, sig.d2Ev_dax],
	             [sig.dEv_db, sig.d2Ev_dbx],
	             [sig.d2Ev_da2, sig.d3Ev_da2x],
	             [sig.d2Ev_dab, sig.d3Ev_dabx],
	             [sig.d2Ev_db2, sig.d3Ev_db2x],
	             [sig.Lm, sig.dLm_dx],
	             [sig.dLm_da, sig.d2Lm_dax],
	             [sig.dLm_db, sig.d2Lm_dbx],
	             [sig.d2Lm_da2, sig.d3Lm_da2x],
	             [sig.d2Lm_dab, sig.d3Lm_dabx],
	             [sig.d2Lm_db2, sig.d3Lm_db2x],
	             [sig.Lv, sig.dLv_dx],
	             [sig.dLv_da, sig.d2Lv_dax],
	             [sig.dLv_db, sig.d2Lv_dbx],
	             [sig.d2Lv_da2, sig.d3Lv_da2x],
	             [sig.d2Lv_dab, sig.d3Lv_dabx],
	             [sig.d2Lv_db2, sig.d3Lv_db2x],
	             [sig.Ga, sig.dGa_dx],
	             [sig.Gb, sig.dGb_dx],
	             [sig.Gab, sig.dGab_dx],
	             [sig.Na, sig.dNa_dx],
	             [sig.Nb, sig.dNb_dx],
	             [sig.Nab, sig.dNab_dx]]

	sp_pairs = [[sp.Em, sp.dEm_dx],
	            [sp.dEm_da, sp.d2Em_dax],
	            [sp.dEm_db, sp.d2Em_dbx],
	            [sp.d2Em_da2, sp.d3Em_da2x],
	            [sp.d2Em_dab, sp.d3Em_dabx],
	            [sp.d2Em_db2, sp.d3Em_db2x],
	            [sp.Ev, sp.dEv_dx],
	            [sp.dEv_da, sp.d2Ev_dax],
	            [sp.dEv_db, sp.d2Ev_dbx],
	            [sp.d2Ev_da2, sp.d3Ev_da2x],
	            [sp.d2Ev_dab, sp.d3Ev_dabx],
	            [sp.d2Ev_db2, sp.d3Ev_db2x],
	            [sp.Lm, sp.dLm_dx],
	            [sp.dLm_da, sp.d2Lm_dax],
	            [sp.dLm_db, sp.d2Lm_dbx],
	            [sp.d2Lm_da2, sp.d3Lm_da2x],
	            [sp.d2Lm_dab, sp.d3Lm_dabx],
	            [sp.d2Lm_db2, sp.d3Lm_db2x],
	            [sp.Lv, sp.dLv_dx],
	            [sp.dLv_da, sp.d2Lv_dax],
	            [sp.dLv_db, sp.d2Lv_dbx],
	            [sp.d2Lv_da2, sp.d3Lv_da2x],
	            [sp.d2Lv_dab, sp.d3Lv_dabx],
	            [sp.d2Lv_db2, sp.d3Lv_db2x],
	            [sp.Ga, sp.dGa_dx],
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
	x = torch.rand(1000, 500)
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
				name = 'Sigmoid.' + name + ':' + ' ' * (9 - len(name))
				pe = ((df_out - xg.grad) / xg.grad).abs().mean().item()
				print(f'Mean abs percent error of {name} {pe * 100}')

			else:
				name = 'Sigmoid.' + name + ':' + ' ' * (9 - len(name))
				pe = ((df_out1 - xg.grad) / xg.grad).abs().mean().item()
				print(f'Mean abs percent error of {name} (a) {pe * 100}')

				pe = ((df_out2 - xgg.grad) / xgg.grad).abs().mean().item()
				print(f'Mean abs percent error of {name} (b) {pe * 100}')


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
				name = 'Softplus.' + name + ':' + ' ' * (8 - len(name))
				pe = ((df_out - xg.grad) / xg.grad).abs().mean().item()
				print(f'Mean abs percent error of {name} {pe * 100}')

			else:
				name = 'Softplus.' + name + ':' + ' ' * (8 - len(name))
				pe = ((df_out1 - xg.grad) / xg.grad).abs().mean().item()
				print(f'Mean abs percent error of {name} (a) {pe * 100}')

				pe = ((df_out2 - xgg.grad) / xgg.grad).abs().mean().item()
				print(f'Mean abs percent error of {name} (b) {pe * 100}')





###