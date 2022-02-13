import torch
from utils.newton_step_ab import NSAB
from utils.cave_base_functions import Sigmoid, Softplus


class NSAB_Mod(torch.nn.Module):
	"""docstring for NSAB_Mod"""
	def __init__(self):
		super(NSAB_Mod, self).__init__()

	def forward(self, x, a, b, func, mean, var, dim):

		# f values
		f = func.f(x, a, b)
		df_da = func.df_da(x, a, b)
		df_db = func.df_db(x, a, b)
		d2f_da2 = func.d2f_da2(x, a, b)
		d2f_dab = func.d2f_dab(x, a, b)
		d2f_db2 = func.d2f_db2(x, a, b)

		# Em values
		em = f.mean(**dim) - mean
		dem_da = df_da.mean(**dim)
		dem_db = df_db.mean(**dim)
		d2em_da2 = d2f_da2.mean(**dim)
		d2em_dab = d2f_dab.mean(**dim)
		d2em_db2 = d2f_db2.mean(**dim)

		# Ev values
		ev = (f ** 2).mean(**dim) - f.mean(**dim) ** 2 - var
		dev_da = 2 * ((f * df_da).mean(**dim) - f.mean(**dim) * dem_da)
		dev_db = 2 * ((f * df_db).mean(**dim) - f.mean(**dim) * dem_db)
		d2ev_da2 = 2 * ((df_da ** 2 + f * d2f_da2).mean(**dim) - dem_da ** 2 - f.mean(**dim) * d2em_da2)
		d2ev_dab = 2 * ((df_da * df_db + f * d2f_dab).mean(**dim) - dem_da * dem_db - f.mean(**dim) * d2em_dab)
		d2ev_db2 = 2 * ((df_db ** 2 + f * d2f_db2).mean(**dim) - dem_db ** 2 - f.mean(**dim) * d2em_db2)

		# L values
		dl_da = 2 * (em * dem_da + ev * dev_da)
		dl_db = 2 * (em * dem_db + ev * dev_db)
		d2l_da2 = 2 * (dem_da ** 2 + em * d2em_da2 + dev_da ** 2 + ev * d2ev_da2)
		d2l_dab = 2 * (dem_da * dem_db + em * d2em_dab + dev_da * dev_db + ev * d2ev_dab)
		d2l_db2 = 2 * (dem_db ** 2 + em * d2em_db2 + dev_db ** 2 + ev * d2ev_db2)
		
		# Newton steps
		na = (dl_da * d2l_db2 - dl_db * d2l_dab) / (d2l_da2 * d2l_db2 - d2l_dab ** 2)
		nb = (dl_db * d2l_da2 - dl_da * d2l_dab) / (d2l_da2 * d2l_db2 - d2l_dab ** 2)

		return na, nb


def test_nsab(opt, opt_mod):
	
	# Initializations
	sig = Sigmoid()
	a = torch.ones(1)
	b = torch.zeros(1)
	mean = torch.rand(10, 1)
	var = torch.rand(10, 1) * 0.1
	lr = torch.ones(1)

	dim = {'dim': [1], 'keepdim': True}

	# Input data (standard normalized)
	x_opt = torch.rand(10, 5)
	x_opt = (x_opt - x_opt.mean(**dim)) / (x_opt.std(**dim))
	x_opt_mod = x_opt.clone()

	# Track x gradients
	x_opt.requires_grad = True
	x_opt_mod.requires_grad = True

	# Calculate grad descent step
	out_opt = opt(x = x_opt,
	              a = a,
	              b = b,
	              func = sig,
	              mean = mean,
	              var = var,
	              dim = dim)
	out_opt_mod = opt_mod(x = x_opt_mod,
	                      a = a,
	                      b = b,
	                      func = sig,
	                      mean = mean,
	                      var = var,
	                      dim = dim)

	# Calculate gradients w.r.t. x
	out_opt = out_opt[0].sum() + out_opt[1].sum()
	out_opt_mod = out_opt_mod[0].sum() + out_opt_mod[1].sum()

	out_opt.backward()
	out_opt_mod.backward()

	# After stats (should trend towards the specified mean and var)
	with torch.no_grad():
		mse = ((x_opt.grad - x_opt_mod.grad) ** 2).mean().item()
		print(f'Grad of {str(opt)[:-2]}:\n{x_opt.grad}\n',
		      f'Grad of {str(opt_mod)[:-2]}:\n{x_opt_mod.grad}\n',
		      f'MSE: {mse}')


if __name__ == '__main__':
	test_nsab(opt = NSAB(), opt_mod = NSAB_Mod())


###