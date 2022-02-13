import torch
from utils.newton_step_a import NSA
from utils.cave_base_functions import Sigmoid, Softplus


class NSA_Mod(torch.nn.Module):
	"""docstring for NSA_Mod"""
	def __init__(self):
		super(NSA_Mod, self).__init__()

	def forward(self, x, a, b, func, var, dim):

		# f values
		f = func.f(x, a, b)
		df_da = func.df_da(x, a, b)
		d2f_da2 = func.d2f_da2(x, a, b)

		# Em values
		dem_da = df_da.mean(**dim)
		d2em_da2 = d2f_da2.mean(**dim)

		# Ev values
		ev = (f ** 2).mean(**dim) - f.mean(**dim) ** 2 - var
		dev_da = 2 * ((f * df_da).mean(**dim) - f.mean(**dim) * dem_da)
		d2ev_da2 = 2 * ((df_da ** 2 + f * d2f_da2).mean(**dim) - dem_da ** 2 - f.mean(**dim) * d2em_da2)

		# Lv values
		dlv_da = 2 * ev * dev_da
		d2lv_da2 = 2 * (dev_da ** 2 + ev * d2ev_da2)

		# Newton step
		na = dlv_da / d2lv_da2

		return dlv_da


def test_nsa(opt, opt_mod):
	
	# Initializations
	sig = Sigmoid()
	a = torch.ones(1)
	b = torch.zeros(1)
	mean = torch.Tensor([0.2])
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
	              var = var,
	              dim = dim)
	out_opt_mod = opt_mod(x = x_opt_mod,
	                      a = a,
	                      b = b,
	                      func = sig,
	                      var = var,
	                      dim = dim)

	# Calculate gradients w.r.t. x
	out_opt = out_opt.sum()
	out_opt_mod = out_opt_mod.sum()

	out_opt.backward()
	out_opt_mod.backward()

	# After stats (should trend towards the specified mean and var)
	with torch.no_grad():
		mse = ((x_opt.grad - x_opt_mod.grad) ** 2).mean().item()
		print(f'Grad of {str(opt)[:-2]}:\n{x_opt.grad}\n',
		      f'Grad of {str(opt_mod)[:-2]}:\n{x_opt_mod.grad}\n',
		      f'MSE: {mse}')


if __name__ == '__main__':
	test_nsa(opt = NSA(), opt_mod = NSA_Mod())


###