import torch
from utils.newton_step_b import NSB
from utils.cave_base_functions import Sigmoid, Softplus


class NSB_Mod(torch.nn.Module):
	"""docstring for NSB_Mod"""
	def __init__(self):
		super(NSB_Mod, self).__init__()

	def forward(self, x, a, b, func, mean, dim):

		# f values
		f = func.f(x, a, b)
		df_db = func.df_db(x, a, b)
		d2f_db2 = func.d2f_db2(x, a, b)

		# Em values
		em = f.mean(**dim) - mean
		dem_db = df_db.mean(**dim)
		d2em_db2 = d2f_db2.mean(**dim)

		# Lm values
		dlm_db = 2 * em * dem_db
		d2lm_db2 = 2 * (dem_db ** 2 + em * d2em_db2)

		# Newton step
		nb = dlm_db / d2lm_db2

		return nb


def test_nsb(opt, opt_mod):
	
	# Initializations
	sig = Sigmoid()
	a = torch.ones(1)
	b = torch.zeros(1)
	mean = torch.rand(10,1)
	var = torch.Tensor([0.01])
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
	              dim = dim)
	out_opt_mod = opt_mod(x = x_opt_mod,
	                      a = a,
	                      b = b,
	                      func = sig,
	                      mean = mean,
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
	test_nsb(opt = NSB(), opt_mod = NSB_Mod())


###