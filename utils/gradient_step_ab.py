import torch
from cave_base_functions import Sigmoid, Softplus


class Gradient_Step_AB(torch.autograd.Function):
	"""docstring for Gradient_Step_AB"""
	
	@staticmethod
	def forward(ctx, x, x_copy, a, b, func, mean, var):
		
		# Save variables for backward
		ctx.save_for_backward(x, a, b)
		ctx.func = func
		ctx.mean = mean
		ctx.var = var

		# f values
		f = func.f(x, a, b)
		df_da = func.df_da(x, a, b)
		df_db = func.df_db(x, a, b)

		# Em values
		em = f.mean() - mean
		dem_da = df_da.mean()
		dem_db = df_db.mean()

		# Ev values
		ev = (f ** 2).mean() - f.mean() ** 2 - var
		dev_da = 2 * ((f * df_da).mean() - f.mean() * dem_da)
		dev_db = 2 * ((f * df_db).mean() - f.mean() * dem_db)

		# L values
		dl_da = 2 * (em * dem_da + ev * dev_da)
		dl_db = 2 * (em * dem_db + ev * dev_db)

		return dl_da, dl_db

	@staticmethod
	def backward(ctx, grad_output1, grad_output2):

		# Read saved tensors
		x, a, b = ctx.saved_tensors

		# f values
		f = ctx.func.f(x, a, b)
		df_da = ctx.func.df_da(x, a, b)
		df_db = ctx.func.df_db(x, a, b)
		df_dx = ctx.func.df_dx(x, a, b)
		d2f_dax = ctx.func.d2f_dax(x, a, b)
		d2f_dbx = ctx.func.d2f_dbx(x, a, b)
		N = x.numel()

		# Em values
		em = f.mean() - ctx.mean
		dem_da = df_da.mean()
		dem_db = df_db.mean()
		dem_dx = df_dx / N
		d2em_dax = d2f_dax / N
		d2em_dbx = d2f_dbx / N

		# Ev values
		ev = (f ** 2).mean() - f.mean() ** 2 - ctx.var
		dev_da = 2 * ((f * df_da).mean() - f.mean() * dem_da)
		dev_db = 2 * ((f * df_db).mean() - f.mean() * dem_db)
		dev_dx = 2 * df_dx * (f / N - f.mean())
		d2ev_dax = 2 * (df_dx * df_da + f * d2f_dax - \
		                (df_dx * df_da.mean() - f.mean() * d2f_dax) / N)
		d2ev_dbx = 2 * (df_dx * df_db + f * d2f_dbx - \
		                (df_dx * df_db.mean() - f.mean() * d2f_dbx) / N)

		# L values
		d2l_dax = 2 * (dem_dx * dem_da + em * d2em_dax + dev_dx * dev_da + ev * d2ev_dax)
		d2l_dbx = 2 * (dem_dx * dem_db + em * d2em_dbx + dev_dx * dev_db + ev * d2ev_dbx)

		return grad_output1 * d2l_dax, grad_output2 * d2l_dbx, None, None, None, None, None
		

class GSAB(torch.nn.Module):
	"""docstring for GSAB"""

	def __init__(self):
		super(GSAB, self).__init__()
		self.gsab = Gradient_Step_AB.apply

	def forward(self, x, a, b, func, mean, var):
		return self.gsab(x, x, a, b, func, mean, var)
		

# QUICK TEST

# Initializations
gsab = GSAB()
sig = Sigmoid()
a = torch.ones(1)
b = torch.zeros(1)
mean = torch.Tensor([0.1])
var = torch.Tensor([0.01])
lr = torch.ones(1)

# Input data standard normalized
data = torch.rand(1000)
data = (data - data.mean()) / (data.std())

# Init stats
mb = sig.f(data, a, b).mean().item()
vb = sig.f(data, a, b).var(unbiased = False).item()
print(f'Mean chosen:\t{mean.item()}\t',
      f'Var chosen: \t{var.item()}')
print(f'Mean before:\t{mb}\t',
      f'Var before:\t{vb}')

# Track data gradients
data.requires_grad = True

# Calculate grad descent step w.r.t. a
out = gsab(x = data,
           a = a,
           b = b,
           func = sig,
           mean = mean,
           var = var)

# Gradient descent step w.r.t. a
a = a - lr * out[0]
b = b - lr * out[1]

# Calculate gradients w.r.t. data
out = out[0] + out[1]
out.backward()

# After stats (should trend towards the specified mean and var)
with torch.no_grad():
	ma = sig.f(data, a, b).mean().item()
	va = sig.f(data, a, b).var(unbiased = False).item()
	print(f'Mean after:\t{ma}\t',
	      f'Var after:\t{va}')


###