import torch
from cave_base_functions import Sigmoid, Softplus


class Gradient_Step_A(torch.autograd.Function):
	"""docstring for Gradient_Step_A"""
	
	@staticmethod
	def forward(ctx, x, a, b, func, var, dim):

		# Save variables for backward
		ctx.save_for_backward(x, a, b)
		ctx.func = func
		ctx.var = var
		ctx.dim = dim

		# f values
		f = func.f(x, a, b)
		df_da = func.df_da(x, a, b)

		# E values
		dem_da = df_da.mean(**dim)
		ev = (f ** 2).mean(**dim) - f.mean(**dim) ** 2 - var
		dev_da = 2 * ((f * df_da).mean(**dim) - f.mean(**dim) * dem_da)

		# L value
		dlv_da = 2 * ev * dev_da

		return dlv_da

	@staticmethod
	def backward(ctx, grad_output):

		# Read saved tensors
		x, a, b = ctx.saved_tensors

		# f values
		f = ctx.func.f(x, a, b)
		df_da = ctx.func.df_da(x, a, b)
		df_dx = ctx.func.df_dx(x, a, b)
		d2f_dax = ctx.func.d2f_dax(x, a, b)

		# Get N
		N = 1
		for i in ctx.dim['dim']:
			N *= x.shape[i]

		# E values
		ev = (f ** 2).mean(**ctx.dim) - f.mean(**ctx.dim) ** 2 - ctx.var
		dem_da = df_da.mean(**ctx.dim)
		dev_da = 2 * ((f * df_da).mean(**ctx.dim) - f.mean(**ctx.dim) * dem_da)
		dev_dx = 2 * df_dx * (f / N - f.mean(**ctx.dim))
		d2ev_dax = 2 * (df_dx * df_da + f * d2f_dax - \
		                (df_dx * df_da.mean(**ctx.dim) - f.mean(**ctx.dim) * d2f_dax) / N)

		# L value
		d2lv_dax = 2 * (dev_dx * dev_da + ev * d2ev_dax)

		return grad_output * d2lv_dax, None, None, None, None, None


class GSA(torch.nn.Module):
	"""docstring for GSA"""

	def __init__(self):
		super(GSA, self).__init__()
		self.gsa = Gradient_Step_A.apply

	def forward(self, x, a, b, func, var, dim):
		return self.gsa(x, a, b, func, var, dim)
		

# QUICK TEST

# Initializations
gsa = GSA()
sig = Sigmoid()
a = torch.ones(1)
b = torch.zeros(1)
mean = torch.Tensor([0.2])
var = torch.rand(5,1) * 0.1
lr = torch.ones(1)

# Input data standard normalized
data = torch.rand(5, 1000)
dim = {'dim': [1], 'keepdim': True}
data = (data - data.mean(**dim)) / (data.std(**dim))

# Init stats
mb = sig.f(data, a, b).mean(**dim)
vb = sig.f(data, a, b).var(unbiased = False, **dim)
print(f'Mean chosen:\n{mean}\n',
      f'Var chosen: \n{var}\n')
print(f'Mean before:\n{mb}\n',
      f'Var before:\n{vb}\n')

# Track data gradients
data.requires_grad = True

# Calculate grad descent step w.r.t. b
out = gsa(x = data,
          a = a,
          b = b,
          func = sig,
          var = var,
          dim = dim)

# Gradient descent step w.r.t. b
a = a - lr * out

# Calculate gradients w.r.t. data
out = out.sum()
out.backward()

# After stats (should trend towards the specified mean and var)
with torch.no_grad():
	ma = sig.f(data, a, b).mean(**dim)
	va = sig.f(data, a, b).var(unbiased = False, **dim)
	print(f'Mean after:\n{ma}\n',
	      f'Var after:\n{va}')


###