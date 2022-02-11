import torch
from .cave_base_functions import Sigmoid, Softplus


class Gradient_Step_A(torch.autograd.Function):
	"""docstring for Gradient_Step_A"""

	@staticmethod
	def forward(ctx, x, a, b, func, mean, var):

		# Save variables for backward
		ctx.save_for_backward(x, a, b)
		ctx.func = func
		ctx.mean = mean
		ctx.var = var

		# f values for variance
		f = func.f(x, a, b)
		df_da = func.df_da(x, a, b)

		# E values for variance
		dem_da = df_da.mean()
		ev = (f ** 2).mean() - f.mean() ** 2 - var
		dev_da = 2 * ((f * df_da).mean() - f.mean() * dem_da)

		# L value for variance
		dlv_da = 2 * ev * dev_da

		# L value for mean if needed
		dlm_da = 0.0
		if mean:

			# E value for mean
			em = f.mean() - mean

			# Update L with L value for mean
			dlm_da = 2 * em * dem_da

		return dlv_da + dlm_da

	@staticmethod
	def backward(ctx, grad_output):

		# Read saved tensors
		x, a, b = ctx.saved_tensors

		# f values for variance
		f = ctx.func.f(x, a, b)
		df_da = ctx.func.df_da(x, a, b)
		df_dx = ctx.func.df_dx(x, a, b)
		d2f_dax = ctx.func.d2f_dax(x, a, b)
		N = x.numel()

		# E values for variance
		ev = (f ** 2).mean() - f.mean() ** 2 - ctx.var
		dem_da = df_da.mean()
		dev_da = 2 * ((f * df_da).mean() - f.mean() * dem_da)
		dev_dx = 2 * df_dx * (f / N - f.mean())
		d2ev_dax = 2 * (df_dx * df_da + f * d2f_dax - \
		                (df_dx * df_da.mean() - f.mean() * d2f_dax) / N)

		# L value for variance
		d2lv_dax = 2 * (dev_dx * dev_da + ev * d2ev_dax)

		# L value for mean if needed
		d2lm_dax = 1.0
		if ctx.mean:

			# E values for mean
			em = f.mean() - ctx.mean
			dem_dx = df_dx / N
			d2em_dax = d2f_dax / N

			# Update L with L value for mean
			d2lm_dax = 2 * (dem_dx * dem_da + em * d2em_dax)

		return grad_output * (d2lm_dax + d2lv_dax), None, None, None, None, None


class GSA(torch.nn.Module):
	"""docstring for GSA"""

	def __init__(self):
		super(GSA, self).__init__()
		self.gsa = Gradient_Step_A.apply

	def forward(self, x, a, b, func, var, mean = None):
		return self.gsa(x, a, b, func, mean, var)


# QUICK TEST

# Initializations
gsa = GSA()
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
out = gsa(x = data,
          a = a,
          b = b,
          func = sig,
          mean = mean,
          var = var)

# Gradient descent step w.r.t. a
a = a - lr * out

# Calculate gradients w.r.t. data
out.backward()

# After stats (should trend towards the specified mean and var)
with torch.no_grad():
	ma = sig.f(data, a, b).mean().item()
	va = sig.f(data, a, b).var(unbiased = False).item()
	print(f'Mean after:\t{ma}\t',
	      f'Var after:\t{va}')


###
