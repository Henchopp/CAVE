import torch
from cave_base_functions import Sigmoid, Softplus


class Newton_Step_A(torch.autograd.Function):
	"""docstring for Newton_Step_A"""

	@staticmethod
	def forward(ctx, x, a, b, func, var):

		# Save variables for backward
		ctx.save_for_backward(x, a, b)
		ctx.func = func
		ctx.var = var

		# f values
		f = func.f(x, a, b)
		df_da = func.df_da(x, a, b)
		d2f_da2 = func.d2f_da2(x, a, b)

		# Em values
		dem_da = df_da.mean()
		d2em_da2 = d2f_da2.mean()

		# Ev values
		ev = (f ** 2).mean() - f.mean() ** 2 - var
		dev_da = 2 * ((f * df_da).mean() - f.mean() * dem_da)
		d2ev_da2 = 2 * ((df_da ** 2 + f * d2f_da2).mean() - dem_da ** 2 - f.mean() * d2em_da2)

		# Lv values
		dlv_da = 2 * ev * dev_da
		d2lv_da2 = 2 * (dev_da ** 2 + ev * d2ev_da2)

		# Newton step
		na = dlv_da / d2lv_da2
		
		return na

	@staticmethod
	def backward(ctx, grad_output):
		
		# Read saved tensors
		x, a, b = ctx.saved_tensors

		# f values
		f = ctx.func.f(x, a, b)
		df_da = ctx.func.df_da(x, a, b)
		df_dx = ctx.func.df_dx(x, a, b)
		d2f_da2 = ctx.func.d2f_da2(x, a, b)
		d2f_dax = ctx.func.d2f_dax(x, a, b)
		d3f_da2x = ctx.func.d3f_da2x(x, a, b)
		N = x.numel()

		# Em values
		dem_da = df_da.mean()
		d2em_da2 = d2f_da2.mean()
		d2em_dax = d2f_dax / N
		d3em_da2x = d3f_da2x / N

		# Ev values
		ev = (f ** 2).mean() - f.mean() ** 2 - ctx.var
		dev_da = 2 * ((f * df_da).mean() - f.mean() * dem_da)
		dev_dx = 2 * df_dx * (f / N - f.mean())
		d2ev_da2 = 2 * ((df_da ** 2 + f * d2f_da2).mean() - dem_da ** 2 - f.mean() * d2em_da2)
		d2ev_dax = 2 * (df_dx * df_da + f * d2f_dax - \
		                (df_dx * dem_da - f.mean() * d2f_dax) / N)
		d3ev_da2x = 2 * ((2 * df_da * d2f_dax + df_dx * d2f_da2 + f * d3f_da2x) / N - \
		                 2 * dem_da * d2em_dax - f * d2em_da2 / N - f.mean() * d3em_da2x)

		# Lv values
		dlv_da = 2 * ev * dev_da
		d2lv_da2 = 2 * (dev_da ** 2 + ev * d2ev_da2)
		d2lv_dax = 2 * (dev_dx * dev_da + ev * d2ev_dax)
		d3lv_da2x = 2 * (2 * dev_da * d2ev_dax + dev_dx * d2ev_da2 + ev * d3ev_da2x)

		# Newton step
		dna_dx = (d2lv_da2 * d2lv_dax - dlv_da * d3lv_da2x) / d2lv_da2 ** 2

		return grad_output * dna_dx, None, None, None, None


class NSA(torch.nn.Module):
	"""docstring for NSA"""

	def __init__(self):
		super(NSA, self).__init__()
		self.nsa = Newton_Step_A.apply

	def forward(self, x, a, b, func, var):
		return self.nsa(x, a, b, func, var)


# QUICK TEST

# Initializations
nsa = NSA()
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
out = nsa(x = data,
          a = a,
          b = b,
          func = sig,
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