import torch
from .cave_base_functions import Sigmoid, Softplus


class Gradient_Step_B(torch.autograd.Function):
	"""docstring for Gradient_Step_B"""

	@staticmethod
	def forward(ctx, x, a, b, func, mean):

		# Save variables for backward
		ctx.save_for_backward(x, a, b)
		ctx.func = func
		ctx.mean = mean

		# f values
		f = func.f(x, a, b)
		df_db = func.df_db(x, a, b)

		# E values
		em = f.mean() - mean
		dem_db = df_db.mean()

		# L value
		dlm_db = 2 * em * dem_db

		return dlm_db

	@staticmethod
	def backward(ctx, grad_output):
		
		# Read saved tensors
		x, a, b = ctx.saved_tensors

		# f values
		f = ctx.func.f(x, a, b)
		df_db = ctx.func.df_db(x, a, b)
		df_dx = ctx.func.df_dx(x, a, b)
		d2f_dbx = ctx.func.d2f_dbx(x, a, b)
		N = x.numel()

		# E values
		em = f.mean() - ctx.mean
		dem_db = df_db.mean()
		dem_dx = df_dx / N
		d2em_dbx = d2f_dbx / N

		# L value
		d2lm_dbx = 2 * (dem_dx * dem_db + em * d2em_dbx)

		return grad_output * d2lm_dbx, None, None, None, None


class GSB(torch.nn.Module):
	"""docstring for GSB"""

	def __init__(self):
		super(GSB, self).__init__()
		self.gsb = Gradient_Step_B.apply

	def forward(self, x, a, b, func, mean):
		return self.gsb(x, a, b, func, mean)


# QUICK TEST

# Initializations
gsb = GSB()
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

# Calculate grad descent step w.r.t. b
out = gsb(x = data,
          a = a,
          b = b,
          func = sig,
          mean = mean,
          var = var)

# Gradient descent step w.r.t. b
b = b - lr * out

# Calculate gradients w.r.t. data
out.backward()

# After stats (should trend towards the specified mean and var)
with torch.no_grad():
	ma = sig.f(data, a, b).mean().item()
	va = sig.f(data, a, b).var(unbiased = False).item()
	print(f'Mean after:\t{ma}\t',
	      f'Var after:\t{va}')


###