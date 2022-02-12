import torch
from cave_base_functions import Sigmoid


class Newton_Step_B(torch.autograd.Function):
	"""docstring for Newton_Step_B"""

	@staticmethod
	def forward(ctx, x, a, b, func, mean):

		# Save variables for backward
		ctx.save_for_backward(x, a, b)
		ctx.func = func
		ctx.mean = mean

		# f values
		f = func.f(x, a, b)
		df_db = func.df_db(x, a, b)
		d2f_db2 = func.d2f_db2(x, a, b)

		# Em values
		em = f.mean() - mean
		dem_db = df_db.mean()
		d2em_db2 = d2f_db2.mean()

		# Lm values
		dlm_db = 2 * em * dem_db
		d2lm_db2 = 2 * (dem_db ** 2 + em * d2em_db2)

		# Newton step
		nb = dlm_db / d2lm_db2

		return nb

	@staticmethod
	def backward(ctx, grad_output):

		# Read saved tensors
		x, a, b = ctx.saved_tensors

		# f values
		f = ctx.func.f(x, a, b)
		df_db = ctx.func.df_db(x, a, b)
		df_dx = ctx.func.df_dx(x, a, b)
		d2f_db2 = ctx.func.d2f_db2(x, a, b)
		d2f_dbx = ctx.func.d2f_dbx(x, a, b)
		d3f_db2x = ctx.func.d3f_db2x(x, a, b)
		N = x.numel()

		# Em values
		em = f.mean() - ctx.mean
		dem_db = df_db.mean()
		dem_dx = df_dx / N
		d2em_db2 = d2f_db2.mean()
		d2em_dbx = d2f_dbx / N
		d3em_db2x = d3f_db2x / N

		# Lm values
		dlm_db = 2 * em * dem_db
		d2lm_db2 = 2 * (dem_db ** 2 + em * d2em_db2)
		d2lm_dbx = 2 * (dem_dx * dem_db + em * d2em_dbx)
		d3lm_db2x = 2 * (2 * dem_db * d2em_dbx + dem_dx * d2em_db2 + em * d3em_db2x)

		# Newton step
		dnb_dx = (d2lm_db2 * d2lm_dbx - dlm_db * d3lm_db2x)

		return grad_output * dnb_dx, None, None, None, None


class NSB(torch.nn.Module):
	"""docstring for NSB"""

	def __init__(self):
		super(NSB, self).__init__()
		self.nsb = Newton_Step_B.apply

	def forward(self, x, a, b, func, mean):
		return self.nsb(x, a, b, func, mean)
	

# QUICK TEST

# Initializations
nsb = NSB()
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
out = nsb(x = data,
          a = a,
          b = b,
          func = sig,
          mean = mean)

# Gradient descent step w.r.t. a
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