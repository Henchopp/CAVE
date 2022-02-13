import torch
if __name__ == '__main__':
	from cave_base_functions import Sigmoid, Softplus


class Gradient_Step_B(torch.autograd.Function):
	"""docstring for Gradient_Step_B"""

	@staticmethod
	def forward(ctx, x, a, b, func, mean, dim):

		# Save variables for backward
		ctx.save_for_backward(x, a, b)
		ctx.func = func
		ctx.mean = mean
		ctx.dim = dim

		# f values
		f = func.f(x, a, b)
		df_db = func.df_db(x, a, b)

		# E values
		em = f.mean(**dim) - mean
		dem_db = df_db.mean(**dim)

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

		# Get N
		N = 1
		for i in ctx.dim['dim']:
			N *= x.shape[i]

		# E values
		em = f.mean(**ctx.dim) - ctx.mean
		dem_db = df_db.mean(**ctx.dim)
		dem_dx = df_dx / N
		d2em_dbx = d2f_dbx / N

		# L value
		d2lm_dbx = 2 * (dem_dx * dem_db + em * d2em_dbx)

		return grad_output * d2lm_dbx, None, None, None, None, None


class GSB(torch.nn.Module):
	"""docstring for GSB"""

	def __init__(self):
		super(GSB, self).__init__()
		self.gsb = Gradient_Step_B.apply

	def forward(self, x, a, b, func, mean, dim):
		return self.gsb(x, a, b, func, mean, dim)


# QUICK TEST
if __name__ == '__main__':

	# Initializations
	gsb = GSB()
	sig = Sigmoid()
	a = torch.ones(1)
	b = torch.zeros(1)
	mean = torch.rand(5,1)
	var = torch.Tensor([0.01])
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
	out = gsb(x = data,
	          a = a,
	          b = b,
	          func = sig,
	          mean = mean,
	          dim = dim)

	# Gradient descent step w.r.t. b
	b = b - lr * out

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