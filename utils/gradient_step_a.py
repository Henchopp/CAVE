import torch
if __name__ == '__main__':
	from cave_base_functions import Sigmoid, Softplus


class Gradient_Step_A(torch.autograd.Function):
	"""docstring for Gradient_Step_A"""
	
	@staticmethod
	def forward(ctx, x, a, b, func, var, dim, lr):

		# Save variables for backward
		ctx.save_for_backward(x, a, b)
		ctx.func = func
		ctx.var = var
		ctx.dim = dim
		ctx.lr = lr

		return func.Ga(x, a, b, var, dim, lr)

	@staticmethod
	def backward(ctx, grad_output):

		# Read saved tensors
		x, a, b = ctx.saved_tensors
		dGa_dx = ctx.func.dGa_dx(x, a, b, ctx.var, ctx.dim, ctx.lr)
		return grad_output * dGa_dx, None, None, None, None, None, None


class GSA(torch.nn.Module):
	"""docstring for GSA"""

	def __init__(self):
		super(GSA, self).__init__()
		self.gsa = Gradient_Step_A.apply

	def forward(self, x, a, b, func, var, dim, lr):
		return self.gsa(x, a, b, func, var, dim, lr)
		

# QUICK TEST
if __name__ == '__main__':

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
	          dim = dim,
	          lr = lr)

	# Gradient descent step w.r.t. b
	a = a - out

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