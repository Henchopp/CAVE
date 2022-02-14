import torch
if __name__ == '__main__':
	from cave_base_functions import Sigmoid, Softplus


class Newton_Step_B(torch.autograd.Function):
	"""docstring for Newton_Step_B"""

	@staticmethod
	def forward(ctx, x, a, b, func, mean, dim, lr):

		# Save variables for backward
		ctx.save_for_backward(x, a, b)
		ctx.func = func
		ctx.mean = mean
		ctx.dim = dim
		ctx.lr = lr

		return func.Nb(x, a, b, mean, dim, lr)

	@staticmethod
	def backward(ctx, grad_output):

		# Read saved tensors
		x, a, b = ctx.saved_tensors
		dNb_dx = ctx.func.dNb_dx(x, a, b, ctx.mean, ctx.dim, ctx.lr)
		return grad_output * dNb_dx, None, None, None, None, None, None


class NSB(torch.nn.Module):
	"""docstring for NSB"""

	def __init__(self):
		super(NSB, self).__init__()
		self.nsb = Newton_Step_B.apply

	def forward(self, x, a, b, func, mean, dim, lr):
		return self.nsb(x, a, b, func, mean, dim, lr)
	

# QUICK TEST
if __name__ == '__main__':

	# Initializations
	nsb = NSB()
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
	out = nsb(x = data,
	          a = a,
	          b = b,
	          func = sig,
	          mean = mean,
	          dim = dim,
	          lr = lr)

	# Gradient descent step w.r.t. b
	b = b - out

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