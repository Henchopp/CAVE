import torch
if __name__ == '__main__':
	from cave_base_functions import Sigmoid, Softplus


class Gradient_Step_AB(torch.autograd.Function):
	"""docstring for Gradient_Step_AB"""
	
	@staticmethod
	def forward(ctx, x, x_copy, a, b, func, mean, var, dim, lr):
		
		# Save variables for backward
		ctx.save_for_backward(x, a, b)
		ctx.func = func
		ctx.mean = mean
		ctx.var = var
		ctx.dim = dim
		ctx.lr = lr

		return func.Gab(x, a, b, mean, var, dim, lr)

	@staticmethod
	def backward(ctx, grad_output1, grad_output2):

		# Read saved tensors
		x, a, b = ctx.saved_tensors
		dGa_dx, dGb_dx = ctx.func.dGab_dx(x, a, b, ctx.mean, ctx.var, ctx.dim, ctx.lr)
		return grad_output1 * dGa_dx, grad_output2 * dGb_dx, None, None, None, None, None, None, None
		

class GSAB(torch.nn.Module):
	"""docstring for GSAB"""

	def __init__(self):
		super(GSAB, self).__init__()
		self.gsab = Gradient_Step_AB.apply

	def forward(self, x, a, b, func, mean, var, dim, lr):
		return self.gsab(x, x, a, b, func, mean, var, dim, lr)
		

# QUICK TEST
if __name__ == '__main__':

	# Initializations
	gsab = GSAB()
	sig = Sigmoid()
	a = torch.ones(1)
	b = torch.zeros(1)
	mean = torch.rand(5,1)
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
	out = gsab(x = data,
	           a = a,
	           b = b,
	           func = sig,
	           mean = mean,
	           var = var,
	           dim = dim,
	           lr = lr)

	# Gradient descent step w.r.t. b
	a = a - out[0]
	b = b - out[1]

	# Calculate gradients w.r.t. data
	out = out[0].sum() + out[1].sum()
	out.backward()

	# After stats (should trend towards the specified mean and var)
	with torch.no_grad():
		ma = sig.f(data, a, b).mean(**dim)
		va = sig.f(data, a, b).var(unbiased = False, **dim)
		print(f'Mean after:\n{ma}\n',
		      f'Var after:\n{va}')


###