import torch


############################################
#   GRADIENT DESCENT SINGLE STEP CLASSES   #
############################################

class Gradient_Step_A(torch.autograd.Function):
	"""docstring for Gradient_Step_A"""
	
	@staticmethod
	def forward(ctx, x, a, b, func, var, dim, unbiased, lr):

		# Adjust variance
		if unbiased:
			N = func.numel(x.shape, dim)
			var = var * (N - 1.0) / N

		# Save variables for backward
		ctx.save_for_backward(x, a, b)
		ctx.func = func
		ctx.var = var
		ctx.dim = dim
		ctx.lr = lr

		return func.Ga(x, a, b, var, dim, lr)

	@staticmethod
	def backward(ctx, grad):

		# Read saved tensors
		x, a, b = ctx.saved_tensors
		dGa_dx = ctx.func.dGa_dx(x, a, b, ctx.var, ctx.dim, ctx.lr)
		return grad * dGa_dx, None, None, None, None, None, None


class GSA(torch.nn.Module):
	"""docstring for GSA"""

	def __init__(self):
		super(GSA, self).__init__()
		self.gsa = Gradient_Step_A.apply

	def forward(self, x, a, b, func, var, dim, unbiased, lr):
		return self.gsa(x, a, b, func, var, dim, unbiased, lr)


class Gradient_Step_B(torch.autograd.Function):
	"""docstring for Gradient_Step_B"""

	@staticmethod
	def forward(ctx, x, a, b, func, mean, dim, lr):

		# Save variables for backward
		ctx.save_for_backward(x, a, b)
		ctx.func = func
		ctx.mean = mean
		ctx.dim = dim
		ctx.lr = lr
		
		return func.Gb(x, a, b, mean, dim, lr)

	@staticmethod
	def backward(ctx, grad):
		
		# Read saved tensors
		x, a, b = ctx.saved_tensors
		dGb_dx = ctx.func.dGb_dx(x, a, b, ctx.mean, ctx.dim, ctx.lr)
		return grad * dGb_dx, None, None, None, None, None, None


class GSB(torch.nn.Module):
	"""docstring for GSB"""

	def __init__(self):
		super(GSB, self).__init__()
		self.gsb = Gradient_Step_B.apply

	def forward(self, x, a, b, func, mean, dim, lr):
		return self.gsb(x, a, b, func, mean, dim, lr)


class Gradient_Step_AB(torch.autograd.Function):
	"""docstring for Gradient_Step_AB"""
	
	@staticmethod
	def forward(ctx, x, x_copy, a, b, func, mean, var, dim, unbiased, lr):

		# Adjust variance
		if unbiased:
			N = func.numel(x.shape, dim)
			var = var * (N - 1.0) / N
		
		# Save variables for backward
		ctx.save_for_backward(x, a, b)
		ctx.func = func
		ctx.mean = mean
		ctx.var = var
		ctx.dim = dim
		ctx.lr = lr

		return func.Gab(x, a, b, mean, var, dim, lr)

	@staticmethod
	def backward(ctx, grad1, grad2):

		# Read saved tensors
		x, a, b = ctx.saved_tensors
		dGa_dx, dGb_dx = ctx.func.dGab_dx(x, a, b, ctx.mean, ctx.var, ctx.dim, ctx.lr)
		return grad1 * dGa_dx, grad2 * dGb_dx, None, None, None, None, None, None, None
		

class GSAB(torch.nn.Module):
	"""docstring for GSAB"""

	def __init__(self):
		super(GSAB, self).__init__()
		self.gsab = Gradient_Step_AB.apply

	def forward(self, x, a, b, func, mean, var, dim, unbiased, lr):
		return self.gsab(x, x, a, b, func, mean, var, dim, unbiased, lr)


###########################################
#   NEWTON'S METHOD SINGLE STEP CLASSES   #
###########################################

class Newton_Step_A(torch.autograd.Function):
	"""docstring for Newton_Step_A"""

	@staticmethod
	def forward(ctx, x, a, b, func, var, dim, unbiased, lr):

		# Adjust variance
		if unbiased:
			N = func.numel(x.shape, dim)
			var = var * (N - 1.0) / N

		# Save variables for backward
		ctx.save_for_backward(x, a, b)
		ctx.func = func
		ctx.var = var
		ctx.dim = dim
		ctx.lr = lr
		
		return func.Na(x, a, b, var, dim, lr)

	@staticmethod
	def backward(ctx, grad):
		
		# Read saved tensors
		x, a, b = ctx.saved_tensors
		dNa_dx = ctx.func.dNa_dx(x, a, b, ctx.var, ctx.dim, ctx.lr)
		return grad * dNa_dx, None, None, None, None, None, None


class NSA(torch.nn.Module):
	"""docstring for NSA"""

	def __init__(self):
		super(NSA, self).__init__()
		self.nsa = Newton_Step_A.apply

	def forward(self, x, a, b, func, var, dim, unbiased, lr):
		return self.nsa(x, a, b, func, var, dim, unbiased, lr)


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
	def backward(ctx, grad):

		# Read saved tensors
		x, a, b = ctx.saved_tensors
		dNb_dx = ctx.func.dNb_dx(x, a, b, ctx.mean, ctx.dim, ctx.lr)
		return grad * dNb_dx, None, None, None, None, None, None


class NSB(torch.nn.Module):
	"""docstring for NSB"""

	def __init__(self):
		super(NSB, self).__init__()
		self.nsb = Newton_Step_B.apply

	def forward(self, x, a, b, func, mean, dim, lr):
		return self.nsb(x, a, b, func, mean, dim, lr)


class Newton_Step_AB(torch.autograd.Function):
	"""docstring for Newton_Step_AB"""

	@staticmethod
	def forward(ctx, x, x_copy, a, b, func, mean, var, dim, unbiased, lr):

		# Adjust variance
		if unbiased:
			N = func.numel(x.shape, dim)
			var = var * (N - 1.0) / N

		# Save variables for backward
		ctx.save_for_backward(x, a, b)
		ctx.func = func
		ctx.mean = mean
		ctx.var = var
		ctx.dim = dim
		ctx.lr = lr

		return func.Nab(x, a, b, mean, var, dim, lr)

	@staticmethod
	def backward(ctx, grad1, grad2):
		
		# Read saved tensors
		x, a, b = ctx.saved_tensors
		dNa_dx, dNb_dx = ctx.func.dNab_dx(x, a, b, ctx.mean, ctx.var, ctx.dim, ctx.lr)
		return grad1 * dNa_dx, grad2 * dNb_dx, None, None, None, None, None, None, None


class NSAB(torch.nn.Module):
	"""docstring for NSAB"""

	def __init__(self):
		super(NSAB, self).__init__()
		self.nsab = Newton_Step_AB.apply

	def forward(self, x, a, b, func, mean, var, dim, unbiased, lr):
		return self.nsab(x, x, a, b, func, mean, var, dim, unbiased, lr)


###