import torch
from .cave_base_functions import Sigmoid, SoftPlus


class Newton_Step_A(torch.autograd.Function):
	"""docstring for Newton_Step_A"""

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
		d2f_da2 = func.d2f_da2(x, a, b)

		# E values for variance
		dem_da = df_da.mean()
		d2em_da2 = d2f_da2.mean()

		ev = (f ** 2).mean() - f.mean() ** 2 - var
		dev_da = 2 * ((f * df_da).mean() - f.mean() * dem_da)
		d2ev_da2 = 2 * ((df_da ** 2 + f * d2f_da2).mean() - dem_da ** 2 - f.mean() * d2em_da2)

		# L values for variance
		dlv_da = 2 * ev * dev_da
		d2lv_da2 = 2 * (dev_da ** 2 + ev * d2ev_da2)

		# Newton step
		na = dlv_da / d2lv_da2
		
		return na

	@staticmethod
	def backward(ctx, grad_output):
		
		# Read saved tensors
		x, a, b = ctx.saved_tensors

		#
		dna_dx = d2lv_da2 * d2lv_dax - dl_da * d3l_da2x

		return grad_output * dna_dx


class NSA(torch.nn.Module):
	"""docstring for NSA"""

	def __init__(self):
		super(NSA, self).__init__()
		self.nsa = Newton_Step_A.apply

	def forward(self, x, a, b, func, var, mean = None):
		return self.gsa(x, a, b, func, mean, var)


###