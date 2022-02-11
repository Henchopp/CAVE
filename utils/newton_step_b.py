import torch
from .cave_base_functions import Sigmoid


class Newton_Step_B(torch.autograd.Function):
	"""docstring for Newton_Step_B"""

	@staticmethod
	def forward(ctx, x, a, b, func, mean, var):

		# Save variables for backward
		ctx.save_for_backward(x, a, b)
		ctx.func = func
		ctx.mean = mean
		ctx.var = var

		# f values
		f = func.f(x, a, b)
		df_db = func.df_db(x, a, b)
		d2f_db2 = func.d2f_db2(x, a, b)

		# E values
		em = f.mean() - mean
		dem_db = df_db.mean()
		d2em_db2 = d2f_db2.mean()

		# L values
		dlm_db = 2 * em * dem_db
		d2lm_db2 = 2 * (dem_db ** 2 + em * d2em_db2)

		# Newton step b
		if var:

			nb = (dl_db * d2l_da2 - dl_da * d2l_dab) / (d2l_da2 * d2l_db2 - d2l_dab ** 2)

		else:
			nb = dlm_db / d2lm_db2

		return nb

	@staticmethod
	def backward(ctx, grad_output):

		# Read saved tensors
		x, a, b = ctx.saved_tensors

		# L values
		dl_db = None

		# Newton step b
		if ctx.var:
			pass
		else:
			dnb_dx = (d2l_db2 * d2l_dbx - dl_db * d3l_db2x)

		return grad_output


class NSB(torch.nn.Module):
	"""docstring for NSB"""

	def __init__(self):
		super(NSB, self).__init__()
		self.nsb = Newton_Step_B.apply

	def forward(self, x, a, b, func, mean, var = None):
		return self.gsa(x, a, b, func, mean, var)



# TEST
# nsb = Newton_Step_B.apply
# sig = Sigmoid()
# a = torch.ones(1)
# b = torch.zeros(1)
#
# data = torch.rand(1000)
# data = (data - data.mean()) / (data.std())
# print(sig.f(data, a, b).mean(), sig.f(data, a, b).var(unbiased = True))
# data.requires_grad = True
#
# out = nsb(data, a, b, sig, 0.01)
# b = b - out
# out.backward()
# with torch.no_grad():
# 	print(sig.f(data, a, b).mean(), sig.f(data, a, b).var(unbiased = True))



###
