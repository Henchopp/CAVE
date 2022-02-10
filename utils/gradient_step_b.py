import torch
from cave_base_functions import Sigmoid, Softplus


class Gradient_Step_B(torch.autograd.Function):
	"""docstring for Gradient_Step_B"""

	@staticmethod
	def forward(ctx, x, a, b, func, mean, var = None):

		# Save variables for backward
		ctx.save_for_backward(x, a, b)
		ctx.func = func
		ctx.mean = mean
		ctx.var = var

		# f values
		f = func.f(x, a, b)
		df_db = func.df_db(x, a, b)

		# E values
		em = f.mean() - mean
		dem_db = df_db.mean()

		# L value
		dlm_db = 2 * em * dem_db

		# L value for joint optimization
		dlv_db = 0.0
		if var:

			# E values for variance
			ev = (f ** 2).mean() - f.mean() ** 2 - var
			dev_db = 2 * ((f * df_db).mean() - f.mean() * dem_db)
			
			# Update L with L value for variance
			dlv_db = 2 * ev * dev_db

		return dlm_db + dlv_db

	@staticmethod
	def backward(ctx, grad_output):
		
		# Read saved tensors
		x, a, b = ctx.saved_tensors

		# f values for mean
		f = ctx.func.f(x, a, b)
		df_db = ctx.func.df_db(x, a, b)
		df_dx = ctx.func.df_dx(x, a, b)
		d2f_dbx = ctx.func.d2f_dbx(x, a, b)
		N = x.numel()

		# E values for mean
		em = f.mean() - ctx.mean
		dem_db = df_db.mean()
		dem_dx = df_dx / N
		d2em_dbx = d2f_dbx / N

		# L value for mean
		d2lm_dbx = 2 * (dem_dx * dem_db + em * d2em_dbx)

		# L value for variance if needed
		if ctx.var:

			# E values for variance
			ev = (f ** 2).mean() - f.mean() ** 2 - ctx.var
			dev_db = 2 * ((f * df_db).mean() - f.mean() * dem_db)
			dev_dx = 2 * df_dx * (f / N - f.mean())
			d2ev_dbx = 2 * (df_dx * df_db + f * d2f_dbx - \
			                (df_dx * df_db.mean() - f.mean() * d2f_dbx) / N)

			# Update L with L value for variance
			d2lv_dbx = 2 * (dev_dx * dev_db + ev * d2ev_dbx)

			return grad_output * (d2lm_dbx + d2lv_dbx), None, None, None, None, None

		return grad_output * d2lm_dbx, None, None, None, None


# QUICK TEST
gsb = Gradient_Step_B.apply
sig = Softplus()
a = torch.ones(1)
b = torch.zeros(1)

data = torch.rand(1000)
data = (data - data.mean()) / (data.std())
print(sig.f(data, a, b).mean(), sig.f(data, a, b).var(unbiased = True))
data.requires_grad = True

out = gsb(data, a, b, sig, 0.01, 0.1)
b = b - out
out.backward()
with torch.no_grad():
	print(sig.f(data, a, b).mean(), sig.f(data, a, b).var(unbiased = True))


###