import torch
from cave_base_functions import Sigmoid, Softplus


class Newton_Step_AB(torch.autograd.Function):
	"""docstring for Newton_Step_AB"""

	@staticmethod
	def forward(ctx, x, x_copy, a, b, func, mean, var):

		# Save variables for backward
		ctx.save_for_backward(x, a, b)
		ctx.func = func
		ctx.mean = mean
		ctx.var = var

		# f values
		f = func.f(x, a, b)
		df_da = func.df_da(x, a, b)
		df_db = func.df_db(x, a, b)
		d2f_da2 = func.d2f_da2(x, a, b)
		d2f_dab = func.d2f_dab(x, a, b)
		d2f_db2 = func.d2f_db2(x, a, b)

		# Em values
		em = f.mean() - mean
		dem_da = df_da.mean()
		dem_db = df_db.mean()
		d2em_da2 = d2f_da2.mean()
		d2em_dab = d2f_dab.mean()
		d2em_db2 = d2f_db2.mean()

		# Ev values
		ev = (f ** 2).mean() - f.mean() ** 2 - var
		dev_da = 2 * ((f * df_da).mean() - f.mean() * dem_da)
		dev_db = 2 * ((f * df_db).mean() - f.mean() * dem_db)
		d2ev_da2 = 2 * ((df_da ** 2 + f * d2f_da2).mean() - dem_da ** 2 - f.mean() * d2em_da2)
		d2ev_dab = 2 * ((df_da * df_db + f * d2f_dab).mean() - dem_da * dem_db - f.mean() * d2em_dab)
		d2ev_db2 = 2 * ((df_db ** 2 + f * d2f_db2).mean() - dem_db ** 2 - f.mean() * d2em_db2)

		# L values
		dl_da = 2 * (em * dem_da + ev * dev_da)
		dl_db = 2 * (em * dem_db + ev * dev_db)
		d2l_da2 = 2 * (dem_da ** 2 + em * d2em_da2 + dev_da ** 2 + ev * d2ev_da2)
		d2l_dab = 2 * (dem_da * dem_db + em * d2em_dab + dev_da * dev_db + ev * d2ev_dab)
		d2l_db2 = 2 * (dem_db ** 2 + em * d2em_db2 + dev_db ** 2 + ev * d2ev_db2)
		
		# Newton steps
		na = (dl_da * d2l_db2 - dl_db * d2l_dab) / (d2l_da2 * d2l_db2 - d2l_dab ** 2)
		nb = (dl_db * d2l_da2 - dl_da * d2l_dab) / (d2l_da2 * d2l_db2 - d2l_dab ** 2)

		return na, nb

	@staticmethod
	def backward(ctx, grad_output1, grad_output2):
		
		# Read saved tensors
		x, a, b = ctx.saved_tensors

		# f values
		f = ctx.func.f(x, a, b)
		df_da = ctx.func.df_da(x, a, b)
		df_db = ctx.func.df_db(x, a, b)
		df_dx = ctx.func.df_dx(x, a, b)
		d2f_da2 = ctx.func.d2f_da2(x, a, b)
		d2f_dab = ctx.func.d2f_dab(x, a, b)
		d2f_db2 = ctx.func.d2f_db2(x, a, b)
		d2f_dax = ctx.func.d2f_dax(x, a, b)
		d2f_dbx = ctx.func.d2f_dbx(x, a, b)
		d3f_da2x = ctx.func.d3f_da2x(x, a, b)
		d3f_dabx = ctx.func.d3f_dabx(x, a, b)
		d3f_db2x = ctx.func.d3f_db2x(x, a, b)
		N = x.numel()

		# Em values
		em = f.mean() - ctx.mean
		dem_da = df_da.mean()
		dem_db = df_db.mean()
		dem_dx = df_dx / N
		d2em_da2 = d2f_da2.mean()
		d2em_dab = d2f_dab.mean()
		d2em_db2 = d2f_db2.mean()
		d2em_dax = d2f_dax / N
		d2em_dbx = d2f_dbx / N
		d3em_da2x = d3f_da2x / N
		d3em_dabx = d3f_dabx / N
		d3em_db2x = d3f_db2x / N

		# Ev values
		ev = (f ** 2).mean() - f.mean() ** 2 - ctx.var
		dev_da = 2 * ((f * df_da).mean() - f.mean() * dem_da)
		dev_db = 2 * ((f * df_db).mean() - f.mean() * dem_db)
		dev_dx = 2 * df_dx * (f / N - f.mean())
		d2ev_da2 = 2 * ((df_da ** 2 + f * d2f_da2).mean() - dem_da ** 2 - f.mean() * d2em_da2)
		d2ev_dab = 2 * ((df_da * df_db + f * d2f_dab).mean() - dem_da * dem_db - f.mean() * d2em_dab)
		d2ev_db2 = 2 * ((df_db ** 2 + f * d2f_db2).mean() - dem_db ** 2 - f.mean() * d2em_db2)
		d2ev_dax = 2 * (df_dx * df_da + f * d2f_dax - \
		                (df_dx * df_da.mean() - f.mean() * d2f_dax) / N)
		d2ev_dbx = 2 * (df_dx * df_db + f * d2f_dbx - \
		                (df_dx * df_db.mean() - f.mean() * d2f_dbx) / N)
		d3ev_da2x = 2 * ((2 * df_da * d2f_dax + df_dx * d2f_da2 + f * d3f_da2x) / N - \
		                 2 * dem_da * d2em_dax - df_dx * d2em_da2 / N - f.mean() * d3em_da2x)
		d3ev_dabx = 2 * ((d2f_dax * df_db + df_da * d2f_dbx + df_dx * d2f_dab + f * d3f_dabx) / N - \
		                 d2em_dax * dem_db - dem_da * d2em_dbx - df_dx * d2em_dab / N - f.mean() * d3em_dabx)
		d3ev_db2x = 2 * ((2 * df_db * d2f_dbx + df_dx * d2f_db2 + f * d3f_db2x) / N - \
		                 2 * dem_db * d2em_dbx - df_dx * d2em_db2 / N - f.mean() * d3em_db2x)

		# L values
		dl_da = 2 * (em * dem_da + ev * dev_da)
		dl_db = 2 * (em * dem_db + ev * dev_db)
		d2l_da2 = 2 * (dem_da ** 2 + em * d2em_da2 + dev_da ** 2 + ev * d2ev_da2)
		d2l_dab = 2 * (dem_da * dem_db + em * d2em_dab + dev_da * dev_db + ev * d2ev_dab)
		d2l_db2 = 2 * (dem_db ** 2 + em * d2em_db2 + dev_db ** 2 + ev * d2ev_db2)
		d2l_dax = 2 * (dem_dx * dem_da + em * d2em_dax + dev_dx * dev_da + ev * d2ev_dax)
		d2l_dbx = 2 * (dem_dx * dem_db + em * d2em_dbx + dev_dx * dev_db + ev * d2ev_dbx)
		d3l_da2x = 2 * (2 * dem_da * d2em_dax + dem_dx * d2em_da2 + em * d3em_da2x + \
		                2 * dev_da * d2ev_dax + dev_dx * d2ev_da2 + ev * d3ev_da2x)
		d3l_dabx = 2 * (d2em_dax * dem_db + dem_da * d2em_dbx + dem_dx * d2em_dab + em * d3em_dabx + \
		                d2ev_dax * dev_db + dev_da * d2ev_dbx + dev_dx * d2ev_dab + ev * d3ev_dabx)
		d3l_db2x = 2 * (2 * dem_db * d2em_dbx + dem_dx * d2em_db2 + em * d3em_db2x + \
		                2 * dev_db * d2ev_dbx + dev_dx * d2ev_db2 + ev * d3ev_db2x)

		# Newton steps
		numa = dl_da * d2l_db2 - dl_db * d2l_dab
		numb = dl_db * d2l_da2 - dl_da * d2l_dab
		den = d2l_da2 * d2l_db2 - d2l_dab ** 2

		dnuma_dx = d2l_dax * d2l_db2 + dl_da * d3l_db2x - d2l_dbx * d2l_dab - dl_db * d3l_dabx
		dnumb_dx = d2l_dbx * d2l_da2 + dl_db * d3l_da2x - d2l_dax * d2l_dab - dl_da * d3l_dabx
		dden_dx = d3l_da2x * d2l_db2 + d2l_da2 * d3l_db2x - 2 * d2l_dab * d3l_dabx

		dna_dx = (den * dnuma_dx - numa * dden_dx) / (den ** 2)
		dnb_dx = (den * dnumb_dx - numb * dden_dx) / (den ** 2)

		return grad_output1 * dna_dx, grad_output2 * dnb_dx, None, None, None, None, None


class NSAB(torch.nn.Module):
	"""docstring for NSAB"""

	def __init__(self):
		super(NSAB, self).__init__()
		self.nsab = Newton_Step_AB.apply

	def forward(self, x, a, b, func, mean, var):
		return self.nsab(x, x, a, b, func, mean, var)


# QUICK TEST

# Initializations
nsab = NSAB()
sig = Sigmoid()
a = torch.ones(1)
b = torch.zeros(1)
mean = torch.Tensor([0.4])
var = torch.Tensor([0.05])
lr = 0.1 * torch.ones(1)

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
out = nsab(x = data,
           a = a,
           b = b,
           func = sig,
           mean = mean,
           var = var)

# Gradient descent step w.r.t. a
a = a - lr * out[0]
b = b - lr * out[1]

# Calculate gradients w.r.t. data
out = out[0] + out[1]
out.backward()

# After stats (should trend towards the specified mean and var)
with torch.no_grad():
	ma = sig.f(data, a, b).mean().item()
	va = sig.f(data, a, b).var(unbiased = False).item()
	print(f'Mean after:\t{ma}\t',
	      f'Var after:\t{va}')


###