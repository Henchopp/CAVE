from abc import ABC, abstractmethod
import os

import numpy as np
import torch
from torch.nn.functional import softplus


class CAVE(torch.nn.Module):
	"""docstring for CAVE"""

	def __init__(self, func, n_step_gd = 10, n_step_nm = 20, lr_gd = 2.0, lr_nm = 1.0,
	             a_init = 1.0, b_init = 0.0):
		super(CAVE, self).__init__()

		# Initialize activation functions
		self.func = func
		if func is None:
			self.func = Sigmoid()

		# Initialize variables to save
		self.n_step_gd = n_step_gd
		self.n_step_nm = n_step_nm
		self.lr_gd = lr_gd
		self.lr_nm = lr_nm
		self.a_init = torch.Tensor([a_init]) if not isinstance(a_init, torch.Tensor) else a_init
		self.b_init = torch.Tensor([b_init]) if not isinstance(b_init, torch.Tensor) else b_init

		# Convert to parameters
		self.a_init = torch.nn.Parameter(self.a_init, requires_grad = False)
		self.b_init = torch.nn.Parameter(self.b_init, requires_grad = False)

		# Initialize gradient/newton steps
		self.gsa = GSA()
		self.gsb = GSB()
		self.gsab = GSAB()
		self.nsa = NSA()
		self.nsb = NSB()
		self.nsab = NSAB()


	# Basic transforms
	def opt_low(self, x, low):
		if self.func.low:
			return self.func.fx(x) - self.func.low + low
		elif self.func.high:
			return -1.0 * (self.func.fx(x) - self.func.high) + low

	def opt_high(self, x, high):
		if self.func.low:
			return -1.0 * (self.func.fx(x) - self.func.low) + high
		elif self.func.high:
			return self.func.fx(x) - self.func.high + high

	def opt_mean(self, x, mean, dim):
		return x - x.mean(**dim) + mean

	def opt_var(self, x, var, dim, unbiased):
		return (var / x.var(unbiased = unbiased, **dim)).sqrt() * x

	def opt_range(self, x, low, high):
		if self.func.low and self.func.high:
			rng = self.func.high - self.func.low
			return (high - low) * (self.func.fx(x) - self.func.low) / rng + low

	def opt_moments(self, x, mean, var, dim, unbiased):
		return (var / x.var(unbiased = unbiased, **dim)).sqrt() * (x - x.mean(**dim)) + mean


	# CAVE transforms
	def opt_grad_mean(self, x, a, b, low, high, mean, var, dim, unbiased):
		db = self.gsb(x, a, b, self.func, mean, dim, self.lr_gd)
		return None, db

	def opt_grad_var(self, x, a, b, low, high, mean, var, dim, unbiased):
		da = self.gsa(x, a, b, self.func, var, dim, unbiased, self.lr_gd)
		return da, None

	def opt_grad_joint(self, x, a, b, low, high, mean, var, dim, unbiased):
		da, db = self.gsab(x, a, b, self.func, mean, var, dim, unbiased, self.lr_gd)
		return da, db

	def opt_newton_mean(self, x, a, b, low, high, mean, var, dim, unbiased):
		db = self.nsb(x, a, b, self.func, mean, dim, self.lr_nm)
		return None, db

	def opt_newton_var(self, x, a, b, low, high, mean, var, dim, unbiased):
		da = self.nsa(x, a, b, self.func, var, dim, unbiased, self.lr_nm)
		return da, None

	def opt_newton_joint(self, x, a, b, low, high, mean, var, dim, unbiased):
		da, db = self.nsab(x, a, b, self.func, mean, var, dim, unbiased, self.lr_nm)
		return da, db

	def opt_cave(self, x, low, high, mean, var, sparse, dim, unbiased):

		# Select optimization methods
		if mean and var:
			func_gd = self.opt_grad_joint
			func_nm = self.opt_newton_joint
		elif mean:
			func_gd = self.opt_grad_mean
			func_nm = self.opt_newton_mean
		elif var:
			func_gd = self.opt_grad_var
			func_nm = self.opt_newton_var

		# Preprocess mean and var
		if not (self.func.low == low and self.func.high == high):
			if self.func.low != None and self.func.high != None and low != None and high != None:
				amp_ratio = (self.func.high - self.func.low) / (high - low)
				if mean != None:
					mean = amp_ratio * mean + self.func.low - amp_ratio * low
				if var != None:
					var = var * amp_ratio ** 2

			elif self.func.low != None and (low != None or high != None):
				if low != None and high != None:
					raise ValueError(f"Cannot specify low and high when "
					                 f"{type(self.func).__name__}.high == None")
				elif mean != None and low != None:
					mean = mean + self.func.low - low
				elif mean != None and high != None:
					mean = -1.0 * (mean - high) + self.func.low

			elif self.func.high != None and high != None:
				if low != None and high != None:
					raise ValueError(f"Cannot specify low and high when "
					                 f"{type(self.func).__name__}.low == None")
				elif mean != None and low != None:
					mean = -1.0 * (mean - low) + self.func.high
				elif mean != None and high != None:
					mean = mean + self.func.high - high


		# Initialize a and b
		a = self.a_init
		b = self.b_init

		# Standard normalize input
		x = (x - x.mean(**dim)) / x.std(**dim)

		# Spread data if sparse output required
		if sparse:
			if mean is not None and mean > 0.5 * (self.func.low + self.func.high):
				x = (x - x.std(**dim) - 1) ** 3
				x = (x - x.mean(**dim)) / x.std(**dim)
				x = x + x.std(**dim)
			elif mean is not None and mean <= 0.5 * (self.func.low + self.func.high):
				x = (x + x.std(**dim) + 1) ** 3
				x = (x - x.mean(**dim)) / x.std(**dim)
				x = x - x.std(**dim)
			elif var is not None:
				x = x * 10


		# Gradient descent
		for _ in range(self.n_step_gd):
			da, db = func_gd(x, a, b, low, high, mean, var, dim, unbiased)
			if da is not None:
				a = a - da
			if db is not None:
				b = b - db

		# Newton's method
		for _ in range(self.n_step_nm):
			da, db = func_nm(x, a, b, low, high, mean, var, dim, unbiased)
			if da is not None:
				a = a - da
			if db is not None:
				b = b - db

		# Postprocess mean and var
		if not (self.func.low == low and self.func.high == high):
			if self.func.low != None and self.func.high != None and low != None and high != None:
				amp_ratio = (self.func.high - self.func.low) / (high - low)
				return (self.func.fx(a * x + b) - self.func.low + amp_ratio * low) / amp_ratio

			elif self.func.low != None and (low != None or high != None):
				if low != None:
					return self.func.fx(a * x + b) - self.func.low + low
				elif high != None:
					return -1.0 * (self.func.fx(a * x + b) - self.func.low) + high

			elif self.func.high != None and high != None:
				if low != None:
					return -1.0 * (self.func.fx(a * x + b) - self.func.high) + low
				elif high != None:
					return self.func.fx(a * x + b) - self.func.high + high

		return self.func.fx(a * x + b)


	# Forward
	def forward(self, x, low = None, high = None, mean = None, var = None,
	            sparse = False, dim = None, unbiased = True):

		# Dimension processing
		if dim is None:
			dim = [i for i in range(x.ndim)]
		elif isinstance(dim, int):
			dim = [dim]
		dim = {'dim': dim, 'keepdim': True}

		# Select CAVE method
		if (low != None or high != None) and (mean != None or var != None):
			return self.opt_cave(x, low, high, mean, var, sparse, dim, unbiased)
		elif low != None and high != None:
			return self.opt_range(x, low, high)
		elif low != None:
			return self.opt_low(x, low)
		elif high != None:
			return self.opt_high(x, high)
		elif mean != None and var != None:
			return self.opt_moments(x, mean, var, dim, unbiased)
		elif mean != None:
			return self.opt_mean(x, mean, dim)
		elif var != None:
			return self.opt_var(x, var, dim, unbiased)
		return x


##########################
#   CAVE BASE FUNCTION   #
##########################

class CAVEBaseFunction(ABC):
	"""docstring for CAVEBaseFunction"""

	def __init__(self, low = None, high = None):
		self.low = low
		self.high = high

	# Get number of elements across dim
	@staticmethod
	def numel(x_shape, dim):

		N = 1
		for i in dim['dim']:
			N *= x_shape[i]
		return N


	##################################################
	#   USER-IMPLEMENTED FUNCTIONS AND DERIVATIVES   #
	##################################################

	@abstractmethod
	def fx(self, x):
		raise NotImplementedError("You must implement the function for your custom "
		                          "CAVEBaseFunction.")

	@abstractmethod
	def dfx(self, x):
		raise NotImplementedError("You must implement the first derivative for your custom "
		                          "CAVEBaseFunction.")

	@abstractmethod
	def d2fx(self, x):
		raise NotImplementedError("You must implement the second derivative for your custom "
		                          "CAVEBaseFunction.")

	@abstractmethod
	def d3fx(self, x):
		raise NotImplementedError("You must implement the third derivative for your custom "
		                          "CAVEBaseFunction.")


	##################################################
	#   GRADIENT DESCENT FUNCTIONS AND DERIVATIVES   #
	##################################################

	# Forward methods
	def Ga(self, x, a, b, var, dim, lr):
		f = self.fx(a * x + b)
		df_da = self.dfx(a * x + b) * x

		dEv_da = 2 * ((f * df_da).mean(**dim) - f.mean(**dim) * df_da.mean(**dim))

		return lr * 2 * (f.var(unbiased = False, **dim) - var) * dEv_da

	def Gb(self, x, a, b, mean, dim, lr):
		return lr * 2 * (self.fx(a * x + b).mean(**dim) - mean) * self.dfx(a * x + b).mean(**dim)

	def Gab(self, x, a, b, mean, var, dim, lr):
		f = self.fx(a * x + b)
		df_db = self.dfx(a * x + b)
		fmean = f.mean(**dim)

		df_da = df_db * x

		Em = fmean - mean
		dEm_da = df_da.mean(**dim)
		dEm_db = df_db.mean(**dim)

		Ev = f.var(unbiased = False, **dim) - var

		dL_da = 2 * (Em * dEm_da + Ev * 2 * ((f * df_da).mean(**dim) - fmean * dEm_da))
		dL_db = 2 * (Em * dEm_db + Ev * 2 * ((f * df_db).mean(**dim) - fmean * dEm_db))

		return lr * dL_da, lr * dL_db

	# Backward methods
	def dGa_dx(self, x, a, b, var, dim, lr):
		N = self.numel(x.shape, dim)

		f = self.fx(a * x + b)
		dfx = self.dfx(a * x + b)
		fmean = f.mean(**dim)

		df_da = dfx * x
		df_dx = dfx * a
		d2f_dax = self.d2fx(a * x + b) * a * x + dfx
		dEm_da = df_da.mean(**dim)

		Ev = f.var(unbiased = False, **dim) - var
		dEv_da = 2 * ((f * df_da).mean(**dim) - fmean * dEm_da)
		dEv_dx = 2 * df_dx / N * (f - fmean)
		d2Ev_dax = 2 / N * ((df_da * df_dx + f * d2f_dax) - \
		                    df_dx * dEm_da - fmean * d2f_dax)

		d2Lv_dax = 2 * (dEv_da * dEv_dx + Ev * d2Ev_dax)
		
		return lr * d2Lv_dax

	def dGb_dx(self, x, a, b, mean, dim, lr):
		N = self.numel(x.shape, dim)

		df_db = self.dfx(a * x + b)

		Em = self.fx(a * x + b).mean(**dim) - mean
		dEm_db = df_db.mean(**dim)
		dEm_dx = df_db * a / N
		d2Em_dbx = self.d2fx(a * x + b) * a / N

		d2Lm_dbx = 2 * (dEm_db * dEm_dx + Em * d2Em_dbx)

		return lr * d2Lm_dbx

	def dGab_dx(self, x, a, b, mean, var, dim, lr):
		N = self.numel(x.shape, dim)

		f = self.fx(a * x + b)
		df_db = self.dfx(a * x + b)
		d2f_dbx = self.d2fx(a * x + b) * a
		fmean = f.mean(**dim)

		df_da = df_db * x
		df_dx = df_db * a
		d2f_dax = d2f_dbx * x + df_db

		Em = fmean - mean
		dEm_da = df_da.mean(**dim)
		dEm_db = df_db.mean(**dim)
		dEm_dx = df_dx / N

		Ev = f.var(unbiased = False, **dim) - var
		dEv_dx = 2 * df_dx / N * (f - fmean)
		d2Ev_dax = 2 / N * ((df_da * df_dx + f * d2f_dax) - \
		                    df_dx * dEm_da - fmean * d2f_dax)
		d2Ev_dbx = 2 / N * ((df_db * df_dx + f * d2f_dbx) - \
		                    df_dx * dEm_db - fmean * d2f_dbx)

		dGa_dx = 2 * (dEm_da * dEm_dx + Em * d2f_dax / N + \
		              2 * ((f * df_da).mean(**dim) - fmean * dEm_da) * dEv_dx + Ev * d2Ev_dax)
		dGb_dx = 2 * (dEm_db * dEm_dx + Em * d2f_dbx / N + \
		              2 * ((f * df_db).mean(**dim) - fmean * dEm_db) * dEv_dx + Ev * d2Ev_dbx)

		return lr * dGa_dx, lr * dGb_dx


	#################################################
	#   NEWTON'S METHOD FUNCTIONS AND DERIVATIVES   #
	#################################################

	# Forward methods
	def Na(self, x, a, b, var, dim, lr):
		f = self.fx(a * x + b)
		df_da = self.dfx(a * x + b) * x
		d2f_da2 = self.d2fx(a * x + b) * x ** 2

		fmean = f.mean(**dim)
		dEm_da = df_da.mean(**dim)

		Ev = f.var(unbiased = False, **dim) - var
		dEv_da = 2 * ((f * df_da).mean(**dim) - fmean * dEm_da)
		d2Ev_da2 = 2 * ((df_da ** 2 + f * d2f_da2).mean(**dim) - \
		                dEm_da ** 2 - fmean * d2f_da2.mean(**dim))

		dLv_da = 2 * Ev * dEv_da
		d2Lv_da2 = 2 * (dEv_da ** 2 + Ev * d2Ev_da2) + 1e-30

		return lr * dLv_da / d2Lv_da2

	def Nb(self, x, a, b, mean, dim, lr):
		Em = self.fx(a * x + b).mean(**dim) - mean
		dEm_db = self.dfx(a * x + b).mean(**dim)

		dLm_db = 2 * Em * dEm_db
		d2Lm_db2 = 2 * (dEm_db ** 2 + Em * self.d2fx(a * x + b).mean(**dim)) + 1e-30

		return lr * dLm_db / d2Lm_db2

	def Nab(self, x, a, b, mean, var, dim, lr):
		f = self.fx(a * x + b)
		df_db = self.dfx(a * x + b)
		d2f_db2 = self.d2fx(a * x + b)

		fmean = f.mean(**dim)

		df_da = df_db * x
		d2f_da2 = d2f_db2 * x ** 2
		d2f_dab = d2f_db2 * x

		Em = fmean - mean
		dEm_da = df_da.mean(**dim)
		dEm_db = df_db.mean(**dim)
		d2Em_da2 = d2f_da2.mean(**dim)
		d2Em_dab = d2f_dab.mean(**dim)
		d2Em_db2 = d2f_db2.mean(**dim)

		Ev = f.var(unbiased = False, **dim) - var
		dEv_da = 2 * ((f * df_da).mean(**dim) - fmean * dEm_da)
		dEv_db = 2 * ((f * df_db).mean(**dim) - fmean * dEm_db)
		d2Ev_da2 = 2 * ((df_da ** 2 + f * d2f_da2).mean(**dim) - \
		                dEm_da ** 2 - fmean * d2Em_da2)
		d2Ev_dab = 2 * ((df_da * df_db + f * d2f_dab).mean(**dim) - \
		                dEm_da * dEm_db - fmean * d2Em_dab)
		d2Ev_db2 = 2 * ((df_db ** 2 + f * d2f_db2).mean(**dim) - \
		                dEm_db ** 2 - fmean * d2Em_db2)

		dL_da = 2 * (Em * dEm_da + Ev * dEv_da)
		dL_db = 2 * (Em * dEm_db + Ev * dEv_db)
		d2L_da2 = 2 * (dEm_da ** 2 + Em * d2Em_da2 + dEv_da ** 2 + Ev * d2Ev_da2)
		d2L_dab = 2 * (dEm_da * dEm_db + Em * d2Em_dab + dEv_da * dEv_db + Ev * d2Ev_dab)
		d2L_db2 = 2 * (dEm_db ** 2 + Em * d2Em_db2 + dEv_db ** 2 + Ev * d2Ev_db2)

		Na = dL_da * d2L_db2 - dL_db * d2L_dab
		Nb = dL_db * d2L_da2 - dL_da * d2L_dab
		D = d2L_da2 * d2L_db2 - d2L_dab ** 2 + 1e-30

		return lr * Na / D, lr * Nb / D

	# Backward methods
	def dNa_dx(self, x, a, b, var, dim, lr):
		N = self.numel(x.shape, dim)

		f = self.fx(a * x + b)
		dfx = self.dfx(a * x + b)
		d2fx = self.d2fx(a * x + b)

		fmean = f.mean(**dim)

		df_da = dfx * x
		df_dx = dfx * a
		d2f_da2 = d2fx * x ** 2
		d2f_dax = d2fx * a * x + dfx
		d3f_da2x = self.d3fx(a * x + b) * a * (x ** 2) + 2 * x * d2fx

		dEm_da = df_da.mean(**dim)
		d2Em_da2 = d2f_da2.mean(**dim)

		Ev = f.var(unbiased = False, **dim) - var
		dEv_da = 2 * ((f * df_da).mean(**dim) - fmean * dEm_da)
		dEv_dx = 2 * df_dx / N * (f - fmean)
		d2Ev_da2 = 2 * ((df_da ** 2 + f * d2f_da2).mean(**dim) - \
		                dEm_da ** 2 - fmean * d2Em_da2)
		d2Ev_dax = 2 / N * ((df_da * df_dx + f * d2f_dax) - \
		                    df_dx * dEm_da - fmean * d2f_dax)
		d3Ev_da2x = 2 / N * (2 * df_da * d2f_dax + df_dx * d2f_da2 + f * d3f_da2x - \
		                     2 * dEm_da * d2f_dax - df_dx * d2Em_da2 - \
		                     fmean * d3f_da2x)

		dLv_da = 2 * Ev * dEv_da
		d2Lv_da2 = 2 * (dEv_da ** 2 + Ev * d2Ev_da2)
		d2Lv_dax = 2 * (dEv_da * dEv_dx + Ev * d2Ev_dax)
		d3Lv_da2x = 2 * (2 * dEv_da * d2Ev_dax + dEv_dx * d2Ev_da2 + Ev * d3Ev_da2x)

		return lr * (d2Lv_da2 * d2Lv_dax - dLv_da * d3Lv_da2x) / (d2Lv_da2 ** 2 + 1e-30)

	def dNb_dx(self, x, a, b, mean, dim, lr):
		N = self.numel(x.shape, dim)

		df_db = self.dfx(a * x + b)
		d2f_db2 = self.d2fx(a * x + b)

		Em = self.fx(a * x + b).mean(**dim) - mean
		dEm_db = df_db.mean(**dim)
		dEm_dx = df_db * a / N
		d2Em_db2 = d2f_db2.mean(**dim)
		d2Em_dbx = d2f_db2 * a / N
		d3Em_db2x = self.d3fx(a * x + b) * a / N

		dLm_db = 2 * Em * dEm_db
		d2Lm_db2 = 2 * (dEm_db ** 2 + Em * d2Em_db2)
		d2Lm_dbx = 2 * (dEm_db * dEm_dx + Em * d2Em_dbx)
		d3Lm_db2x = 2 * (2 * dEm_db * d2Em_dbx + dEm_dx * d2Em_db2 + Em * d3Em_db2x)

		return lr * (d2Lm_db2 * d2Lm_dbx - dLm_db * d3Lm_db2x) / (d2Lm_db2 ** 2 + 1e-30)

	def dNab_dx(self, x, a, b, mean, var, dim, lr):
		N = self.numel(x.shape, dim)

		f = self.fx(a * x + b)
		df_db = self.dfx(a * x + b)
		d2f_db2 = self.d2fx(a * x + b)
		d3f_db2x = self.d3fx(a * x + b) * a

		fmean = f.mean(**dim)

		df_da = df_db * x
		df_dx = df_db * a
		d2f_da2 = d2f_db2 * x ** 2
		d2f_dab = d2f_db2 * x
		d2f_dax = d2f_db2 * a * x + df_db
		d2f_dbx = d2f_db2 * a
		d3f_da2x = d3f_db2x * (x ** 2) + 2 * x * d2f_db2
		d3f_dabx = d3f_db2x * x + d2f_db2

		Em = fmean - mean
		dEm_da = df_da.mean(**dim)
		dEm_db = df_db.mean(**dim)
		dEm_dx = df_dx / N
		d2Em_da2 = d2f_da2.mean(**dim)
		d2Em_dab = d2f_dab.mean(**dim)
		d2Em_db2 = d2f_db2.mean(**dim)
		d2Em_dax = d2f_dax / N
		d2Em_dbx = d2f_dbx / N
		d3Em_da2x = d3f_da2x / N
		d3Em_dabx = d3f_dabx / N
		d3Em_db2x = d3f_db2x / N

		Ev = f.var(unbiased = False, **dim) - var
		dEv_da = 2 * ((f * df_da).mean(**dim) - fmean * dEm_da)
		dEv_db = 2 * ((f * df_db).mean(**dim) - fmean * dEm_db)
		dEv_dx = 2 * df_dx / N * (f - fmean)
		d2Ev_da2 = 2 * ((df_da ** 2 + f * d2f_da2).mean(**dim) - \
		                dEm_da ** 2 - fmean * d2Em_da2)
		d2Ev_dab = 2 * ((df_da * df_db + f * d2f_dab).mean(**dim) - \
		                dEm_da * dEm_db - \
		                fmean * d2Em_dab)
		d2Ev_db2 = 2 * ((df_db ** 2 + f * d2f_db2).mean(**dim) - \
		                dEm_db ** 2 - fmean * d2Em_db2)
		d2Ev_dax = 2 / N * ((df_da * df_dx + f * d2f_dax) - \
		                    df_dx * dEm_da - fmean * d2f_dax)
		d2Ev_dbx = 2 / N * ((df_db * df_dx + f * d2f_dbx) - \
		                    df_dx * dEm_db - fmean * d2f_dbx)
		d3Ev_da2x = 2 / N * (2 * df_da * d2f_dax + df_dx * d2f_da2 + f * d3f_da2x - \
		                     2 * dEm_da * d2f_dax - df_dx * d2Em_da2 - \
		                     fmean * d3f_da2x)
		d3Ev_dabx = 2 / N * (d2f_dax * df_db + d2f_dbx * df_da + df_dx * d2f_dab + f * d3f_dabx - \
		                     d2f_dax * dEm_db - d2f_dbx * dEm_da - \
		                     df_dx * d2Em_dab - d3f_dabx * fmean)
		d3Ev_db2x = 2 / N * (2 * df_db * d2f_dbx + df_dx * d2f_db2 + f * d3f_db2x - \
		                     2 * dEm_db * d2f_dbx - df_dx * d2Em_db2 - \
		                     fmean * d3f_db2x)

		dL_da = 2 * (Em * dEm_da + Ev * dEv_da)
		dL_db = 2 * (Em * dEm_db + Ev * dEv_db)
		d2L_da2 = 2 * (dEm_da ** 2 + Em * d2Em_da2 + dEv_da ** 2 + Ev * d2Ev_da2)
		d2L_dab = 2 * (dEm_da * dEm_db + Em * d2Em_dab + dEv_da * dEv_db + Ev * d2Ev_dab)
		d2L_db2 = 2 * (dEm_db ** 2 + Em * d2Em_db2 + dEv_db ** 2 + Ev * d2Ev_db2)
		d2L_dax = 2 * (dEm_da * dEm_dx + Em * d2Em_dax + dEv_da * dEv_dx + Ev * d2Ev_dax)
		d2L_dbx = 2 * (dEm_db * dEm_dx + Em * d2Em_dbx + dEv_db * dEv_dx + Ev * d2Ev_dbx)
		d3L_da2x = 2 * (2 * dEm_da * d2Em_dax + dEm_dx * d2Em_da2 + Em * d3Em_da2x + \
		                2 * dEv_da * d2Ev_dax + dEv_dx * d2Ev_da2 + Ev * d3Ev_da2x)
		d3L_dabx = 2 * (dEm_da * d2Em_dbx + dEm_db * d2Em_dax + dEm_dx * d2Em_dab + Em * d3Em_dabx + \
		                dEv_da * d2Ev_dbx + dEv_db * d2Ev_dax + dEv_dx * d2Ev_dab + Ev * d3Ev_dabx)
		d3L_db2x = 2 * (2 * dEm_db * d2Em_dbx + dEm_dx * d2Em_db2 + Em * d3Em_db2x + \
		                2 * dEv_db * d2Ev_dbx + dEv_dx * d2Ev_db2 + Ev * d3Ev_db2x)

		Na = dL_da * d2L_db2 - dL_db * d2L_dab
		Nb = dL_db * d2L_da2 - dL_da * d2L_dab
		D = d2L_da2 * d2L_db2 - d2L_dab ** 2

		dNa_dx = d2L_dax * d2L_db2 + dL_da * d3L_db2x - d2L_dbx * d2L_dab - dL_db * d3L_dabx
		dNb_dx = d2L_dbx * d2L_da2 + dL_db * d3L_da2x - d2L_dax * d2L_dab - dL_da * d3L_dabx
		dD_dx = d3L_da2x * d2L_db2 + d3L_db2x * d2L_da2 - 2 * d2L_dab * d3L_dabx

		dNa_dx = (D * dNa_dx - Na * dD_dx) / (D ** 2 + 1e-30)
		dNb_dx = (D * dNb_dx - Nb * dD_dx) / (D ** 2 + 1e-30)

		return lr * dNa_dx, lr * dNb_dx


################################
#   PRE-CODED BASE FUNCTIONS   #
################################

# Sigmoid function
class Sigmoid(CAVEBaseFunction):
	"""docstring for Sigmoid"""

	def __init__(self):
		super().__init__(low = 0.0, high = 1.0)

	def fx(self, x):
		return x.sigmoid()

	def dfx(self, x):
		sig = x.sigmoid()
		return sig * (1 - sig)

	def d2fx(self, x):
		sig = x.sigmoid()
		return sig * (1 - sig) * (1 - 2 * sig)

	def d3fx(self, x):
		sig = x.sigmoid()
		return sig * (1 - sig) * (6 * sig * (sig - 1) + 1)


# Softplus function
class Softplus(CAVEBaseFunction):
	"""docstring for Softplus"""

	def __init__(self):
		super().__init__(low = 0.0, high = None)

	def fx(self, x):
		return softplus(x)

	def dfx(self, x):
		return x.sigmoid()

	def d2fx(self, x):
		sig = x.sigmoid()
		return sig * (1 - sig)

	def d3fx(self, x):
		sig = x.sigmoid()
		return sig * (1 - sig) * (1 - 2 * sig)


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
		return grad * dGa_dx, None, None, None, None, None, None, None


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
		return grad1 * dGa_dx, grad2 * dGb_dx, None, None, None, None, None, None, None, None
		

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
		return grad * dNa_dx, None, None, None, None, None, None, None


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
		return grad1 * dNa_dx, grad2 * dNb_dx, None, None, None, None, None, None, None, None


class NSAB(torch.nn.Module):
	"""docstring for NSAB"""

	def __init__(self):
		super(NSAB, self).__init__()
		self.nsab = Newton_Step_AB.apply

	def forward(self, x, a, b, func, mean, var, dim, unbiased, lr):
		return self.nsab(x, x, a, b, func, mean, var, dim, unbiased, lr)


###