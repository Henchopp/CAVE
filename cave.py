import os

import numpy as np
import torch

from CAVE.utils.cave_base_functions import Sigmoid, Softplus
from CAVE.utils.cave_steps import GSA, GSB, GSAB, NSA, NSB, NSAB


class CAVE(torch.nn.Module):
	"""docstring for CAVE"""

	def __init__(self, func, n_step_gd = 10, n_step_nm = 20, lr_gd = 2.0, lr_nm = 1.0,
	             a_init = 1.0, b_init = 0.0, output_log = False):
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

		# Output log
		self.log = output_log
		if output_log:
			self.log = []
			self.log_dir = os.path.join(os.path.dirname(os.getcwd()),
			                            'matlab',
			                            'output_logs',
			                            output_log)
			if not os.path.exists(self.log_dir):
				os.makedirs(self.log_dir)
			self.log.append([n_step_gd, n_step_nm])


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
			if mean and mean > 0.5:
				x = (x - x.std(**dim) - 1) ** 3
				x = (x - x.mean(**dim)) / x.std(**dim)
				x = x + x.std(**dim)
			elif mean and mean <= 0.5:
				x = (x + x.std(**dim) + 1) ** 3
				x = (x - x.mean(**dim)) / x.std(**dim)
				x = x - x.std(**dim)
			elif var:
				x = x * 10

		# Gradient descent
		for _ in range(self.n_step_gd):
			da, db = func_gd(x, a, b, low, high, mean, var, dim, unbiased)
			if da is not None:
				a = a - da
			if db is not None:
				b = b - db

			if self.log:
				self.log.append([a.item(), b.item()])

		# Newton's method
		for _ in range(self.n_step_nm):
			da, db = func_nm(x, a, b, low, high, mean, var, dim, unbiased)
			if da is not None:
				a = a - da
			if db is not None:
				b = b - db

			if self.log:
				self.log.append([a.item(), b.item()])

		if self.log:
			np.savetxt(os.path.join(self.log_dir, 'ab.csv'),
			           np.array(self.log),
			           delimiter = ',')
			np.savetxt(os.path.join(self.log_dir, 'x.csv'),
			           x.view(-1, 1).detach().numpy())

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
		print("FINAL", a * x, b)
		return self.func.fx(a * x + b)


	# Forward
	def forward(self, x, low = None, high = None, mean = None, var = None,
	            sparse = False, dim = None, unbiased = True):

		# Log input
		if isinstance(self.log, list):
			if mean is not None and var is not None:
				self.log.append([mean, var])
			elif mean is not None:
				self.log.append([mean, float('nan')])
			else:
				self.log.append([float('nan'), var])
			self.log.append([self.a_init.item(), self.b_init.item()])

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


###
