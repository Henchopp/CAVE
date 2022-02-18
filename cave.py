import os

import numpy as np
import torch

from utils.cave_base_functions import Sigmoid, Softplus
from utils.gradient_step_a import GSA
from utils.gradient_step_b import GSB
from utils.gradient_step_ab import GSAB
from utils.newton_step_a import NSA
from utils.newton_step_b import NSB
from utils.newton_step_ab import NSAB


class CAVE(torch.nn.Module):
	"""docstring for CAVE"""

	def __init__(self, func, n_step_gd = 10, n_step_nm = 5, lr_gd = 1.0, lr_nm = 1.0,
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

	def opt_var(self, x, var, dim):
		return (var / x.var(unbiased = False, **dim)).sqrt() * x

	def opt_range(self, x, low, high):
		if self.func.low and self.func.high:
			rng = self.func.high - self.func.low
			return (high - low) * (self.func.fx(x) - self.func.low) / rng + low

	def opt_moments(self, x, mean, var, dim):
		return (var / x.var(unbiased = False, **dim)).sqrt() * (x - x.mean(**dim)) + mean


	# CAVE transforms
	def opt_grad_mean(self, x, a, b, low, high, mean, var, dim):
		db = self.gsb(x, a, b, self.func, mean, dim, self.lr_gd)
		return None, db

	def opt_grad_var(self, x, a, b, low, high, mean, var, dim):
		da = self.gsa(x, a, b, self.func, var, dim, self.lr_gd)
		return da, None

	def opt_grad_joint(self, x, a, b, low, high, mean, var, dim):
		da, db = self.gsab(x, a, b, self.func, mean, var, dim, self.lr_gd)
		return da, db

	def opt_newton_mean(self, x, a, b, low, high, mean, var, dim):
		db = self.nsb(x, a, b, self.func, mean, dim, self.lr_nm)
		return None, db

	def opt_newton_var(self, x, a, b, low, high, mean, var, dim):
		da = self.nsa(x, a, b, self.func, var, dim, self.lr_nm)
		return da, None

	def opt_newton_joint(self, x, a, b, low, high, mean, var, dim):
		da, db = self.nsab(x, a, b, self.func, mean, var, dim, self.lr_nm)
		return da, db

	def opt_cave(self, x, low, high, mean, var, sparse, dim):

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

		# Initialize a and b
		a = self.a_init
		b = self.b_init

		# Standard normalize input
		x = (x - x.mean(**dim)) / x.std(**dim)

		# Spread data if sparse output required
		if sparse:
			if mean > 0.5:
				x = (x - x.std(**dim) - 1) ** 3
				x = (x - x.mean(**dim)) / x.std(**dim)
				x = x + x.std(**dim)
			else:
				x = (x + x.std(**dim) + 1) ** 3
				x = (x - x.mean(**dim)) / x.std(**dim)
				x = x - x.std(**dim)


		# Gradient descent
		for _ in range(self.n_step_gd):
			da, db = func_gd(x, a, b, low, high, mean, var, dim)
			if da:
				a = a - da
			if db:
				b = b - db

			if self.log:
				self.log.append([a.item(), b.item()])

		# Newton's method
		for _ in range(self.n_step_nm):
			da, db = func_nm(x, a, b, low, high, mean, var, dim)
			if da:
				a = a - da
			if db:
				b = b - db

			if self.log:
				self.log.append([a.item(), b.item()])

		if self.log:
			np.savetxt(os.path.join(self.log_dir, 'ab.csv'),
			           np.array(self.log),
			           delimiter = ',')
			np.savetxt(os.path.join(self.log_dir, 'x.csv'),
			           x.view(-1, 1).detach().numpy())

		return self.func.fx(a * x + b)


	# Forward
	def forward(self, x, low = None, high = None, mean = None, var = None, sparse = False, dim = None):

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
		if (low or high) and (mean or var):
			return self.opt_cave(x, low, high, mean, var, sparse, dim)
		elif low and high:
			return self.opt_range(x, low, high)
		elif low:
			return self.opt_low(x, low)
		elif high:
			return self.opt_high(x, high)
		elif mean and var:
			return self.opt_moments(x, mean, var, dim)
		elif mean:
			return self.opt_mean(x, mean, dim)
		elif var:
			return self.opt_var(x, var, dim)
		return x


###