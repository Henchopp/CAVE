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

	def __init__(self, n_step_gd, n_step_nm, lr_gd, lr_nm,
	             adam = False, a_init = 1.0, b_init = 0.0):
		super(CAVE, self).__init__()

		# Initializing activation functions
		self.softplus = Softplus()
		self.sigmoid = Sigmoid()

		# Initializing variables to save
		self.a = a_init
		self.b = b_init
		self.n_step_gd = n_step_gd
		self.n_step_nm = n_step_nm
		self.lr_gd = lr_gd
		self.lr_nm = lr_nm
		self.adam = adam


	# Basic transforms
	def opt_none(self, x):
		return x

	def opt_low(self, x, low):
		return self.softplus.fx(x) + low

	def opt_high(self, x, high):
		return -self.softplus.fx(x) + high

	def opt_mean(self, x, mean, dim):
		return x - x.mean(**dim) + mean

	def opt_var(self, x, var, dim):
		return (var / x.var(unbiased = False, **dim)).sqrt() * x

	def opt_range(self, x, low, high):
		return (high - low) * self.sigmoid.fx(x) + low

	def opt_moments(self, x, mean, var, dim):
		return (var / x.var(unbiased = False, **dim)).sqrt() * (x - x.mean(**dim)) + mean


	# CAVE Processing
	def cave_preprocess(self, x, low, high, mean, var):
		return

	def cave_postprocess(self, x, low, high, mean, var):
		return


	# CAVE transforms
	def opt_grad_mean(self, x, params):

		gd_step = GSB()

		# performing gradient descent
		for i in range(self.n_step_gd):
			self.b = self.b - gd_step(x, self.a, self.b, self.softplus, params["mean"], dim, self.lr_gd)

	def opt_grad_var(self, x, params):
		gd_step = GSA()

		# performing newton's method
		for i in range(self.n_step_gd):
			self.a = self.a - gd_step(x, self.a, self.b, self.softplus, params["var"], dim, self.lr_gd)

	def opt_grad_joint(self, x, params):
		gd_step = GSAB()

		# performing newton's method
		for i in range(self.n_step_gd):
			a, b = gd_step(x, self.a, self.b, self.softplus, params["var"], dim, self.lr_gd)
			self.a = self.a - a
			self.b = self.b - b

	def opt_newton_mean(self, x, params):

		nm_step = NSB()

		# performing newton's method
		for i in range(self.n_step_nm):
			self.b = self.b - nm_step(x, self.a, self.b, self.softplus, parans["mean"], dim, self.lr_nm)

	def opt_newton_var(self, x, params):
		nm_step = NSA()

		# performing newton's method
		for i in range(self.n_step_nm):
			self.a = self.a - nm_step(x, self.a, self.b, self.softplus, params["var"], dim, self.lr_nm)

	def opt_newton_joint(self, x, params):
		nm_step = NSAB()

		# performing newton's method
		for i in range(self.n_step_nm):
			a, b = nm_step(x, self.a, self.b, self.softplus, params["var"], dim, self.lr_nm)
			self.a = self.a - a,
			self.b = self.b - b

	def opt_cave(self, x, low, high, mean, var):

		def compute_function(activ_var, low, high):

			if(high is not None and low is not None):
				return (high - low) * self.sigmoid(activ_var) + low
			elif(high is not None and low is None):
				return -self.softplus(activ_var) + high
			else:
				return self.softplus(activ_var) + low

		case = int("".join(
		str(1 if c == True else 0)
		for c in reversed([low != None, high != None, mean != None, var != None])
		), 2)

		# defining params
		params = { "mean": mean, "var": var, "low": low, "high": high }

		ret_func = None

		if(case >= 5 and case <= 7):
			# gradient descent
			self.opt_grad_mean(x, params)
			# newton's method
			self.opt_newton_mean(x, params)

			return compute_function(x + self.b, low, high)

		elif(case >= 9 and case <= 11):
			# gradient descent
			self.opt_grad_var(x, params)
			# newton's method
			self.opt_newton_var(x, params)

			return compute_function(self.a * x, low, high)
		elif(case >= 13 and case <= 15):
			# gradient descent
			self.opt_grad_joint(x, params)
			# newton's method
			self.opt_newton_joint(x, params)

			return compute_function(self.a * x + self.b, low, high)


	# Forward
	def forward(self, x, low = None, high = None, mean = None, var = None):

		case = int("".join(
		str(1 if c == True else 0)
		for c in reversed([low != None, high != None, mean != None, var != None])
		), 2)

		opt_fcns = {
			0: self.opt_none,
			1: self.opt_low,
			2: self.opt_high,
			3: self.opt_range,
			4: self.opt_mean,
			5: self.opt_cave, # mean and low,
			6: self.opt_cave, # mean and high,
			7: self.opt_cave, # mean and range,
			8: self.opt_var,
			9: self.opt_cave, # var and low,
			10: self.opt_cave, # var and high,
			11: self.opt_cave, # var and range,
			12: self.opt_moments,
			13: self.opt_cave, # moments and low,
			14: self.opt_cave, # moments and high,
			15: self.opt_cave, # everything
		}

		return opt_fcns[case]


'''
INPUT VARIABLES for CAVE

forward
-	low = None
-	high = None
-	mean = None
-	var = None

__init__
-	a_init = 1.0
-	b_init = 0.0
-	n_step_gd
-	n_step_nm
-	lr_gd
-	lr_nm
-	adam = False


USAGE (CAVE)
cave = CAVE(init params)
data = torch.rand(100)
mean = torch.Tensor([0.1])
var = torch.Tensor([0.05])
low = torch.zeros(1)
high = torch.ones(1)
output = cave(...) # arguments are forward method


USAGE (opt_all: standard gradient descent)
gsa = GSA()
gsb = GSB()
nsa = NSA()
nsb = NSB()
data = torch.rand(100)
mean = 0.1
var = 0.05
low = 0.0
high = 1.0

# See files for argument usage
gsa_out = gsa(...)
gsb_out = gsb(...)
nsa_out = nsa(...)
nsb_out = nsb(...)

if gradient descent:
	a -= lr_gd * gsa_out
	b -= lr_gd * gsb_out

elif newton's method:
	a -= lr_nm * nsa_out
	b -= lr_nm * nsb_out
'''

###