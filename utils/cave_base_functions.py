from abc import ABC, abstractmethod

import torch
from torch.nn.functional import softplus


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


	###################################
	#   f FUNCTIONS AND DERIVATIVES   #
	###################################

	# Forward methods
	def f(self, x, a, b):
		return self.fx(a * x + b)

	def df_da(self, x, a, b):
		return self.dfx(a * x + b) * x

	def df_db(self, x, a, b):
		return self.dfx(a * x + b)

	def d2f_da2(self, x, a, b):
		return self.d2fx(a * x + b) * x ** 2

	def d2f_dab(self, x, a, b):
		return self.d2fx(a * x + b) * x

	def d2f_db2(self, x, a, b):
		return self.d2fx(a * x + b)

	# Backward methods
	def df_dx(self, x, a, b):
		return self.dfx(a * x + b) * a

	def d2f_dax(self, x, a, b):
		y = a * x + b
		return self.d2fx(y) * a * x + self.dfx(y)

	def d2f_dbx(self, x, a, b):
		return self.d2fx(a * x + b) * a

	def d3f_da2x(self, x, a, b):
		y = a * x + b
		return self.d3fx(y) * a * (x ** 2) + 2 * x * self.d2fx(y)

	def d3f_dabx(self, x, a, b):
		y = a * x + b
		return self.d3fx(y) * a * x + self.d2fx(y)

	def d3f_db2x(self, x, a, b):
		return self.d3fx(a * x + b) * a


	####################################
	#   Em FUNCTIONS AND DERIVATIVES   #
	####################################

	# Forward methods
	def Em(self, x, a, b, mean, dim):
		return self.f(x, a, b).mean(**dim) - mean

	def dEm_da(self, x, a, b, dim):
		return self.df_da(x, a, b).mean(**dim)

	def dEm_db(self, x, a, b, dim):
		return self.df_db(x, a, b).mean(**dim)

	def d2Em_da2(self, x, a, b, dim):
		return self.d2f_da2(x, a, b).mean(**dim)

	def d2Em_dab(self, x, a, b, dim):
		return self.d2f_dab(x, a, b).mean(**dim)

	def d2Em_db2(self, x, a, b, dim):
		return self.d2f_db2(x, a, b).mean(**dim)

	# Backward methods
	def dEm_dx(self, x, a, b, dim):
		return self.df_dx(x, a, b) / self.numel(x.shape, dim)

	def d2Em_dax(self, x, a, b, dim):
		return self.d2f_dax(x, a, b) / self.numel(x.shape, dim)

	def d2Em_dbx(self, x, a, b, dim):
		return self.d2f_dbx(x, a, b) / self.numel(x.shape, dim)

	def d3Em_da2x(self, x, a, b, dim):
		return self.d3f_da2x(x, a, b) / self.numel(x.shape, dim)

	def d3Em_dabx(self, x, a, b, dim):
		return self.d3f_dabx(x, a, b) / self.numel(x.shape, dim)

	def d3Em_db2x(self, x, a, b, dim):
		return self.d3f_db2x(x, a, b) / self.numel(x.shape, dim)


	####################################
	#   Ev FUNCTIONS AND DERIVATIVES   #
	####################################

	# Forward methods
	def Ev(self, x, a, b, var, dim):
		f = self.f(x, a, b)
		return f.var(unbiased = False) - var

	def dEv_da(self, x, a, b, dim):
		f = self.f(x, a, b)
		df_da = self.df_da(x, a, b)
		return 2 * ((f * df_da).mean(**dim) - f.mean(**dim) * df_da.mean(**dim))

	def dEv_db(self, x, a, b, dim):
		f = self.f(x, a, b)
		df_db = self.df_db(x, a, b)
		return 2 * ((f * df_db).mean(**dim) - f.mean(**dim) * df_db.mean(**dim))

	def d2Ev_da2(self, x, a, b, dim):
		f = self.f(x, a, b)
		df_da = self.df_da(x, a, b)
		d2f_da2 = self.d2f_da2(x, a, b)
		return 2 * ((df_da ** 2 + f * d2f_da2).mean(**dim) - \
		            df_da.mean(**dim) ** 2 - f.mean(**dim) * d2f_da2.mean(**dim))

	def d2Ev_dab(self, x, a, b, dim):
		f = self.f(x, a, b)
		df_da = self.df_da(x, a, b)
		df_db = self.df_db(x, a, b)
		d2f_dab = self.d2f_dab(x, a, b)
		return 2 * ((df_da * df_db + f * d2f_dab).mean(**dim) - \
		            df_da.mean(**dim) * df_db.mean(**dim) - \
		            f.mean(**dim) * d2f_dab.mean(**dim))

	def d2Ev_db2(self, x, a, b, dim):
		f = self.f(x, a, b)
		df_db = self.df_db(x, a, b)
		d2f_db2 = self.d2f_db2(x, a, b)
		return 2 * ((df_db ** 2 + f * d2f_db2).mean(**dim) - \
		            df_db.mean(**dim) ** 2 - f.mean(**dim) * d2f_db2.mean(**dim))

	# Backward methods
	def dEv_dx(self, x, a, b, dim):
		f = self.f(x, a, b)
		df_dx = self.df_dx(x, a, b)
		N = self.numel(x.shape, dim)
		return 2 * df_dx / N * (f - f.mean())

	def d2Ev_dax(self, x, a, b, dim):
		f = self.f(x, a, b)
		df_da = self.df_da(x, a, b)
		df_dx = self.df_dx(x, a, b)
		d2f_dax = self.d2f_dax(x, a, b)
		N = self.numel(x.shape, dim)
		return 2 / N * ((df_da * df_dx + f * d2f_dax) - \
		                df_dx * df_da.mean(**dim) - f.mean(**dim) * d2f_dax)

	def d2Ev_dbx(self, x, a, b, dim):
		f = self.f(x, a, b)
		df_db = self.df_db(x, a, b)
		df_dx = self.df_dx(x, a, b)
		d2f_dbx = self.d2f_dbx(x, a, b)
		N = self.numel(x.shape, dim)
		return 2 / N * ((df_db * df_dx + f * d2f_dbx) - \
		                df_dx * df_db.mean(**dim) - f.mean(**dim) * d2f_dbx)

	def d3Ev_da2x(self, x, a, b, dim):
		f = self.f(x, a, b)
		df_da = self.df_da(x, a, b)
		df_dx = self.df_dx(x, a, b)
		d2f_da2 = self.d2f_da2(x, a, b)
		d2f_dax = self.d2f_dax(x, a, b)
		d3f_da2x = self.d3f_da2x(x, a, b)
		N = self.numel(x.shape, dim)
		return 2 / N * (2 * df_da * d2f_dax + df_dx * d2f_da2 + f * d3f_da2x - \
		                2 * df_da.mean(**dim) * d2f_dax - df_dx * d2f_da2.mean(**dim) - \
		                f.mean(**dim) * d3f_da2x)

	def d3Ev_dabx(self, x, a, b, dim):
		f = self.f(x, a, b)
		df_da = self.df_da(x, a, b)
		df_db = self.df_db(x, a, b)
		df_dx = self.df_dx(x, a, b)
		d2f_dab = self.d2f_dab(x, a, b)
		d2f_dax = self.d2f_dax(x, a, b)
		d2f_dbx = self.d2f_dbx(x, a, b)
		d3f_dabx = self.d3f_dabx(x, a, b)
		N = self.numel(x.shape, dim)
		return 2 / N * (d2f_dax * df_db + d2f_dbx * df_da + df_dx * d2f_dab + f * d3f_dabx - \
		                d2f_dax * df_db.mean(**dim) - d2f_dbx * df_da.mean(**dim) - \
		                df_dx * d2f_dab.mean(**dim) - d3f_dabx * f.mean(**dim))

	def d3Ev_db2x(self, x, a, b, dim):
		f = self.f(x, a, b)
		df_db = self.df_db(x, a, b)
		df_dx = self.df_dx(x, a, b)
		d2f_db2 = self.d2f_db2(x, a, b)
		d2f_dbx = self.d2f_dbx(x, a, b)
		d3f_db2x = self.d3f_db2x(x, a, b)
		N = self.numel(x.shape, dim)
		return 2 / N * (2 * df_db * d2f_dbx + df_dx * d2f_db2 + f * d3f_db2x - \
		                2 * df_db.mean(**dim) * d2f_dbx - df_dx * d2f_db2.mean(**dim) - \
		                f.mean(**dim) * d3f_db2x)


	####################################
	#   Lm FUNCTIONS AND DERIVATIVES   #
	####################################

	# Forward methods
	def Lm(self, x, a, b, mean, dim):
		return self.Em(x, a, b, mean, dim) ** 2

	def dLm_da(self, x, a, b, mean, dim):
		return 2 * self.Em(x, a, b, mean, dim) * self.dEm_da(x, a, b, dim)

	def dLm_db(self, x, a, b, mean, dim):
		return 2 * self.Em(x, a, b, mean, dim) * self.dEm_db(x, a, b, dim)

	def d2Lm_da2(self, x, a, b, mean, dim):
		Em = self.Em(x, a, b, mean, dim)
		dEm_da = self.dEm_da(x, a, b, dim)
		d2Em_da2 = self.d2Em_da2(x, a, b, dim)
		return 2 * (dEm_da ** 2 + Em * d2Em_da2)

	def d2Lm_dab(self, x, a, b, mean, dim):
		Em = self.Em(x, a, b, mean, dim)
		dEm_da = self.dEm_da(x, a, b, dim)
		dEm_db = self.dEm_db(x, a, b, dim)
		d2Em_dab = self.d2Em_dab(x, a, b, dim)
		return 2 * (dEm_da * dEm_db + Em * d2Em_dab)

	def d2Lm_db2(self, x, a, b, mean, dim):
		Em = self.Em(x, a, b, mean, dim)
		dEm_db = self.dEm_db(x, a, b, dim)
		d2Em_db2 = self.d2Em_db2(x, a, b, dim)
		return 2 * (dEm_db ** 2 + Em * d2Em_db2)

	# Backward methods
	def dLm_dx(self, x, a, b, mean, dim):
		return 2 * self.Em(x, a, b, mean, dim) * self.dEm_dx(x, a, b, dim)

	def d2Lm_dax(self, x, a, b, mean, dim):
		Em = self.Em(x, a, b, mean, dim)
		dEm_da = self.dEm_da(x, a, b, dim)
		dEm_dx = self.dEm_dx(x, a, b, dim)
		d2Em_dax = self.d2Em_dax(x, a, b, dim)
		return 2 * (dEm_da * dEm_dx + Em * d2Em_dax)

	def d2Lm_dbx(self, x, a, b, mean, dim):
		Em = self.Em(x, a, b, mean, dim)
		dEm_db = self.dEm_db(x, a, b, dim)
		dEm_dx = self.dEm_dx(x, a, b, dim)
		d2Em_dbx = self.d2Em_dbx(x, a, b, dim)
		return 2 * (dEm_db * dEm_dx + Em * d2Em_dbx)

	def d3Lm_da2x(self, x, a, b, mean, dim):
		Em = self.Em(x, a, b, mean, dim)
		dEm_da = self.dEm_da(x, a, b, dim)
		dEm_dx = self.dEm_dx(x, a, b, dim)
		d2Em_da2 = self.d2Em_da2(x, a, b, dim)
		d2Em_dax = self.d2Em_dax(x, a, b, dim)
		d3Em_da2x = self.d3Em_da2x(x, a, b, dim)
		return 2 * (2 * dEm_da * d2Em_dax + dEm_dx * d2Em_da2 + Em * d3Em_da2x)

	def d3Lm_dabx(self, x, a, b, mean, dim):
		Em = self.Em(x, a, b, mean, dim)
		dEm_da = self.dEm_da(x, a, b, dim)
		dEm_db = self.dEm_db(x, a, b, dim)
		dEm_dx = self.dEm_dx(x, a, b, dim)
		d2Em_dab = self.d2Em_dab(x, a, b, dim)
		d2Em_dax = self.d2Em_dax(x, a, b, dim)
		d2Em_dbx = self.d2Em_dbx(x, a, b, dim)
		d3Em_dabx = self.d3Em_dabx(x, a, b, dim)
		return 2 * (dEm_da * d2Em_dbx + dEm_db * d2Em_dax + dEm_dx * d2Em_dab + Em * d3Em_dabx)

	def d3Lm_db2x(self, x, a, b, mean, dim):
		Em = self.Em(x, a, b, mean, dim)
		dEm_db = self.dEm_db(x, a, b, dim)
		dEm_dx = self.dEm_dx(x, a, b, dim)
		d2Em_db2 = self.d2Em_db2(x, a, b, dim)
		d2Em_dbx = self.d2Em_dbx(x, a, b, dim)
		d3Em_db2x = self.d3Em_db2x(x, a, b, dim)
		return 2 * (2 * dEm_db * d2Em_dbx + dEm_dx * d2Em_db2 + Em * d3Em_db2x)


	####################################
	#   Lv FUNCTIONS AND DERIVATIVES   #
	####################################

	# Forward methods
	def Lv(self, x, a, b, var, dim):
		return self.Ev(x, a, b, var, dim) ** 2

	def dLv_da(self, x, a, b, var, dim):
		return 2 * self.Ev(x, a, b, var, dim) * self.dEv_da(x, a, b, dim)

	def dLv_db(self, x, a, b, var, dim):
		return 2 * self.Ev(x, a, b, var, dim) * self.dEv_db(x, a, b, dim)

	def d2Lv_da2(self, x, a, b, var, dim):
		Ev = self.Ev(x, a, b, var, dim)
		dEv_da = self.dEv_da(x, a, b, dim)
		d2Ev_da2 = self.d2Ev_da2(x, a, b, dim)
		return 2 * (dEv_da ** 2 + Ev * d2Ev_da2)

	def d2Lv_dab(self, x, a, b, var, dim):
		Ev = self.Ev(x, a, b, var, dim)
		dEv_da = self.dEv_da(x, a, b, dim)
		dEv_db = self.dEv_db(x, a, b, dim)
		d2Ev_dab = self.d2Ev_dab(x, a, b, dim)
		return 2 * (dEv_da * dEv_db + Ev * d2Ev_dab)

	def d2Lv_db2(self, x, a, b, var, dim):
		Ev = self.Ev(x, a, b, var, dim)
		dEv_db = self.dEv_db(x, a, b, dim)
		d2Ev_db2 = self.d2Ev_db2(x, a, b, dim)
		return 2 * (dEv_db ** 2 + Ev * d2Ev_db2)

	# Backward methods
	def dLv_dx(self, x, a, b, var, dim):
		return 2 * self.Ev(x, a, b, var, dim) * self.dEv_dx(x, a, b, dim)

	def d2Lv_dax(self, x, a, b, var, dim):
		Ev = self.Ev(x, a, b, var, dim)
		dEv_da = self.dEv_da(x, a, b, dim)
		dEv_dx = self.dEv_dx(x, a, b, dim)
		d2Ev_dax = self.d2Ev_dax(x, a, b, dim)
		return 2 * (dEv_da * dEv_dx + Ev * d2Ev_dax)

	def d2Lv_dbx(self, x, a, b, var, dim):
		Ev = self.Ev(x, a, b, var, dim)
		dEv_db = self.dEv_db(x, a, b, dim)
		dEv_dx = self.dEv_dx(x, a, b, dim)
		d2Ev_dbx = self.d2Ev_dbx(x, a, b, dim)
		return 2 * (dEv_db * dEv_dx + Ev * d2Ev_dbx)

	def d3Lv_da2x(self, x, a, b, var, dim):
		Ev = self.Ev(x, a, b, var, dim)
		dEv_da = self.dEv_da(x, a, b, dim)
		dEv_dx = self.dEv_dx(x, a, b, dim)
		d2Ev_da2 = self.d2Ev_da2(x, a, b, dim)
		d2Ev_dax = self.d2Ev_dax(x, a, b, dim)
		d3Ev_da2x = self.d3Ev_da2x(x, a, b, dim)
		return 2 * (2 * dEv_da * d2Ev_dax + dEv_dx * d2Ev_da2 + Ev * d3Ev_da2x)

	def d3Lv_dabx(self, x, a, b, var, dim):
		Ev = self.Ev(x, a, b, var, dim)
		dEv_da = self.dEv_da(x, a, b, dim)
		dEv_db = self.dEv_db(x, a, b, dim)
		dEv_dx = self.dEv_dx(x, a, b, dim)
		d2Ev_dab = self.d2Ev_dab(x, a, b, dim)
		d2Ev_dax = self.d2Ev_dax(x, a, b, dim)
		d2Ev_dbx = self.d2Ev_dbx(x, a, b, dim)
		d3Ev_dabx = self.d3Ev_dabx(x, a, b, dim)
		return 2 * (dEv_da * d2Ev_dbx + dEv_db * d2Ev_dax + dEv_dx * d2Ev_dab + Ev * d3Ev_dabx)

	def d3Lv_db2x(self, x, a, b, var, dim):
		Ev = self.Ev(x, a, b, var, dim)
		dEv_db = self.dEv_db(x, a, b, dim)
		dEv_dx = self.dEv_dx(x, a, b, dim)
		d2Ev_db2 = self.d2Ev_db2(x, a, b, dim)
		d2Ev_dbx = self.d2Ev_dbx(x, a, b, dim)
		d3Ev_db2x = self.d3Ev_db2x(x, a, b, dim)
		return 2 * (2 * dEv_db * d2Ev_dbx + dEv_dx * d2Ev_db2 + Ev * d3Ev_db2x)


	##################################################
	#   GRADIENT DESCENT FUNCTIONS AND DERIVATIVES   #
	##################################################

	# Forward methods
	def Ga(self, x, a, b, var, dim, lr):
		return lr * self.dLv_da(x, a, b, var, dim)

	def Gb(self, x, a, b, mean, dim, lr):
		return lr * self.dLm_db(x, a, b, mean, dim)

	def Gab(self, x, a, b, mean, var, dim, lr):
		dL_da = self.dLm_da(x, a, b, mean, dim) + self.dLv_da(x, a, b, var, dim)
		dL_db = self.dLm_db(x, a, b, mean, dim) + self.dLv_db(x, a, b, var, dim)
		return lr * dL_da, lr * dL_db

	# Backward methods
	def dGa_dx(self, x, a, b, var, dim, lr):
		return lr * self.d2Lv_dax(x, a, b, var, dim)

	def dGb_dx(self, x, a, b, mean, dim, lr):
		return lr * self.d2Lm_dbx(x, a, b, mean, dim)

	def dGab_dx(self, x, a, b, mean, var, dim, lr):
		dGa_dx = self.d2Lm_dax(x, a, b, mean, dim) + self.d2Lv_dax(x, a, b, var, dim)
		dGb_dx = self.d2Lm_dbx(x, a, b, mean, dim) + self.d2Lv_dbx(x, a, b, var, dim)
		return lr * dGa_dx, lr * dGb_dx


	#################################################
	#   NEWTON'S METHOD FUNCTIONS AND DERIVATIVES   #
	#################################################

	# Forward methods
	def Na(self, x, a, b, var, dim, lr):
		return lr * self.dLv_da(x, a, b, var, dim) / self.d2Lv_da2(x, a, b, var, dim)

	def Nb(self, x, a, b, mean, dim, lr):
		return lr * self.dLm_db(x, a, b, mean, dim) / self.d2Lm_db2(x, a, b, mean, dim)

	def Nab(self, x, a, b, mean, var, dim, lr):
		dL_da = self.dLm_da(x, a, b, mean, dim) + self.dLv_da(x, a, b, var, dim)
		dL_db = self.dLm_db(x, a, b, mean, dim) + self.dLv_db(x, a, b, var, dim)
		d2L_da2 = self.d2Lm_da2(x, a, b, mean, dim) + self.d2Lv_da2(x, a, b, var, dim)
		d2L_dab = self.d2Lm_dab(x, a, b, mean, dim) + self.d2Lv_dab(x, a, b, var, dim)
		d2L_db2 = self.d2Lm_db2(x, a, b, mean, dim) + self.d2Lv_db2(x, a, b, var, dim)

		Na = dL_da * d2L_db2 - dL_db * d2L_dab
		Nb = dL_db * d2L_da2 - dL_da * d2L_dab
		D = d2L_da2 * d2L_db2 - d2L_dab ** 2
		return lr * Na / D, lr * Nb / D

	# Backward methods
	def dNa_dx(self, x, a, b, var, dim, lr):
		dLv_da = self.dLv_da(x, a, b, var, dim)
		d2Lv_da2 = self.d2Lv_da2(x, a, b, var, dim)
		d2Lv_dax = self.d2Lv_dax(x, a, b, var, dim)
		d3Lv_da2x = self.d3Lv_da2x(x, a, b, var, dim)
		return lr * (d2Lv_da2 * d2Lv_dax - dLv_da * d3Lv_da2x) / (d2Lv_da2 ** 2)

	def dNb_dx(self, x, a, b, mean, dim, lr):
		dLm_db = self.dLm_db(x, a, b, mean, dim)
		d2Lm_db2 = self.d2Lm_db2(x, a, b, mean, dim)
		d2Lm_dbx = self.d2Lm_dbx(x, a, b, mean, dim)
		d3Lm_db2x = self.d3Lm_db2x(x, a, b, mean, dim)
		return lr * (d2Lm_db2 * d2Lm_dbx - dLm_db * d3Lm_db2x) / (d2Lm_db2 ** 2)

	def dNab_dx(self, x, a, b, mean, var, dim, lr):
		dL_da = self.dLm_da(x, a, b, mean, dim) + self.dLv_da(x, a, b, var, dim)
		dL_db = self.dLm_db(x, a, b, mean, dim) + self.dLv_db(x, a, b, var, dim)
		d2L_da2 = self.d2Lm_da2(x, a, b, mean, dim) + self.d2Lv_da2(x, a, b, var, dim)
		d2L_dab = self.d2Lm_dab(x, a, b, mean, dim) + self.d2Lv_dab(x, a, b, var, dim)
		d2L_db2 = self.d2Lm_db2(x, a, b, mean, dim) + self.d2Lv_db2(x, a, b, var, dim)
		d2L_dax = self.d2Lm_dax(x, a, b, mean, dim) + self.d2Lv_dax(x, a, b, var, dim)
		d2L_dbx = self.d2Lm_dbx(x, a, b, mean, dim) + self.d2Lv_dbx(x, a, b, var, dim)
		d3L_da2x = self.d3Lm_da2x(x, a, b, mean, dim) + self.d3Lv_da2x(x, a, b, var, dim)
		d3L_dabx = self.d3Lm_dabx(x, a, b, mean, dim) + self.d3Lv_dabx(x, a, b, var, dim)
		d3L_db2x = self.d3Lm_db2x(x, a, b, mean, dim) + self.d3Lv_db2x(x, a, b, var, dim)

		Na = dL_da * d2L_db2 - dL_db * d2L_dab
		Nb = dL_db * d2L_da2 - dL_da * d2L_dab
		D = d2L_da2 * d2L_db2 - d2L_dab ** 2

		dNa_dx = d2L_dax * d2L_db2 + dL_da * d3L_db2x - d2L_dbx * d2L_dab - dL_db * d3L_dabx
		dNb_dx = d2L_dbx * d2L_da2 + dL_db * d3L_da2x - d2L_dax * d2L_dab - dL_da * d3L_dabx
		dD_dx = d3L_da2x * d2L_db2 + d3L_db2x * d2L_da2 - 2 * d2L_dab * d3L_dabx

		dNa_dx = (D * dNa_dx - Na * dD_dx) / (D ** 2)
		dNb_dx = (D * dNb_dx - Nb * dD_dx) / (D ** 2)

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


###