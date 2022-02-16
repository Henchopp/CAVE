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


	##################################################
	#   GRADIENT DESCENT FUNCTIONS AND DERIVATIVES   #
	##################################################

	# Forward methods
	def Ga(self, x, a, b, var, dim, lr):
		f = self.fx(a * x + b)
		df_da = self.dfx(a * x + b) * x

		Ev = f.var(unbiased = False, **dim) - var
		dEv_da = 2 * ((f * df_da).mean(**dim) - f.mean(**dim) * df_da.mean(**dim))

		dLv_da = 2 * Ev * dEv_da
		return lr * dLv_da

	def Gb(self, x, a, b, mean, dim, lr):
		f = self.fx(a * x + b)
		df_db = self.dfx(a * x + b)

		Em = f.mean(**dim) - mean
		dEm_db = df_db.mean(**dim)

		dLm_db = 2 * Em * dEm_db

		return lr * dLm_db

	def Gab(self, x, a, b, mean, var, dim, lr):
		f = self.fx(a * x + b)
		dfx = self.dfx(a * x + b)
		fmean = f.mean(**dim)

		df_da = dfx * x
		df_db = dfx

		Em = fmean - mean
		dEm_da = df_da.mean(**dim)
		dEm_db = df_db.mean(**dim)

		Ev = f.var(unbiased = False, **dim) - var
		dEv_da = 2 * ((f * df_da).mean(**dim) - fmean * dEm_da)
		dEv_db = 2 * ((f * df_db).mean(**dim) - fmean * dEm_db)

		dLm_da = 2 * Em * dEm_da
		dLv_da = 2 * Ev * dEv_da
		dLm_db = 2 * Em * dEm_db
		dLv_db = 2 * Ev * dEv_db

		dL_da = dLm_da + dLv_da
		dL_db = dLm_db + dLv_db

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
		df_da_mean = df_da.mean(**dim)

		Ev = f.var(unbiased = False, **dim) - var
		dEv_da = 2 * ((f * df_da).mean(**dim) - fmean * df_da_mean)
		dEv_dx = 2 * df_dx / N * (f - fmean)
		d2Ev_dax = 2 / N * ((df_da * df_dx + f * d2f_dax) - \
		                    df_dx * df_da_mean - fmean * d2f_dax)

		d2Lv_dax = 2 * (dEv_da * dEv_dx + Ev * d2Ev_dax)
		
		return lr * d2Lv_dax

	def dGb_dx(self, x, a, b, mean, dim, lr):
		N = self.numel(x.shape, dim)

		f = self.fx(a * x + b)
		dfx = self.dfx(a * x + b)

		df_db = dfx
		df_dx = dfx * a
		d2f_dbx = self.d2fx(a * x + b) * a

		Em = f.mean(**dim) - mean
		dEm_db = df_db.mean(**dim)
		dEm_dx = df_dx / N
		d2Em_dbx = d2f_dbx / N

		d2Lm_dbx = 2 * (dEm_db * dEm_dx + Em * d2Em_dbx)

		return lr * d2Lm_dbx

	def dGab_dx(self, x, a, b, mean, var, dim, lr):
		N = self.numel(x.shape, dim)

		f = self.fx(a * x + b)
		dfx = self.dfx(a * x + b)
		d2fx = self.d2fx(a * x + b)
		fmean = f.mean(**dim)

		df_da = dfx * x
		df_db = dfx
		df_dx = dfx * a
		d2f_dax = d2fx * a * x + dfx
		d2f_dbx = d2fx * a

		Em = fmean - mean
		dEm_da = df_da.mean(**dim)
		dEm_db = df_db.mean(**dim)
		dEm_dx = df_dx / N
		d2Em_dax = d2f_dax / N
		d2Em_dbx = d2f_dbx / N

		Ev = f.var(unbiased = False, **dim) - var
		dEv_da = 2 * ((f * df_da).mean(**dim) - fmean * dEm_da)
		dEv_db = 2 * ((f * df_db).mean(**dim) - fmean * dEm_db)
		dEv_dx = 2 * df_dx / N * (f - fmean)
		d2Ev_dax = 2 / N * ((df_da * df_dx + f * d2f_dax) - \
		                    df_dx * dEm_da - fmean * d2f_dax)
		d2Ev_dbx = 2 / N * ((df_db * df_dx + f * d2f_dbx) - \
		                    df_dx * dEm_db - fmean * d2f_dbx)

		d2Lm_dax = 2 * (dEm_da * dEm_dx + Em * d2Em_dax)
		d2Lm_dbx = 2 * (dEm_db * dEm_dx + Em * d2Em_dbx)

		d2Lv_dax = 2 * (dEv_da * dEv_dx + Ev * d2Ev_dax)
		d2Lv_dbx = 2 * (dEv_db * dEv_dx + Ev * d2Ev_dbx)

		dGa_dx = d2Lm_dax + d2Lv_dax
		dGb_dx = d2Lm_dbx + d2Lv_dbx

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
		df_da_mean = df_da.mean(**dim)

		Ev = f.var(unbiased = False, **dim) - var
		dEv_da = 2 * ((f * df_da).mean(**dim) - fmean * df_da_mean)
		d2Ev_da2 = 2 * ((df_da ** 2 + f * d2f_da2).mean(**dim) - \
		                df_da_mean ** 2 - fmean * d2f_da2.mean(**dim))

		dLv_da = 2 * Ev * dEv_da
		d2Lv_da2 = 2 * (dEv_da ** 2 + Ev * d2Ev_da2)

		return lr * dLv_da / d2Lv_da2

	def Nb(self, x, a, b, mean, dim, lr):
		f = self.fx(a * x + b)
		df_db = self.dfx(a * x + b)
		d2f_db2 = self.d2fx(a * x + b)

		Em = f.mean(**dim) - mean
		dEm_db = df_db.mean(**dim)
		d2Em_db2 = d2f_db2.mean(**dim)

		dLm_db = 2 * Em * dEm_db
		d2Lm_db2 = 2 * (dEm_db ** 2 + Em * d2Em_db2)

		return lr * dLm_db / d2Lm_db2

	def Nab(self, x, a, b, mean, var, dim, lr):
		f = self.fx(a * x + b)
		dfx = self.dfx(a * x + b)
		d2fx = self.d2fx(a * x + b)

		fmean = f.mean(**dim)

		df_da = dfx * x
		df_db = dfx
		d2f_da2 = d2fx * x ** 2
		d2f_dab = d2fx * x
		d2f_db2 = d2fx

		df_da_mean = df_da.mean(**dim)
		df_db_mean = df_db.mean(**dim)

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
		                df_da_mean ** 2 - fmean * d2Em_da2)
		d2Ev_dab = 2 * ((df_da * df_db + f * d2f_dab).mean(**dim) - \
		                df_da_mean * df_db_mean - fmean * d2Em_dab)
		d2Ev_db2 = 2 * ((df_db ** 2 + f * d2f_db2).mean(**dim) - \
		                df_db_mean ** 2 - fmean * d2Em_db2)

		dLm_da = 2 * Em * dEm_da
		dLm_db = 2 * Em * dEm_db
		d2Lm_da2 = 2 * (dEm_da ** 2 + Em * d2Em_da2)
		d2Lm_dab = 2 * (dEm_da * dEm_db + Em * d2Em_dab)
		d2Lm_db2 = 2 * (dEm_db ** 2 + Em * d2Em_db2)

		dLv_da = 2 * Ev * dEv_da
		dLv_db = 2 * Ev * dEv_db
		d2Lv_da2 = 2 * (dEv_da ** 2 + Ev * d2Ev_da2)
		d2Lv_dab = 2 * (dEv_da * dEv_db + Ev * d2Ev_dab)
		d2Lv_db2 = 2 * (dEv_db ** 2 + Ev * d2Ev_db2)

		dL_da = dLm_da + dLv_da
		dL_db = dLm_db + dLv_db
		d2L_da2 = d2Lm_da2 + d2Lv_da2
		d2L_dab = d2Lm_dab + d2Lv_dab
		d2L_db2 = d2Lm_db2 + d2Lv_db2

		Na = dL_da * d2L_db2 - dL_db * d2L_dab
		Nb = dL_db * d2L_da2 - dL_da * d2L_dab
		D = d2L_da2 * d2L_db2 - d2L_dab ** 2

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

		df_da_mean = df_da.mean(**dim)
		d2f_da2_mean = d2f_da2.mean(**dim)

		Ev = f.var(unbiased = False, **dim) - var
		dEv_da = 2 * ((f * df_da).mean(**dim) - fmean * df_da_mean)
		dEv_dx = 2 * df_dx / N * (f - fmean)
		d2Ev_da2 = 2 * ((df_da ** 2 + f * d2f_da2).mean(**dim) - \
		                df_da_mean ** 2 - fmean * d2f_da2_mean)
		d2Ev_dax = 2 / N * ((df_da * df_dx + f * d2f_dax) - \
		                    df_dx * df_da_mean - fmean * d2f_dax)
		d3Ev_da2x = 2 / N * (2 * df_da * d2f_dax + df_dx * d2f_da2 + f * d3f_da2x - \
		                     2 * df_da_mean * d2f_dax - df_dx * d2f_da2_mean - \
		                     fmean * d3f_da2x)

		dLv_da = 2 * Ev * dEv_da
		d2Lv_da2 = 2 * (dEv_da ** 2 + Ev * d2Ev_da2)
		d2Lv_dax = 2 * (dEv_da * dEv_dx + Ev * d2Ev_dax)
		d3Lv_da2x = 2 * (2 * dEv_da * d2Ev_dax + dEv_dx * d2Ev_da2 + Ev * d3Ev_da2x)

		return lr * (d2Lv_da2 * d2Lv_dax - dLv_da * d3Lv_da2x) / (d2Lv_da2 ** 2)

	def dNb_dx(self, x, a, b, mean, dim, lr):
		N = self.numel(x.shape, dim)

		f = self.fx(a * x + b)
		dfx = self.dfx(a * x + b)
		d2fx = self.d2fx(a * x + b)

		df_db = dfx
		df_dx = dfx * a
		d2f_db2 = d2fx
		d2f_dbx = d2fx * a
		d3f_db2x = self.d3fx(a * x + b) * a

		Em = f.mean(**dim) - mean
		dEm_db = df_db.mean(**dim)
		dEm_dx = df_dx / N
		d2Em_db2 = d2f_db2.mean(**dim)
		d2Em_dbx = d2f_dbx / N
		d3Em_db2x = d3f_db2x / N

		dLm_db = 2 * Em * dEm_db
		d2Lm_db2 = 2 * (dEm_db ** 2 + Em * d2Em_db2)
		d2Lm_dbx = 2 * (dEm_db * dEm_dx + Em * d2Em_dbx)
		d3Lm_db2x = 2 * (2 * dEm_db * d2Em_dbx + dEm_dx * d2Em_db2 + Em * d3Em_db2x)

		return lr * (d2Lm_db2 * d2Lm_dbx - dLm_db * d3Lm_db2x) / (d2Lm_db2 ** 2)

	def dNab_dx(self, x, a, b, mean, var, dim, lr):
		N = self.numel(x.shape, dim)

		f = self.fx(a * x + b)
		dfx = self.dfx(a * x + b)
		d2fx = self.d2fx(a * x + b)
		d3fx = self.d3fx(a * x + b)

		fmean = f.mean(**dim)

		df_da = dfx * x
		df_db = dfx
		df_dx = dfx * a
		d2f_da2 = d2fx * x ** 2
		d2f_dab = d2fx * x
		d2f_db2 = d2fx
		d2f_dax = d2fx * a * x + dfx
		d2f_dbx = d2fx * a
		d3f_da2x = d3fx * a * (x ** 2) + 2 * x * d2fx
		d3f_dabx = d3fx * a * x + d2fx
		d3f_db2x = d3fx * a

		df_da_mean = df_da.mean(**dim)
		df_db_mean = df_db.mean(**dim)
		d2f_da2_mean = d2f_da2.mean(**dim)
		d2f_dab_mean = d2f_dab.mean(**dim)
		d2f_db2_mean = d2f_db2.mean(**dim)

		Em = fmean - mean
		dEm_da = df_da_mean
		dEm_db = df_db_mean
		dEm_dx = df_dx / N
		d2Em_da2 = d2f_da2_mean
		d2Em_dab = d2f_dab_mean
		d2Em_db2 = d2f_db2_mean
		d2Em_dax = d2f_dax / N
		d2Em_dbx = d2f_dbx / N
		d3Em_da2x = d3f_da2x / N
		d3Em_dabx = d3f_dabx / N
		d3Em_db2x = d3f_db2x / N

		Ev = f.var(unbiased = False, **dim) - var
		dEv_da = 2 * ((f * df_da).mean(**dim) - fmean * df_da_mean)
		dEv_db = 2 * ((f * df_db).mean(**dim) - fmean * df_db_mean)
		dEv_dx = 2 * df_dx / N * (f - fmean)
		d2Ev_da2 = 2 * ((df_da ** 2 + f * d2f_da2).mean(**dim) - \
		                df_da_mean ** 2 - fmean * d2f_da2_mean)
		d2Ev_dab = 2 * ((df_da * df_db + f * d2f_dab).mean(**dim) - \
		                df_da_mean * df_db_mean - \
		                fmean * d2f_dab_mean)
		d2Ev_db2 = 2 * ((df_db ** 2 + f * d2f_db2).mean(**dim) - \
		                df_db_mean ** 2 - fmean * d2f_db2_mean)
		d2Ev_dax = 2 / N * ((df_da * df_dx + f * d2f_dax) - \
		                    df_dx * df_da_mean - fmean * d2f_dax)
		d2Ev_dbx = 2 / N * ((df_db * df_dx + f * d2f_dbx) - \
		                    df_dx * df_db_mean - fmean * d2f_dbx)
		d3Ev_da2x = 2 / N * (2 * df_da * d2f_dax + df_dx * d2f_da2 + f * d3f_da2x - \
		                     2 * df_da_mean * d2f_dax - df_dx * d2f_da2_mean - \
		                     fmean * d3f_da2x)
		d3Ev_dabx = 2 / N * (d2f_dax * df_db + d2f_dbx * df_da + df_dx * d2f_dab + f * d3f_dabx - \
		                     d2f_dax * df_db_mean - d2f_dbx * df_da_mean - \
		                     df_dx * d2f_dab_mean - d3f_dabx * fmean)
		d3Ev_db2x = 2 / N * (2 * df_db * d2f_dbx + df_dx * d2f_db2 + f * d3f_db2x - \
		                     2 * df_db_mean * d2f_dbx - df_dx * d2f_db2_mean - \
		                     fmean * d3f_db2x)

		dLm_da = 2 * Em * dEm_da
		dLm_db = 2 * Em * dEm_db
		d2Lm_da2 = 2 * (dEm_da ** 2 + Em * d2Em_da2)
		d2Lm_dab = 2 * (dEm_da * dEm_db + Em * d2Em_dab)
		d2Lm_db2 = 2 * (dEm_db ** 2 + Em * d2Em_db2)
		d2Lm_dax = 2 * (dEm_da * dEm_dx + Em * d2Em_dax)
		d2Lm_dbx = 2 * (dEm_db * dEm_dx + Em * d2Em_dbx)
		d3Lm_da2x = 2 * (2 * dEm_da * d2Em_dax + dEm_dx * d2Em_da2 + Em * d3Em_da2x)
		d3Lm_dabx = 2 * (dEm_da * d2Em_dbx + dEm_db * d2Em_dax + dEm_dx * d2Em_dab + Em * d3Em_dabx)
		d3Lm_db2x = 2 * (2 * dEm_db * d2Em_dbx + dEm_dx * d2Em_db2 + Em * d3Em_db2x)

		dLv_da = 2 * Ev * dEv_da
		dLv_db = 2 * Ev * dEv_db
		d2Lv_da2 = 2 * (dEv_da ** 2 + Ev * d2Ev_da2)
		d2Lv_dab = 2 * (dEv_da * dEv_db + Ev * d2Ev_dab)
		d2Lv_db2 = 2 * (dEv_db ** 2 + Ev * d2Ev_db2)
		d2Lv_dax = 2 * (dEv_da * dEv_dx + Ev * d2Ev_dax)
		d2Lv_dbx = 2 * (dEv_db * dEv_dx + Ev * d2Ev_dbx)
		d3Lv_da2x = 2 * (2 * dEv_da * d2Ev_dax + dEv_dx * d2Ev_da2 + Ev * d3Ev_da2x)
		d3Lv_dabx = 2 * (dEv_da * d2Ev_dbx + dEv_db * d2Ev_dax + dEv_dx * d2Ev_dab + Ev * d3Ev_dabx)
		d3Lv_db2x = 2 * (2 * dEv_db * d2Ev_dbx + dEv_dx * d2Ev_db2 + Ev * d3Ev_db2x)

		dL_da = dLm_da + dLv_da
		dL_db = dLm_db + dLv_db
		d2L_da2 = d2Lm_da2 + d2Lv_da2
		d2L_dab = d2Lm_dab + d2Lv_dab
		d2L_db2 = d2Lm_db2 + d2Lv_db2
		d2L_dax = d2Lm_dax + d2Lv_dax
		d2L_dbx = d2Lm_dbx + d2Lv_dbx
		d3L_da2x = d3Lm_da2x + d3Lv_da2x
		d3L_dabx = d3Lm_dabx + d3Lv_dabx
		d3L_db2x = d3Lm_db2x + d3Lv_db2x

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