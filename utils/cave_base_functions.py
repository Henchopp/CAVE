from abc import ABC, abstractmethod

import torch
from torch.nn.functional import softplus


class CAVEBaseFunction(ABC):
	"""docstring for CAVEBaseFunction"""

	def __init__(self, low = None, high = None):
		self.low = low
		self.high = high

	# User-implemented function and derivatives
	@abstractmethod
	def fx(self, x: torch.Tensor) -> torch.Tensor:
		raise NotImplementedError("You must implement the function for your custom "
		                          "CAVEBaseFunction.")

	@abstractmethod
	def dfx(self, x: torch.Tensor) -> torch.Tensor:
		raise NotImplementedError("You must implement the first derivative for your custom "
		                          "CAVEBaseFunction.")

	@abstractmethod
	def d2fx(self, x: torch.Tensor) -> torch.Tensor:
		raise NotImplementedError("You must implement the second derivative for your custom "
		                          "CAVEBaseFunction.")

	@abstractmethod
	def d3fx(self, x: torch.Tensor) -> torch.Tensor:
		raise NotImplementedError("You must implement the third derivative for your custom "
		                          "CAVEBaseFunction.")


	# Forward functions
	def f(self, x: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
		return self.fx(a * x + b)

	def df_da(self, x: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
		return self.dfx(a * x + b) * x

	def df_db(self, x: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
		return self.dfx(a * x + b)

	def d2f_da2(self, x: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
		return self.d2fx(a * x + b) * x ** 2

	def d2f_dab(self, x: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
		return self.d2fx(a * x + b) * x

	def d2f_db2(self, x: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
		return self.d2fx(a * x + b)


	# Derivatives w.r.t. x
	def df_dx(self, x: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
		return self.dfx(a * x + b) * a

	def d2f_dax(self, x: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
		y = a * x + b
		return self.d2fx(y) * a * x + self.dfx(y)

	def d2f_dbx(self, x: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
		return self.d2fx(a * x + b) * a

	def d3f_da2x(self, x: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
		y = a * x + b
		return self.d3fx(y) * a * x ** 2 + 2 * self.d2fx(y) * x

	def d3f_dabx(self, x: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
		y = a * x + b
		return self.d3fx(y) * a * x + self.d2fx(y)

	def d3f_db2x(self, x: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
		return self.d3fx(a * x + b) * a


# Sigmoid function
class Sigmoid(CAVEBaseFunction):
	"""docstring for Sigmoid"""

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